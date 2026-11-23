"""Hand-agnostic grasp similarity factors.

F1 hand shape        : per-finger features in a palm frame, normalised by palm length.
F3 hand-side contact : per-part soft contact vector, compared with soft IoU.
F4 object-side contact: object contact map (all hand) + per-finger-group contact maps, compared with soft IoU.
Contacts use the GenDexGrasp aligned distance d * exp(2 (1 - |cos|)); |cos| because OakInk meshes have
inconsistent winding, so normal orientation is unreliable.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .hands import FINGERS, GROUPS, PARTS, HandState, part_group


@dataclass
class FactorConfig:
    sigma_tip: float = 0.15  # tip position, in palm lengths
    sigma_angle: float = 0.26  # rad (~15 deg)
    sigma_curl: float = 0.10  # |tip - base| / chain length
    sigma_aperture: float = 0.15  # thumb-tip to finger-tip distance, in palm lengths
    f1_temp: float = 2.0
    sigma_hand_contact: float = 0.005  # m
    sigma_obj_contact: float = 0.02  # m
    obj_channels: str = "part"  # per-object-point contact channels: "part" (16) or "group" (palm + 5 fingers)
    w1: float = 0.5
    w3: float = 0.0  # F3 is (almost) the marginal of the per-part object map F4b, so it is off by default
    w4: float = 0.5


def palm_frame(s: HandState, sign: float):
    nonthumb = [f for f in FINGERS if f in s.fingers and f != "thumb"]
    bases = torch.stack([s.fingers[f][:, 0] for f in nonthumb], 1)
    y = bases.mean(1) - s.wrist
    L = y.norm(dim=-1)
    y = F.normalize(y, dim=-1)
    across = s.fingers[nonthumb[0]][:, 0] - s.fingers[nonthumb[-1]][:, 0]
    z = F.normalize(torch.cross(across, y, dim=-1), dim=-1) * sign
    x = torch.cross(y, z, dim=-1)
    return torch.stack([x, y, z], -1), L  # R: B,3,3 (columns = axes), L: B


def shape_features(s: HandState, sign: float, fingers):
    """Returns dict name -> (B, k) feature tensor, and the sigma group of each feature."""
    R, L = palm_frame(s, sign)
    local = lambda p: torch.einsum("bij,bi->bj", R, p - s.wrist) / L[:, None]
    feats = {}
    for f in fingers:
        c = s.fingers[f]
        base, tip = c[:, 0], c[:, -1]
        d = torch.einsum("bij,bi->bj", R, F.normalize(tip - base, dim=-1))
        feats[f"{f}.tip"] = (local(tip), "tip")
        feats[f"{f}.angle"] = (torch.stack([torch.asin(d[:, 2].clamp(-1, 1)), torch.atan2(d[:, 0], d[:, 1])], 1),
                               "angle")
        chain_len = (c[:, 1:] - c[:, :-1]).norm(dim=-1).sum(1)
        feats[f"{f}.curl"] = (((tip - base).norm(dim=-1) / chain_len)[:, None], "curl")
        if f != "thumb" and "thumb" in fingers:
            feats[f"{f}.aperture"] = (((s.fingers["thumb"][:, -1] - tip).norm(dim=-1) / L)[:, None], "aperture")
    return feats


def f1_shape(s, s_ref, sign, sign_ref, cfg: FactorConfig):
    fingers = [f for f in FINGERS if f in s.fingers and f in s_ref.fingers]
    a, b = shape_features(s, sign, fingers), shape_features(s_ref, sign_ref, fingers)
    sig = {"tip": cfg.sigma_tip, "angle": cfg.sigma_angle, "curl": cfg.sigma_curl, "aperture": cfg.sigma_aperture}
    z = []
    for k, (v, kind) in a.items():
        diff = v - b[k][0]
        if kind == "angle":
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        z.append(diff / sig[kind])
    z = torch.cat(z, 1)
    return torch.exp(-(z ** 2).mean(1).sqrt() / cfg.f1_temp)  # RMS of normalised errors, Laplacian tail


def contact_matrix(hand, obj, obj_n):
    """Aligned distance between every object point and every hand point: B,M,N.
    |h - o| * exp(2 (1 - |cos|)), cos between (h - o) and the object normal at o. Built from matmuls only."""
    d = torch.cdist(obj, hand)  # B,M,N
    hn = obj_n @ hand.transpose(1, 2)  # n_o . h
    on = (obj * obj_n).sum(-1, keepdim=True)  # n_o . o
    cos = ((hn - on).abs() / (d + 1e-9)).clamp(max=1.0)
    return d * torch.exp(2 * (1 - cos))


def soft_iou(a, b, dims):
    num = torch.minimum(a, b).sum(dims)
    den = torch.maximum(a, b).sum(dims)
    return torch.where(den > 1e-6, num / den.clamp_min(1e-6), torch.ones_like(den))


def _channels(part_ids, cfg):
    if cfg.obj_channels == "part":
        return part_ids, len(PARTS)
    g = torch.tensor([GROUPS.index(part_group(PARTS[p])) for p in part_ids.tolist()], device=part_ids.device)
    return g, len(GROUPS)


def contact_maps(s: HandState, obj, obj_n, cfg):
    """hand per-part contact (B, len(PARTS)), object map (B, M), object per-channel maps (B, C, M)."""
    B = s.points.shape[0]
    A = contact_matrix(s.points, obj.expand(B, -1, -1), obj_n.expand(B, -1, -1))
    dev, M = A.device, A.shape[1]
    pid = s.part_ids.to(dev)
    hc_pt = torch.exp(-(A.min(1).values / cfg.sigma_hand_contact) ** 2)  # B,N
    hc = torch.zeros(B, len(PARTS), device=dev).scatter_reduce(1, pid.expand(B, -1), hc_pt, "amax")
    oc = torch.exp(-(A.min(2).values / cfg.sigma_obj_contact) ** 2)
    ch, n_ch = _channels(pid, cfg)
    amin = torch.full((B, M, n_ch), float("inf"), device=dev).scatter_reduce(
        2, ch.view(1, 1, -1).expand(B, M, -1), A, "amin")
    og = torch.exp(-(amin.transpose(1, 2) / cfg.sigma_obj_contact) ** 2)  # missing channels -> 0
    return hc, oc, og


def prepare_reference(s_ref: HandState, obj, obj_n, cfg):
    hc, oc, og = contact_maps(s_ref, obj, obj_n, cfg)
    return dict(state=s_ref, hc=hc, oc=oc, og=og)


def score(s: HandState, s_ref: HandState, sign, sign_ref, obj, obj_n, cfg: FactorConfig, chunk=64, ref_maps=None,
          sym_perms=None):
    """s: candidates (B), s_ref: reference (B=1). obj/obj_n: 1,M,3. ref_maps: cached prepare_reference() output.
    sym_perms: list of (M,) object-point permutations (identity first); F4 takes the max over them.
    Returns dict of (B,) CPU tensors."""
    B = s.points.shape[0]
    ref_maps = ref_maps or prepare_reference(s_ref, obj, obj_n, cfg)
    shared_parts = sorted(set(s.part_ids.tolist()) & set(s_ref.part_ids.tolist()))
    shared_ch = shared_parts if cfg.obj_channels == "part" else \
        sorted({GROUPS.index(part_group(PARTS[p])) for p in shared_parts})
    ref_expand = HandState(points=s_ref.points.expand(B, -1, -1), part_ids=s_ref.part_ids,
                           wrist=s_ref.wrist.expand(B, -1), fingers={k: v.expand(B, -1, -1)
                                                                      for k, v in s_ref.fingers.items()})
    s1 = f1_shape(s, ref_expand, sign, sign_ref, cfg)
    hc_ref, oc_ref, og_ref = ref_maps["hc"][:, shared_parts], ref_maps["oc"], ref_maps["og"][:, shared_ch]
    s3, s4a, s4b, hcs, ocs, sym = [], [], [], [], [], []
    for i in range(0, B, chunk):
        sub = HandState(points=s.points[i:i + chunk], part_ids=s.part_ids, wrist=s.wrist[i:i + chunk],
                        fingers={k: v[i:i + chunk] for k, v in s.fingers.items()})
        hc, oc, og = contact_maps(sub, obj, obj_n, cfg)
        hc, og = hc[:, shared_parts], og[:, shared_ch]
        s3.append(soft_iou(hc, hc_ref, 1))
        best = None
        for gi, pm in enumerate(sym_perms or [None]):
            a_ = soft_iou(oc if pm is None else oc[:, pm], oc_ref, 1)
            b_ = soft_iou(og if pm is None else og[:, :, pm], og_ref, (1, 2))
            s_ = (a_ * b_).sqrt()
            if best is None:
                best, ba, bb, bi = s_, a_, b_, torch.zeros_like(s_, dtype=torch.long)
            else:
                m = s_ > best
                best, ba, bb = torch.where(m, s_, best), torch.where(m, a_, ba), torch.where(m, b_, bb)
                bi = torch.where(m, torch.full_like(bi, gi), bi)
        s4a.append(ba)
        s4b.append(bb)
        sym.append(bi)
        hcs.append(hc)
        ocs.append(oc)
    s3, s4a, s4b = torch.cat(s3), torch.cat(s4a), torch.cat(s4b)
    s4 = (s4a * s4b).sqrt()
    total = s1 ** cfg.w1 * s3 ** cfg.w3 * s4 ** cfg.w4
    out = dict(s1_shape=s1, s3_hand_contact=s3, s4_obj_contact=s4, s4a_obj_map=s4a, s4b_obj_part_map=s4b,
               total=total, hand_contact=torch.cat(hcs), obj_contact=torch.cat(ocs), hand_contact_ref=hc_ref,
               obj_contact_ref=oc_ref, sym_idx=torch.cat(sym))
    out = {k: v.cpu() for k, v in out.items()}
    out["shared_parts"] = shared_parts
    return out
