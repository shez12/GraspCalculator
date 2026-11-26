"""GPU-batched v2 Morphology-Invariant & Generalizable Affordance Grasp Scorer.

Works across arbitrary dexterous hands (MANO, Sharpa, Shadow, Allegro, etc.) via HandState.
Features:
  - F1_inv: bone flexion cosines + palm-normalized fingertip-to-thumb distances + palm-normalized fingertip heights.
  - F4p_inv: top-4 active finger affordance contact (sigma = 0.15 * L_palm) * thumb-opposing normal force-closure * affordance purity.
  - F4d_inv: [6 hand groups x 2 object parts] Soft IoU * affordance-only presence.
  - F_loc: dimensionless 1D functional-axis contact position & hand orientation alignment.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .hands import HandState

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
GROUP_NAMES = ("palm", "thumb", "index", "middle", "ring", "little")
GROUP_PIDS = (
    (0,),
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (13, 14, 15),
)


def _palm_scale_batched(fingers: dict[str, torch.Tensor], wrist: torch.Tensor) -> torch.Tensor:
    """(B,) intrinsic hand scale L_palm."""
    w_palm = torch.norm(fingers["index"][:, 0] - fingers["little"][:, 0], dim=-1)
    l_palm = torch.norm(fingers["middle"][:, 0] - wrist, dim=-1)
    return torch.clamp(0.5 * (w_palm + l_palm), min=0.04)


def _invariant_pose_feat_batched(
    fingers: dict[str, torch.Tensor],
    wrist: torch.Tensor,
    sign: float,
) -> torch.Tensor:
    """(B, 24) morphology-invariant hand pose descriptor."""
    L = _palm_scale_batched(fingers, wrist)  # (B,)
    v1 = fingers["index"][:, 0] - wrist
    v2 = fingers["little"][:, 0] - wrist
    n_palm = F.normalize(torch.cross(v1, v2, dim=-1), dim=-1) * float(sign)
    palm_c = 0.5 * (wrist + fingers["middle"][:, 0])

    feats = []
    # 1. Per-finger consecutive bone flexion cosines: 5 x 3 = 15 dims
    for fn in FINGER_NAMES:
        j = fingers[fn]  # (B, 4, 3)
        chain = torch.cat([wrist.unsqueeze(1), j], dim=1)  # (B, 5, 3)
        bones = F.normalize(chain[:, 1:] - chain[:, :-1], dim=-1)  # (B, 4, 3)
        cos_ang = (bones[:, :-1] * bones[:, 1:]).sum(dim=-1)  # (B, 3)
        feats.append(cos_ang)

    # 2. Fingertip-to-thumb opposition distances normalized by L_palm: 4 dims
    t_tip = fingers["thumb"][:, -1]  # (B, 3)
    opp_d = torch.stack(
        [torch.norm(fingers[fn][:, -1] - t_tip, dim=-1) / L for fn in FINGER_NAMES[1:]],
        dim=-1,
    ).clamp(0.0, 2.5)
    feats.append(opp_d)

    # 3. Fingertip signed heights above palm plane normalized by L_palm: 5 dims
    heights = torch.stack(
        [((fingers[fn][:, -1] - palm_c) * n_palm).sum(dim=-1) / L for fn in FINGER_NAMES],
        dim=-1,
    ).clamp(-2.0, 2.0)
    feats.append(heights)

    return torch.cat(feats, dim=-1)  # (B, 24)


def _contact_and_axis_batched(
    points: torch.Tensor,
    part_ids: torch.Tensor,
    fingers: dict[str, torch.Tensor],
    wrist: torch.Tensor,
    sign: float,
    obj: torch.Tensor,
    obj_n: torch.Tensor,
    aff_mask: torch.Tensor,
    chunk: int = 256,
) -> dict[str, torch.Tensor]:
    """Fully GPU-batched extraction of [6x2] soft contact matrix, F4p raw, affordance share, and 1D functional axis.

    Args:
        points: (B, Nh, 3)
        part_ids: (Nh,) or (B, Nh)
        obj: (B, No, 3)
        obj_n: (B, No, 3)
        aff_mask: (B, No) float/bool (1 = affordance, 0 = other)
    """
    B, Nh, _ = points.shape
    device = points.device
    pids = (part_ids[0] if part_ids.ndim == 2 else part_ids).to(device)  # (Nh,)
    aff_bool = aff_mask > 0.5  # (B, No)
    oth_bool = ~aff_bool

    L = _palm_scale_batched(fingers, wrist)  # (B,)
    sigma = 0.15 * L  # (B,)

    # Precompute group point masks on hand
    g_indices = []
    for gpids in GROUP_PIDS:
        m = torch.zeros(Nh, dtype=torch.bool, device=device)
        for p in gpids:
            m |= (pids == p)
        g_indices.append(m.nonzero(as_tuple=False).squeeze(-1))

    mat = torch.zeros(B, 6, 2, device=device, dtype=torch.float32)
    group_normals = torch.zeros(B, 6, 3, device=device, dtype=torch.float32)
    weighted_pos = torch.zeros(B, 3, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(B, device=device, dtype=torch.float32)

    has_aff = aff_bool.any(dim=1)  # (B,)
    has_oth = oth_bool.any(dim=1)  # (B,)
    big = torch.tensor(1e6, device=device, dtype=torch.float32)

    # Chunk over batch to keep VRAM tiny even for B=6144
    for s0 in range(0, B, chunk):
        s1 = min(B, s0 + chunk)
        pts_c = points[s0:s1]
        obj_c = obj[s0:s1]
        obj_nc = obj_n[s0:s1]
        aff_c = aff_bool[s0:s1]  # (b, No)
        oth_c = oth_bool[s0:s1]  # (b, No)
        sig_c = sigma[s0:s1]  # (b,)
        b_sz = s1 - s0

        for gi, gidx in enumerate(g_indices):
            if gidx.numel() == 0:
                continue
            gpts = pts_c[:, gidx]  # (b, Ng, 3)
            d_all = torch.cdist(gpts, obj_c)  # (b, Ng, No)

            # Affordance min distance & nearest surface point/normal
            d_aff = d_all.masked_fill(~aff_c.unsqueeze(1), big)
            d_min_o, nn_o = d_aff.min(dim=-1)  # (b, Ng)
            d_best, best_g = d_min_o.min(dim=-1)  # (b,)
            best_o = nn_o.gather(1, best_g.unsqueeze(1)).squeeze(1)  # (b,)

            c_aff = torch.exp(-0.5 * (d_best / sig_c).pow(2)) * has_aff[s0:s1].float()
            mat[s0:s1, gi, 0] = c_aff

            b_arange = torch.arange(b_sz, device=device)
            group_normals[s0:s1, gi] = obj_nc[b_arange, best_o]
            if gi >= 1:  # fingers only (skip palm) for affordance contact centroid
                pt_obj = obj_c[b_arange, best_o]
                weighted_pos[s0:s1] += c_aff.unsqueeze(-1) * pt_obj
                weight_sum[s0:s1] += c_aff

            # Non-affordance min distance
            d_oth = d_all.masked_fill(~oth_c.unsqueeze(1), big)
            d_best_oth = d_oth.amin(dim=(1, 2))
            c_oth = torch.exp(-0.5 * (d_best_oth / sig_c).pow(2)) * has_oth[s0:s1].float()
            mat[s0:s1, gi, 1] = c_oth

    # Top-4 active fingers affordance coverage
    finger_c = mat[:, 1:6, 0]  # (B, 5)
    top4 = torch.topk(finger_c, k=4, dim=-1).values
    f4p_cov = top4.mean(dim=-1)  # (B,)

    # Force-closure opposition bonus: thumb normal vs other contacting fingers' normals
    nt = F.normalize(group_normals[:, 1], dim=-1)  # (B, 3)
    nf = F.normalize(group_normals[:, 2:6], dim=-1)  # (B, 4, 3)
    opp_dots = torch.clamp(-(nt.unsqueeze(1) * nf).sum(dim=-1), 0.0, 1.0)  # (B, 4)
    active_f = (mat[:, 2:6, 0] > 0.2)  # (B, 4)
    opp_masked = opp_dots.masked_fill(~active_f, -1.0)
    max_opp = opp_masked.max(dim=-1).values  # (B,)
    valid_opp = (mat[:, 1, 0] > 0.2) & (max_opp >= 0.0)
    opp = torch.where(valid_opp, 0.5 + 0.5 * max_opp.clamp(min=0.0), torch.full_like(max_opp, 0.5))

    f4p_raw = f4p_cov * opp

    # Affordance contact share (purity)
    sum_aff = mat[:, :, 0].sum(dim=-1)
    sum_oth = mat[:, :, 1].sum(dim=-1)
    aff_share = sum_aff / (sum_aff + sum_oth + 1e-6)

    # Generalizable 1D functional axis (afford_center -> other_center)
    aff_cnt = aff_bool.sum(dim=-1)  # (B,)
    oth_cnt = oth_bool.sum(dim=-1)  # (B,)
    has_axis = (aff_cnt > 4) & (oth_cnt > 4)

    aff_f = aff_bool.float().unsqueeze(-1)
    oth_f = oth_bool.float().unsqueeze(-1)
    aff_mean = (obj * aff_f).sum(dim=1) / aff_cnt.clamp(min=1).float().unsqueeze(-1)
    oth_mean = (obj * oth_f).sum(dim=1) / oth_cnt.clamp(min=1).float().unsqueeze(-1)
    axis_v = oth_mean - aff_mean
    axis_norm = torch.norm(axis_v, dim=-1)
    has_axis = has_axis & (axis_norm > 1e-3)
    a_hat = axis_v / axis_norm.clamp(min=1e-6).unsqueeze(-1)  # (B, 3)

    # Project affordance points onto a_hat to get [p_min, p_max] span
    proj_all = (obj * a_hat.unsqueeze(1)).sum(dim=-1)  # (B, No)
    proj_min = proj_all.masked_fill(~aff_bool, big).min(dim=-1).values
    proj_max = proj_all.masked_fill(~aff_bool, -big).max(dim=-1).values
    # Shrink 5% on each end to match quantile(0.05..0.95) robustness without slow per-row quantile
    raw_span = (proj_max - proj_min).clamp(min=0.02)
    p_lo = proj_min + 0.05 * raw_span
    p_hi = proj_max - 0.05 * raw_span
    span = (p_hi - p_lo).clamp(min=0.02)

    hand_mean = points.mean(dim=1)
    c_pos = torch.where(
        (weight_sum > 1e-3).unsqueeze(-1),
        weighted_pos / weight_sum.clamp(min=1e-6).unsqueeze(-1),
        hand_mean,
    )
    s_ax = (((c_pos * a_hat).sum(dim=-1) - p_lo) / span).clamp(-0.5, 1.5)

    hand_dir = F.normalize(fingers["index"][:, 0] - wrist, dim=-1)
    ori_hand = (hand_dir * a_hat).sum(dim=-1)
    v1 = fingers["index"][:, 0] - wrist
    v2 = fingers["little"][:, 0] - wrist
    n_palm = F.normalize(torch.cross(v1, v2, dim=-1), dim=-1) * float(sign)
    ori_palm = (n_palm * a_hat).sum(dim=-1)

    return dict(
        mat=mat,
        f4p_raw=f4p_raw,
        aff_share=aff_share,
        max_aff=mat[:, :, 0].max(dim=-1).values,
        has_axis=has_axis,
        s_ax=s_ax,
        ori_hand=ori_hand,
        ori_palm=ori_palm,
    )


class InvariantGraspScorer:
    """Drop-in GPU-batched replacement for GraspScorer using v2 morphology-invariant features."""

    def __init__(
        self,
        ref_hand: HandState,
        ref_sign: float,
        ref_obj: torch.Tensor,
        ref_obj_n: torch.Tensor,
        ref_aff_mask: torch.Tensor,
        device: str | torch.device = "cuda",
        chunk: int = 256,
        vocab: list[str] | None = None,
    ):
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self.vocab = list(vocab) if vocab is not None else ["grip", "other"]
        with torch.no_grad():
            pts = ref_hand.points.to(self.device)
            if pts.ndim == 2:
                pts = pts.unsqueeze(0)
            pids = ref_hand.part_ids.to(self.device)
            wrist = ref_hand.wrist.to(self.device)
            if wrist.ndim == 1:
                wrist = wrist.unsqueeze(0)
            fingers = {
                k: (v.unsqueeze(0) if v.ndim == 2 else v).to(self.device)
                for k, v in ref_hand.fingers.items()
            }
            obj = (ref_obj.unsqueeze(0) if ref_obj.ndim == 2 else ref_obj).to(self.device)
            obj_n = (ref_obj_n.unsqueeze(0) if ref_obj_n.ndim == 2 else ref_obj_n).to(self.device)
            aff = (ref_aff_mask.unsqueeze(0) if ref_aff_mask.ndim == 1 else ref_aff_mask).to(self.device)

            self.ref_pose = _invariant_pose_feat_batched(fingers, wrist, ref_sign)  # (1, 24)
            fr = _contact_and_axis_batched(pts, pids, fingers, wrist, ref_sign, obj, obj_n, aff, chunk=1)
            self.ref_mat = fr["mat"]  # (1, 6, 2)
            self.ref_aff_share = float(fr["aff_share"].item())
            self.ref_has_axis = bool(fr["has_axis"].item())
            self.ref_s_ax = float(fr["s_ax"].item())
            self.ref_ori_hand = float(fr["ori_hand"].item())
            self.ref_ori_palm = float(fr["ori_palm"].item())

    def ref_feature_vector(self) -> torch.Tensor:
        """Return dimensionless 39-dim morphology-invariant reference vector:
        ref_pose (24) + ref_mat (6x2=12) + [ref_s_ax, ref_ori_hand, ref_ori_palm] (3).
        """
        loc = (
            [self.ref_s_ax, self.ref_ori_hand, self.ref_ori_palm]
            if self.ref_has_axis
            else [0.5, 0.0, 0.0]
        )
        loc_t = torch.tensor(loc, device=self.device, dtype=torch.float32)
        return torch.cat(
            [self.ref_pose[0].float(), self.ref_mat[0].reshape(-1).float(), loc_t],
            dim=0,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        afford_parts: list[str] | None = None,
        device: str | torch.device = "cuda",
        chunk: int = 256,
    ) -> "InvariantGraspScorer":
        r = torch.load(path, map_location="cpu", weights_only=False)
        hand = HandState(
            points=r["points"],
            part_ids=r["part_ids"],
            wrist=r["wrist"],
            fingers=r["fingers"],
        )
        vocab = r.get("vocab", ["grip", "other"])
        lab = r["lab"]
        if afford_parts:
            aff_ids = [vocab.index(p) for p in afford_parts if p in vocab]
            aff_mask = torch.zeros_like(lab, dtype=torch.float32)
            for aid in aff_ids:
                aff_mask = torch.where(lab == aid, torch.ones_like(aff_mask), aff_mask)
        else:
            aff_mask = (lab == 0).float()
        return cls(
            ref_hand=hand,
            ref_sign=float(r["sign"]),
            ref_obj=r["obj"],
            ref_obj_n=r["obj_n"],
            ref_aff_mask=aff_mask,
            device=device,
            chunk=chunk,
        )

    @torch.no_grad()
    def score(
        self,
        hand: HandState,
        sign: float,
        obj: torch.Tensor,
        obj_n: torch.Tensor,
        obj_lab: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Same signature as GraspScorer.score(hand, sign, obj, obj_n, obj_lab).
        Note: obj_lab uses 0 = affordance, 1 = other (matching SimToolReal & graspcalc convention).
        """
        aff_mask = (obj_lab == 0).float()
        f_c = _invariant_pose_feat_batched(hand.fingers, hand.wrist, sign)  # (B, 24)
        f1 = torch.exp(-(f_c - self.ref_pose).pow(2).mean(dim=-1) / 0.35)

        fc = _contact_and_axis_batched(
            hand.points, hand.part_ids, hand.fingers, hand.wrist, sign,
            obj, obj_n, aff_mask, chunk=self.chunk,
        )

        # F4p_v2: affordance coverage * opposition * affordance purity
        purity = torch.clamp(fc["aff_share"] / max(self.ref_aff_share, 0.15), max=1.0)
        f4p = fc["f4p_raw"] * purity

        # F4d_v2: [6 x 2] Soft IoU * affordance-only presence
        mc = fc["mat"]
        mr = self.ref_mat
        inter = torch.minimum(mc, mr).sum(dim=(1, 2))
        union = torch.maximum(mc, mr).sum(dim=(1, 2))
        soft_iou = inter / (union + 1e-6)
        presence_aff = torch.clamp(fc["max_aff"] / 0.5, max=1.0)
        f4d = soft_iou * presence_aff

        # F_loc: generalizable 1D functional-axis position & orientation
        if self.ref_has_axis:
            d_s = (fc["s_ax"] - self.ref_s_ax).abs()
            d_ori = 0.5 * ((fc["ori_hand"] - self.ref_ori_hand).pow(2) + (fc["ori_palm"] - self.ref_ori_palm).pow(2))
            f_loc_raw = torch.exp(-(d_s / 0.35).pow(2)) * torch.exp(-d_ori / 0.65)
            f_loc = torch.where(fc["has_axis"], f_loc_raw, torch.ones_like(f_loc_raw))
        else:
            f_loc = torch.ones_like(f1)

        gate = torch.sigmoid((f4p - 0.32) / 0.08)
        total = f1 * gate * f4d * f_loc
        return dict(total=total, f1=f1, f4p=f4p, f4d=f4d, f_loc=f_loc)
