"""Train the hand-agnostic reference-conditioned Point Transformer grasp scorer.

Distills the analytic scorer (total = F1 * gate(F4p) * F4d, sigma = 1 cm)
across OakInk tool categories (both left-mirrored and right hands) plus
perturbed negative/intermediate grasps so the network is calibrated across
the full [0, 1] score range for online RL rewards.

Usage:
    PYTHONPATH=/home/szc/work/graspcalculator:/home/szc/.cache/grasp_score_pypath \
    /home/szc/.conda/envs/py310/bin/python -m graspcalc.train_transformer_scorer \
        --out /home/szc/work/graspcalculator/checkpoints/transformer_scorer_v1.pt \
        --epochs 40 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .hands import HandState, MANOHand, mirror
from .oakink import load_real_grasps, load_virtual_grasps, stack
from .paths import MANO_ROOT, OAKBASE_DIR, SHAPE_DIR
from .score_parts import load_obj_oakbase, oakbase_cat, part_labels
from .scorer import GraspScorer
from .transformer_scorer import GraspTransformerScorer, extract_grasp_tokens

DEFAULT_CATS = (
    "hammer",
    "screwdriver",
    "pen",
    "scissors",
    "toothbrush",
    "flashlight",
    "fryingpan",
    "power_drill",
    "trigger_sprayer",
    "knife",
)

CAT_AFFORD = {
    "hammer": ("grip",),
    "screwdriver": ("handle",),
    "pen": ("pen_body",),
    "scissors": ("handle",),
    "toothbrush": ("brush_handle", "handle"),
    "flashlight": ("handle",),
    "fryingpan": ("pan_handle", "handle"),
    "power_drill": ("drill_grip", "grip"),
    "trigger_sprayer": ("trigger",),
    "knife": ("handle",),
}


def resolve_ob_cat(cat: str) -> str:
    if cat == "pen":
        return "marker"
    return oakbase_cat(OAKBASE_DIR, cat)


def perturb_hand(s: HandState, noise_trans: float, noise_rot_deg: float) -> HandState:
    """Apply random rigid perturbation to a batched HandState (creates realistic RL negatives)."""
    dev = s.points.device
    B = s.points.shape[0]
    t = (torch.rand(B, 1, 3, device=dev) * 2.0 - 1.0) * noise_trans
    axis = F.normalize(torch.randn(B, 3, device=dev), dim=-1)
    ang = (torch.rand(B, device=dev) * 2.0 - 1.0) * (noise_rot_deg * np.pi / 180.0)
    # Rodrigues rotation
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    I = torch.eye(3, device=dev).unsqueeze(0).expand(B, -1, -1)
    R = I + torch.sin(ang)[:, None, None] * K + (1.0 - torch.cos(ang))[:, None, None] * (K @ K)

    c = s.wrist[:, None, :]
    rot = lambda x: torch.einsum("bij,bnj->bni", R, x - c) + c + t
    pts = rot(s.points)
    wrist = s.wrist + t[:, 0]
    fingers = {k: rot(v) for k, v in s.fingers.items()}
    return HandState(points=pts, part_ids=s.part_ids, wrist=wrist, fingers=fingers)


def collect_dataset(
    cats: list[str],
    max_objs_per_cat: int,
    max_grasps_per_obj: int,
    n_noisy_per_grasp: int,
    n_obj_pts: int,
    n_obj_tokens: int,
    device: str,
):
    print(f"[data] loading OakInk real + virtual grasps from {SHAPE_DIR}...")
    mano = MANOHand(MANO_ROOT, side="right", device=device)
    real = load_real_grasps(SHAPE_DIR)
    virt = load_virtual_grasps(SHAPE_DIR, real)
    all_g = real + virt

    by_cat_obj: dict[str, dict[str, list]] = {}
    for g in all_g:
        if g["cat"] in cats:
            by_cat_obj.setdefault(g["cat"], {}).setdefault(g["obj_id"], []).append(g)

    samples = []  # list of dicts with tensors on CPU
    for cat in cats:
        obj_map = by_cat_obj.get(cat, {})
        if not obj_map:
            continue
        ob_cat = resolve_ob_cat(cat)
        afford_names = CAT_AFFORD.get(cat, ("handle", "grip"))

        # Pick objects with >= 2 grasps; prefer real objects first
        oids = sorted(obj_map.keys(), key=lambda o: (o.startswith("o"), o))
        oids = [o for o in oids if len(obj_map[o]) >= 2][:max_objs_per_cat]

        # Find a good `use` reference grasp on the first valid object
        ref_bundle = None
        valid_objs = []
        for oid in oids:
            loaded = load_obj_oakbase(SHAPE_DIR, OAKBASE_DIR, ob_cat, oid, n_obj_pts)
            if loaded is None:
                continue
            obj_t, obj_n_t, name = loaded
            p_names, lab_np, _ = part_labels(OAKBASE_DIR, ob_cat, name, obj_t[0].numpy())
            aff_ids = [i for i, pn in enumerate(p_names) if pn in afford_names]
            if not aff_ids:
                aff_ids = [i for i, pn in enumerate(p_names) if any(k in pn for k in ("grip", "handle", "body"))]
            if not aff_ids:
                continue
            lab_t = torch.from_numpy(lab_np).long()
            aff_mask = torch.isin(lab_t, torch.tensor(aff_ids)).float()
            if aff_mask.mean().item() < 0.02:
                continue
            valid_objs.append((oid, obj_t, obj_n_t, p_names, lab_t, aff_mask, aff_ids))

        if len(valid_objs) < 2:
            print(f"[skip] {cat}: only {len(valid_objs)} valid objects")
            continue

        # Select reference grasp (prefer `use` intent on first object)
        ref_oid, ref_obj, ref_obj_n, ref_pnames, ref_lab, ref_aff, ref_aff_ids = valid_objs[0]
        ref_cands = obj_map[ref_oid]
        ref_g = next((g for g in ref_cands if g["intent"] == "use"), ref_cands[0])

        # Build both left-mirrored (primary for Sharpa) and right-hand references
        for side, do_mirror in (("left", True), ("right", False)):
            sign = -1.0 if do_mirror else 1.0
            with torch.no_grad():
                p_r, sh_r, t_r = stack("pose", [ref_g]).to(device), stack("shape", [ref_g]).to(device), stack("tsl", [ref_g]).to(device)
                s_ref = mano(p_r, sh_r, t_r)
                o_ref = ref_obj[0].to(device)
                on_ref = ref_obj_n[0].to(device)
                if do_mirror:
                    s_ref = mirror(s_ref, axis=0)
                    o_ref = o_ref.clone(); o_ref[:, 0] = -o_ref[:, 0]
                    on_ref = on_ref.clone(); on_ref[:, 0] = -on_ref[:, 0]

                ref_aff_names = [ref_pnames[i] for i in ref_aff_ids]
                scorer = GraspScorer(
                    s_ref=s_ref,
                    sign_ref=sign,
                    ref_obj=o_ref,
                    ref_obj_n=on_ref,
                    ref_lab=ref_lab.to(device),
                    vocab=ref_pnames,
                    afford_parts=ref_aff_names,
                    dist_sigma=0.01,
                    gate_tau=0.4,
                    gate_width=0.05,
                    device=device,
                    chunk=64,
                )
                rh_tok, ro_tok = extract_grasp_tokens(
                    s_ref, sign, o_ref, ref_aff.to(device), n_obj_tokens=n_obj_tokens
                )
                rh_tok = rh_tok[0].cpu()  # (16, 8)
                ro_tok = ro_tok[0].cpu()  # (K, 5)

            n_cat_samples = 0
            for oid, obj_t, obj_n_t, p_names, lab_t, aff_mask, aff_ids in valid_objs:
                gs = obj_map[oid][:max_grasps_per_obj]
                with torch.no_grad():
                    pose, shape, tsl = stack("pose", gs).to(device), stack("shape", gs).to(device), stack("tsl", gs).to(device)
                    s_clean = mano(pose, shape, tsl)
                    o_dev = obj_t[0].to(device)
                    on_dev = obj_n_t[0].to(device)
                    if do_mirror:
                        s_clean = mirror(s_clean, axis=0)
                        o_dev = o_dev.clone(); o_dev[:, 0] = -o_dev[:, 0]
                        on_dev = on_dev.clone(); on_dev[:, 0] = -on_dev[:, 0]

                    # Build clean + perturbed candidates
                    cand_states = [s_clean]
                    for ni in range(n_noisy_per_grasp):
                        nt = 0.015 if ni == 0 else 0.05
                        nr = 15.0 if ni == 0 else 45.0
                        cand_states.append(perturb_hand(s_clean, nt, nr))

                    # Concatenate all candidates for this object
                    s_all = HandState(
                        points=torch.cat([c.points for c in cand_states], dim=0),
                        part_ids=s_clean.part_ids,
                        wrist=torch.cat([c.wrist for c in cand_states], dim=0),
                        fingers={
                            k: torch.cat([c.fingers[k] for c in cand_states], dim=0)
                            for k in s_clean.fingers
                        },
                    )
                    # Map candidate part labels into ref_pnames space (affordance vs non-affordance)
                    lab_mapped = torch.where(
                        aff_mask.to(device) > 0.5,
                        torch.full_like(lab_t.to(device), ref_aff_ids[0]),
                        torch.full_like(lab_t.to(device), (ref_aff_ids[0] + 1) % max(len(ref_pnames), 2)),
                    )
                    res = scorer.score(s_all, sign, o_dev, on_dev, lab_mapped)
                    ch_tok, co_tok = extract_grasp_tokens(
                        s_all, sign, o_dev, aff_mask.to(device), n_obj_tokens=n_obj_tokens
                    )
                    targets = torch.stack(
                        [res["total"], res["f1"], res["f4p"], res["f4d"]], dim=-1
                    ).cpu()
                    ch_cpu = ch_tok.cpu()
                    co_cpu = co_tok.cpu()

                samples.append(
                    {
                        "cat": cat,
                        "oid": oid,
                        "side": side,
                        "ref_h": rh_tok,
                        "ref_o": ro_tok,
                        "cand_h": ch_cpu,
                        "cand_o": co_cpu,
                        "targets": targets,
                    }
                )
                n_cat_samples += len(targets)
            print(f"  [{cat:15s} {side:5s}] objs={len(valid_objs):2d} samples={n_cat_samples:5d}")
    return samples


def flatten_groups(groups: list[dict]):
    ref_h, ref_o, cand_h, cand_o, targets, group_ids = [], [], [], [], [], []
    for gid, g in enumerate(groups):
        n = g["targets"].shape[0]
        ref_h.append(g["ref_h"].unsqueeze(0).expand(n, -1, -1))
        ref_o.append(g["ref_o"].unsqueeze(0).expand(n, -1, -1))
        cand_h.append(g["cand_h"])
        cand_o.append(g["cand_o"])
        targets.append(g["targets"])
        group_ids.append(torch.full((n,), gid, dtype=torch.long))
    return (
        torch.cat(cand_h, 0),
        torch.cat(cand_o, 0),
        torch.cat(ref_h, 0),
        torch.cat(ref_o, 0),
        torch.cat(targets, 0),
        torch.cat(group_ids, 0),
    )


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic ranking loss within matching (cat, oid, side) groups."""
    same_g = gid[:, None] == gid[None, :]
    diff_t = target[:, None] - target[None, :]
    valid = same_g & (diff_t > 0.02)
    if not valid.any():
        return pred.new_tensor(0.0)
    diff_p = pred[:, None] - pred[None, :]
    return F.softplus(-10.0 * diff_p[valid]).mean()


def evaluate(model: GraspTransformerScorer, val_groups: list[dict], device: str):
    model.eval()
    spears = []
    maes = []
    with torch.no_grad():
        for g in val_groups:
            ch = g["cand_h"].to(device)
            co = g["cand_o"].to(device)
            rh = g["ref_h"].unsqueeze(0).to(device)
            ro = g["ref_o"].unsqueeze(0).to(device)
            hr, zr = model.encode(rh, ro)
            hc, zc = model.encode(ch, co)
            out = model.score_from_encoded(hc, zc, hr, zr)
            p = out["total"].cpu().numpy()
            t = g["targets"][:, 0].numpy()
            maes.append(np.abs(p - t).mean())
            if np.std(t) > 1e-4 and np.std(p) > 1e-4:
                r = spearmanr(p, t).statistic
                if np.isfinite(r):
                    spears.append(float(r))
    return float(np.mean(maes)), float(np.mean(spears)) if spears else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/szc/work/graspcalculator/checkpoints/transformer_scorer_v1.pt")
    p.add_argument("--cats", nargs="+", default=list(DEFAULT_CATS))
    p.add_argument("--max_objs_per_cat", type=int, default=16)
    p.add_argument("--max_grasps_per_obj", type=int, default=32)
    p.add_argument("--n_noisy", type=int, default=2)
    p.add_argument("--n_obj_pts", type=int, default=512)
    p.add_argument("--n_obj_tokens", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    t0 = time.time()
    groups = collect_dataset(
        args.cats,
        args.max_objs_per_cat,
        args.max_grasps_per_obj,
        args.n_noisy,
        args.n_obj_pts,
        args.n_obj_tokens,
        args.device,
    )

    # Split by object ID (20% unseen objects per category)
    train_groups, val_groups = [], []
    by_cat: dict[str, list[str]] = {}
    for g in groups:
        by_cat.setdefault(g["cat"], [])
        if g["oid"] not in by_cat[g["cat"]]:
            by_cat[g["cat"]].append(g["oid"])
    val_oids = set()
    for cat, oids in by_cat.items():
        n_val = max(1, len(oids) // 5)
        for o in oids[-n_val:]:
            val_oids.add((cat, o))
    for g in groups:
        if (g["cat"], g["oid"]) in val_oids:
            val_groups.append(g)
        else:
            train_groups.append(g)

    tr_ch, tr_co, tr_rh, tr_ro, tr_y, tr_gid = flatten_groups(train_groups)
    print(
        f"[split] train_groups={len(train_groups)} ({len(tr_y)} samples), "
        f"val_groups={len(val_groups)} (unseen objects), data prep {time.time()-t0:.1f}s"
    )

    model_cfg = dict(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        n_obj_tokens=args.n_obj_tokens,
    )
    model = GraspTransformerScorer(**model_cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_spear = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tr = len(tr_y)

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        tot_loss = 0.0
        n_b = 0
        for i in range(0, n_tr, args.batch_size):
            idx = perm[i : i + args.batch_size]
            ch = tr_ch[idx].to(args.device)
            co = tr_co[idx].to(args.device)
            rh = tr_rh[idx].to(args.device)
            ro = tr_ro[idx].to(args.device)
            y = tr_y[idx].to(args.device)
            gid = tr_gid[idx].to(args.device)

            out = model(ch, co, rh, ro)
            pred = torch.stack([out["total"], out["f1"], out["f4p"], out["f4d"]], dim=-1)
            weights = torch.tensor([2.0, 1.0, 1.0, 1.5], device=args.device)
            reg_loss = (F.smooth_l1_loss(pred, y, reduction="none") * weights).mean()
            rk_loss = pairwise_rank_loss(out["total"], y[:, 0], gid)
            loss = reg_loss + 0.2 * rk_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.item())
            n_b += 1
        sched.step()

        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            mae, spear = evaluate(model, val_groups, args.device)
            print(f"  ep {ep:3d}/{args.epochs} | loss={tot_loss/max(n_b,1):.4f} | val_unseen_MAE={mae:.4f} | val_unseen_Spearman={spear:.4f}")
            if spear > best_spear:
                best_spear = spear
                torch.save(
                    {
                        "model_cfg": model_cfg,
                        "state_dict": model.state_dict(),
                        "val_mae": mae,
                        "val_spearman": spear,
                        "cats": args.cats,
                    },
                    out_path,
                )

    # Benchmark inference speed at B=1024 and B=4096 (fp16)
    model.load_state_dict(torch.load(out_path, map_location=args.device, weights_only=False)["state_dict"])
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        rh = torch.randn(1, 16, 8, device=args.device)
        ro = torch.randn(1, args.n_obj_tokens, 5, device=args.device)
        hr, zr = model.encode(rh, ro)
        for B in (1024, 4096):
            ch = torch.randn(B, 16, 8, device=args.device)
            co = torch.randn(B, args.n_obj_tokens, 5, device=args.device)
            # warmup
            for _ in range(3):
                hc, zc = model.encode(ch, co)
                model.score_from_encoded(hc, zc, hr, zr)
            torch.cuda.synchronize()
            ts = time.perf_counter()
            for _ in range(10):
                hc, zc = model.encode(ch, co)
                model.score_from_encoded(hc, zc, hr, zr)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - ts) * 100.0
            print(f"[bench] B={B:4d} fp16 forward: {ms:.2f} ms")

    print(f"[done] saved best checkpoint to {out_path} (best unseen-object Spearman={best_spear:.4f})")


if __name__ == "__main__":
    main()
