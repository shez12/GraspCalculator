"""Build 12-category mirrored left-hand OakInk functional grasp references (refs_by_cat/)."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/szc/work/graspcalculator")

from graspcalc.hands import MANOHand, mirror
from graspcalc.oakink import load_real_grasps, stack
from graspcalc.paths import MANO_ROOT, OAKBASE_DIR, SHAPE_DIR
from graspcalc.score_parts import load_obj_oakbase, oakbase_cat, part_labels
from graspcalc.scorer import GraspScorer

OUT_DIR = Path("/home/szc/work/graspcalculator/refs_by_cat")
MANIFEST = Path("/home/szc/data/simtoolreal_oakink_assets/manifest.json")

FALLBACK_AFFORD = {
    "hammer": ("grip", "handle"),
    "screwdriver": ("handle", "grip"),
    "pen": ("pen_body", "body", "barrel", "handle"),
    "knife": ("handle", "grip"),
    "scissors": ("handle", "ring", "grip"),
    "toothbrush": ("brush_handle", "handle", "body"),
    "flashlight": ("handle", "body", "grip"),
    "fryingpan": ("pan_handle", "handle", "grip"),
    "power_drill": ("drill_grip", "grip", "handle", "trigger"),
    "trigger_sprayer": ("trigger", "handle", "grip", "body"),
    "wrench": ("handle", "grip", "body"),
    "pincer": ("handle", "grip"),
}


def get_afford_part_names(ob_cat: str, obj_name: str, part_names: list[str], cat: str) -> list[str]:
    """Match prepare_oakink_assets.py affordance rule: attr contains held_by_hand/trigger, fallback by name."""
    part_dir = os.path.join(OAKBASE_DIR, ob_cat, obj_name)
    afford = []
    for jf in sorted(glob.glob(os.path.join(part_dir, "part_*.json"))):
        meta = json.load(open(jf))
        raw = meta.get("name", "")
        short = raw[len(ob_cat) + 1 :] if raw.startswith(ob_cat + "_") else raw
        attrs = meta.get("attr", [])
        if isinstance(attrs, str):
            attrs = [attrs]
        if any(a in ("held_by_hand", "trigger", "held_by_two_hands") for a in attrs):
            if short in part_names:
                afford.append(short)
    if not afford:
        fb = FALLBACK_AFFORD.get(cat, ("handle", "grip"))
        afford = [p for p in part_names if any(k in p.lower() for k in fb)]
    if not afford:
        afford = [part_names[0]]
    return sorted(set(afford))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = json.load(open(MANIFEST))
    cats = sorted({a["category"] for a in m["assets"]})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hand = MANOHand(MANO_ROOT, side="right", device=device)
    all_real = load_real_grasps(SHAPE_DIR)

    index = {}
    fig = plt.figure(figsize=(14, 10), dpi=140)
    fig.suptitle("12-Category Mirrored Left-Hand OakInk Functional Grasp References", fontsize=14, fontweight="bold")

    for idx_c, cat in enumerate(cats):
        ob_cat = {"pen": "marker", "scissors": "scissor", "fryingpan": "frying_pan"}.get(cat, oakbase_cat(OAKBASE_DIR, cat))
        use_gs = [g for g in all_real if g["cat"] == cat and g.get("intent") == "use"]
        if not use_gs:
            use_gs = [g for g in all_real if g["cat"] == cat]

        # Evaluate each candidate grasp's own affordance contact (F4p on its affordance part)
        # and pick the cleanest functional grasp (highest affordance contact + close to category medoid).
        cands = []
        for g in use_gs[:24]:
            oid = g["obj_id"]
            try:
                obj_t, obj_nt, obj_name = load_obj_oakbase(SHAPE_DIR, OAKBASE_DIR, ob_cat, oid, 2048)
                pnames, lab_raw, _ = part_labels(OAKBASE_DIR, ob_cat, obj_name, obj_t[0].numpy())
            except Exception:
                continue
            aff_names = get_afford_part_names(ob_cat, obj_name, pnames, cat)
            aff_ids = {pnames.index(a) for a in aff_names}
            bin_lab = np.where(np.isin(lab_raw, list(aff_ids)), 0, 1).astype(np.int64)
            if (bin_lab == 0).sum() < 20:
                continue

            with torch.no_grad():
                s = hand(stack("pose", [g]).to(device), stack("shape", [g]).to(device), stack("tsl", [g]).to(device))
                s_left = mirror(s)
                sign_left = -hand.palm_normal_sign
                mir = torch.tensor([-1.0, 1.0, 1.0], device=device)
                obj_l = obj_t[0].to(device) * mir
                obj_nl = obj_nt[0].to(device) * mir
                lab_l = torch.as_tensor(bin_lab, dtype=torch.long, device=device)

                # Check hand-to-affordance proximity (fraction of hand points within 1.5 cm of affordance part)
                aff_pts = obj_l[lab_l == 0]
                d_aff = torch.cdist(s_left.points[0], aff_pts).min(dim=-1).values
                near_aff = float((d_aff < 0.015).float().mean().item())
                cands.append((near_aff, g, s_left, sign_left, obj_l, obj_nl, lab_l, aff_names))

        if not cands:
            raise RuntimeError(f"No valid OakBase use-grasp found for category {cat}")

        # Pick best candidate: among top-half by near_aff, pick medoid (highest mean F1 to others)
        cands.sort(key=lambda x: x[0], reverse=True)
        top = cands[: max(1, min(6, len(cands)))]
        best = top[0]
        if len(top) >= 3:
            best_score = -1.0
            for ci in top:
                sc_i = GraspScorer(
                    s_ref=ci[2],
                    sign_ref=ci[3],
                    ref_obj=ci[4],
                    ref_obj_n=ci[5],
                    ref_lab=ci[6],
                    vocab=["grip", "other"],
                    afford_parts=["grip"],
                    device=device,
                    chunk=16,
                )
                sims = []
                for cj in top:
                    r = sc_i.score(cj[2], cj[3], cj[4].unsqueeze(0), cj[5].unsqueeze(0), cj[6].unsqueeze(0))
                    sims.append(float(r["total"].item()))
                mean_sim = float(np.mean(sims))
                if mean_sim > best_score:
                    best_score = mean_sim
                    best = ci

        near_aff, g, s_left, sign_left, obj_l, obj_nl, lab_l, aff_names = best
        out_pt = OUT_DIR / f"{cat}.pt"
        torch.save(
            dict(
                points=s_left.points.cpu(),
                part_ids=s_left.part_ids.cpu(),
                wrist=s_left.wrist.cpu(),
                fingers={k: v.cpu() for k, v in s_left.fingers.items()},
                sign=float(sign_left),
                side="left",
                obj=obj_l.cpu(),
                obj_n=obj_nl.cpu(),
                lab=lab_l.cpu(),
                vocab=["grip", "other"],
                afford_parts_orig=aff_names,
                ref=f"{g['obj_id']}/{g['gid']}",
                intent=g.get("intent", "use"),
                cat=cat,
            ),
            out_pt,
        )
        index[cat] = {
            "path": str(out_pt),
            "ref": f"{g['obj_id']}/{g['gid']}",
            "intent": g.get("intent", "use"),
            "afford_parts_orig": aff_names,
            "near_aff_15mm": round(near_aff, 3),
            "num_candidates": len(cands),
        }
        print(f"  [{cat:16s}] ref={g['obj_id']}/{g['gid']} aff={aff_names} near15mm={near_aff:.3f} (from {len(cands)} cands)")

        # 3D scatter preview
        ax = fig.add_subplot(3, 4, idx_c + 1, projection="3d")
        hp = s_left.points[0].cpu().numpy()[::4]
        op = obj_l.cpu().numpy()[::4]
        ol = lab_l.cpu().numpy()[::4]
        ax.scatter(op[ol == 1, 0], op[ol == 1, 1], op[ol == 1, 2], s=2, c="#888888", alpha=0.4)
        ax.scatter(op[ol == 0, 0], op[ol == 0, 1], op[ol == 0, 2], s=3, c="#d62728", alpha=0.7)
        ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], s=3, c="#1f77b4", alpha=0.85)
        ax.set_title(f"{cat}\n{g['obj_id']} ({','.join(aff_names[:2])})", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Equal aspect
        all_p = np.concatenate([hp, op], axis=0)
        mid = all_p.mean(axis=0)
        rad = np.max(np.abs(all_p - mid)) * 1.1
        ax.set_xlim(mid[0] - rad, mid[0] + rad)
        ax.set_ylim(mid[1] - rad, mid[1] + rad)
        ax.set_zlim(mid[2] - rad, mid[2] + rad)

    with open(OUT_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "refs_overview.png")
    plt.close(fig)
    print(f"[done] saved 12 refs + index.json + refs_overview.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
