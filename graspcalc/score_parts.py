"""Cross-object grasp scoring with OakBase functional-part labels.

Every object point gets the OakBase part (e.g. teapot body/spout/handle/lid) of its nearest part point.
For a grasp, A[k, p] = sum over object points o of part p of og[k, o], where og[k, o] is the soft contact
between hand semantic part k (16) and object point o.  P = A / sum(A) is the share of the contact made by
hand part k on object part p; it does not depend on the instance, so one reference grasp can score grasps on
every object of the category.
    F4p_a = soft IoU of the object-part shares (sum over k)
    F4p_b = soft IoU of the joint 16 x P shares
    F4p   = sqrt(F4p_a * F4p_b)
    gate  = sigmoid((F4p - tau) / width) * part_ok,  total = F1 * gate   (F4p decides where; F1 ranks hand shape)
    part_ok = 1 iff every object part holding >= ref_part_min of the reference contact holds
              >= min_part_ratio * (reference share) of the candidate contact (hard requirement)
    --afford_parts handle: only the listed affordance parts count; the requirement checks only them and F4p is the
    soft IoU of the 16 x K contact restricted to them (which hand parts touch the handle, renormalised).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .factors import FactorConfig, contact_maps, f1_shape, soft_iou
from .hands import PARTS, HandState, MANOHand, part_group
from .oakink import GROUP_COLORS, load_mesh, load_obj, load_real_grasps, load_virtual_grasps, stack
from .paths import MANO_ROOT, OAKBASE_DIR, SHAPE_DIR

PART_COLORS = [(0.80, 0.80, 0.80), (0.55, 0.75, 1.0), (0.60, 0.92, 0.60), (1.0, 0.85, 0.45), (0.85, 0.65, 1.0)]


OAKBASE_CAT = {"scissors": "scissor", "cameras": "camera", "fryingpan": "frying_pan", "gamecontroller": "game_controller",
               "lotion_pump": "lotion_bottle", "squeezable": "squeeze_tube"}


def oakbase_cat(oakbase, cat):
    """OakInk-Shape and OakBase name a few categories differently (e.g. scissors / scissor)."""
    return cat if os.path.isdir(os.path.join(oakbase, cat)) else OAKBASE_CAT.get(cat, cat)


def part_labels(oakbase, cat, name, pts):
    names, clouds = [], []
    for j in sorted(glob.glob(os.path.join(oakbase, cat, name, "part_*.json"))):
        n = json.load(open(j))["name"]
        names.append(n[len(cat) + 1:] if n.startswith(cat + "_") else n)
        clouds.append(np.asarray(trimesh.load(j[:-5] + ".ply").vertices))
    lab = np.concatenate([np.full(len(c), i) for i, c in enumerate(clouds)])
    d, idx = cKDTree(np.concatenate(clouds)).query(pts)
    return names, lab[idx], d


def load_obj_oakbase(shape_dir, oakbase, cat, oid, n_pts, tol=0.005):
    """Object surface restricted to what OakBase segments as the object (drops base plates / table sheets)."""
    mesh, name = load_mesh(shape_dir, oid)
    files = sorted(glob.glob(os.path.join(oakbase, cat, name, "part_*.ply")))
    if not files:
        return None
    cloud = np.concatenate([np.asarray(trimesh.load(f).vertices) for f in files])
    keep = cKDTree(cloud).query(mesh.triangles_center)[0] < tol
    drop = 1 - mesh.area_faces[keep].sum() / mesh.area
    if drop > 0.2:
        print(f"[skip] {oid} {name}: OakBase parts cover only {(1 - drop) * 100:.0f}% of the mesh (misaligned)")
        return None
    if drop > 0.01:
        print(f"[oakbase-crop] {oid} {name}: dropped {drop * 100:.0f}% of the surface not covered by OakBase parts")
        mesh = mesh.copy()
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    return load_obj(shape_dir, oid, n_pts, mesh, name)


def expand(s: HandState, B):
    return HandState(points=s.points.expand(B, -1, -1), part_ids=s.part_ids, wrist=s.wrist.expand(B, -1),
                     fingers={k: v.expand(B, -1, -1) for k, v in s.fingers.items()})


def index(s: HandState, i):
    return HandState(points=s.points[i], part_ids=s.part_ids, wrist=s.wrist[i],
                     fingers={k: v[i] for k, v in s.fingers.items()})


def part_shares(s, obj, obj_n, onehot, cfg, chunk=64):
    """returns P (B, 16, n_obj_parts) contact shares and the scalar object contact map (B, M)."""
    Ps, ocs = [], []
    for i in range(0, s.points.shape[0], chunk):
        _, oc, og = contact_maps(index(s, slice(i, i + chunk)), obj, obj_n, cfg)
        A = og @ onehot
        Ps.append(A / A.sum((1, 2), keepdim=True).clamp_min(1e-9))
        ocs.append(oc)
    return torch.cat(Ps), torch.cat(ocs)


def f4p(P, P_ref):
    a = soft_iou(P.sum(1), P_ref.sum(1).expand(len(P), -1), 1)
    b = soft_iou(P, P_ref.expand(len(P), -1, -1), (1, 2))
    return (a * b).sqrt(), a, b


def visualize(path, title, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5 * len(panels), 5.6))
    for k, (name, o, lab, pts, part_ids, oc) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1, projection="3d")
        w = np.clip(np.nan_to_num(oc), 0, 1)[:, None]
        base = np.array([PART_COLORS[l % len(PART_COLORS)] for l in lab])
        ax.scatter(*o.T, c=np.clip((1 - w) * base + w * np.array([0.85, 0.0, 0.0]), 0, 1), s=2 + 6 * w[:, 0], alpha=0.8)
        ax.scatter(*pts.T, c=[GROUP_COLORS[part_group(PARTS[p])] for p in part_ids.tolist()], s=1.5, alpha=0.35)
        allp = np.concatenate([o, pts])
        c, r = (allp.max(0) + allp.min(0)) / 2, (allp.max(0) - allp.min(0)).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()
        ax.set_title(name, fontsize=8.5)
    fig.suptitle(title, fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(a):
    torch.set_grad_enabled(False)
    cfg = FactorConfig()
    cfg.f1_temp *= a.f1_loosen
    hand = MANOHand(a.mano_root, device=a.device)
    sign = hand.palm_normal_sign
    grasps = [g for g in load_real_grasps(a.shape_dir) if g["cat"] == a.cat]
    virt = load_virtual_grasps(a.shape_dir, grasps)
    is_virtual = {g["obj_id"] for g in virt}.__contains__
    grasps += virt
    all_grasps = list(grasps)
    objs = list(dict.fromkeys(g["obj_id"] for g in grasps))
    if not a.ref:
        g0 = next((g for g in grasps if not is_virtual(g["obj_id"]) and g["intent"] == "use"), grasps[0])
        a.ref = f"{g0['obj_id']}/{g0['gid']}"
    out_dir = a.out_dir or f"out_parts_{a.cat}"
    os.makedirs(out_dir, exist_ok=True)

    ob_cat = oakbase_cat(a.oakbase, a.cat)
    vocab, data = [], {}
    for oid in objs:
        loaded = load_obj_oakbase(a.shape_dir, a.oakbase, ob_cat, oid, a.n_obj_pts)
        if loaded is None:
            print(f"[skip] {oid}: no OakBase parts")
            continue
        obj, obj_n, name = loaded
        names, lab, dist = part_labels(a.oakbase, ob_cat, name, obj[0].numpy())
        for n in names:
            if n not in vocab:
                vocab.append(n)
        data[oid] = dict(name=name, obj=obj, obj_n=obj_n, lab=np.array([vocab.index(names[l]) for l in lab]),
                         lab_dist_mm=float(np.percentile(dist, 95) * 1000), gs=[g for g in grasps if g["obj_id"] == oid])
    print(f"category {a.cat}: {len(objs)} objects, {len(grasps)} grasps, parts {vocab}")
    objs = list(data)
    grasps = [g for g in grasps if g["obj_id"] in data]
    for oid, d in data.items():
        onehot = torch.nn.functional.one_hot(torch.as_tensor(d["lab"]), len(vocab)).float().to(a.device)
        d["st"] = hand(stack("pose", d["gs"]), stack("shape", d["gs"]), stack("tsl", d["gs"]))
        d["P"], d["oc"] = part_shares(d["st"], d["obj"].to(a.device), d["obj_n"].to(a.device), onehot, cfg)

    aff = [vocab.index(p) for p in a.afford_parts.split(",") if p in vocab] if a.afford_parts else []
    if a.afford_parts:
        print(f"affordance parts: {[vocab[i] for i in aff]}")

    def score_all(ref_oid, ref_gid):
        d_ref = data[ref_oid]
        i_ref = next(i for i, g in enumerate(d_ref["gs"]) if g["gid"] == ref_gid)
        s_ref, P_ref = index(d_ref["st"], slice(i_ref, i_ref + 1)), d_ref["P"][i_ref:i_ref + 1]
        out = {}
        for oid, d in data.items():
            B = len(d["gs"])
            s1 = f1_shape(d["st"], expand(s_ref, B), sign, sign, cfg)
            if aff:
                Pa, Pra = d["P"][:, :, aff], P_ref[:, :, aff]
                s4a = (Pa.sum((1, 2)) / Pra.sum((1, 2)).clamp_min(1e-9)).clamp(max=1)
                s4 = s4b = soft_iou(Pa / Pa.sum((1, 2), keepdim=True).clamp_min(1e-9),
                                    (Pra / Pra.sum((1, 2), keepdim=True).clamp_min(1e-9)).expand(B, -1, -1), (1, 2))
            else:
                s4, s4a, s4b = f4p(d["P"], P_ref)
            gate = torch.sigmoid((s4 - a.gate_tau) / a.gate_width) if a.gate_tau > 0 else s4.sqrt() / s1.sqrt().clamp_min(1e-9)
            need = P_ref.sum(1) >= a.ref_part_min
            if aff:
                need = torch.zeros_like(need)
                need[:, aff] = True
            ok = ((d["P"].sum(1) >= a.min_part_ratio * P_ref.sum(1)) | ~need).all(1).float()
            gate = gate * ok
            out[oid] = dict(s1=s1.cpu(), s4=s4.cpu(), s4a=s4a.cpu(), s4b=s4b.cpu(), gate=gate.cpu(), ok=ok.cpu(),
                            total=(s1 * gate).cpu())
        return out, i_ref

    # 1) one reference for the whole category
    ref_oid, ref_gid = a.ref.split("/")
    res, i_ref = score_all(ref_oid, ref_gid)
    d_ref = data[ref_oid]
    ref_g = d_ref["gs"][i_ref]
    share = lambda P: " ".join(f"{vocab[p]} {v * 100:.0f}%" for p, v in enumerate(P.sum(0).tolist()) if v > 0.01)
    rows_all, by_intent = [], {}
    for oid, d in data.items():
        r = res[oid]
        rows = []
        for i, g in enumerate(d["gs"]):
            is_ref = oid == ref_oid and i == i_ref
            rows.append(dict(obj_id=oid, obj=d["name"], gid=g["gid"], intent=g["intent"], is_ref=is_ref,
                             total=r["total"][i].item(), s1_shape=r["s1"][i].item(), s4p=r["s4"][i].item(),
                             s4p_a=r["s4a"][i].item(), s4p_b=r["s4b"][i].item(), gate=r["gate"][i].item(), part_ok=bool(r["ok"][i]),
                             part_share={vocab[p]: v for p, v in enumerate(d["P"][i].sum(0).tolist())}))
            if not is_ref:
                by_intent.setdefault(g["intent"], []).append(rows[-1]["total"])
        order = sorted(range(len(rows)), key=lambda i: -rows[i]["total"])
        print(f"\n=== {d['name']} ({oid})  n={len(rows)}  ref={ref_oid}/{ref_gid}  part-label p95 dist {d['lab_dist_mm']:.1f} mm")
        print(f"{'gid':<12}{'intent':<10}{'total':>7}{'F1':>7}{'F4p':>7}{'F4p_a':>7}{'F4p_b':>7}{'gate':>7}  object-part share of contact")
        for i in order:
            x = rows[i]
            print(f"{x['gid']:<12}{x['intent']:<10}{x['total']:>7.3f}{x['s1_shape']:>7.3f}{x['s4p']:>7.3f}"
                  f"{x['s4p_a']:>7.3f}{x['s4p_b']:>7.3f}{x['gate']:>7.3f}  {share(d['P'][i].cpu())}{'  <- ref' if x['is_ref'] else ''}")
        cand = [i for i in order if not rows[i]["is_ref"]]
        pick = [("best", cand[0]), ("median", cand[len(cand) // 2]), ("worst", cand[-1])]
        panels = [(f"REF {ref_oid}/{ref_gid} ({ref_g['intent']})\n{share(d_ref['P'][i_ref].cpu())}",
                   d_ref["obj"][0].numpy(), d_ref["lab"], d_ref["st"].points[i_ref].cpu().numpy(), hand.part_ids,
                   d_ref["oc"][i_ref].cpu().numpy())]
        for tag, i in pick:
            x = rows[i]
            panels.append((f"{tag}: {x['gid']} ({x['intent']})\nS={x['total']:.2f} F1={x['s1_shape']:.2f} "
                           f"F4p={x['s4p']:.2f} gate={x['gate']:.2f}\n{share(d['P'][i].cpu())}", d["obj"][0].numpy(), d["lab"],
                           d["st"].points[i].cpu().numpy(), hand.part_ids, d["oc"][i].cpu().numpy()))
        legend = ", ".join(f"{n}={['grey', 'blue', 'green', 'yellow', 'purple'][k % 5]}" for k, n in enumerate(vocab))
        visualize(os.path.join(out_dir, f"viz_{oid}.png"),
                  f"{d['name']} ({oid}) scored against ONE reference on {ref_oid}  |  object parts: {legend}; red = contact",
                  panels)
        rows_all += [rows[i] for i in order]
    json.dump(rows_all, open(os.path.join(out_dir, "part_scores.json"), "w"), indent=1)
    print("\nmean total by intent (all objects, ref excluded):")
    for k, v in sorted(by_intent.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {k:<10} n={len(v):<4} mean {np.mean(v):.3f}  median {np.median(v):.3f}")

    # 2) retrieval check: a real grasp used as reference should rank its own transferred copy first on each object
    print("\nretrieval check (ref = real grasp on the real object; rank of its transferred copy on each virtual object):")
    ranks, retr = [], []
    for real_oid, g in [(o, g) for o in objs if not is_virtual(o) for g in data[o]["gs"]]:
        hits = [(oid, i) for oid, d in data.items() if is_virtual(oid) for i, x in enumerate(d["gs"]) if x["gid"] == g["gid"]]
        if not hits:
            continue
        r, _ = score_all(real_oid, g["gid"])
        rk = [int((r[oid]["total"] > r[oid]["total"][i]).sum()) + 1 for oid, i in hits]
        n = [len(data[oid]["gs"]) for oid, _ in hits]
        ranks += rk
        i_r = next(i for i, x in enumerate(data[real_oid]["gs"]) if x["gid"] == g["gid"])
        retr.append(dict(ref=f"{real_oid}/{g['gid']}", intent=g["intent"], n_objects=len(hits), top1=float(np.mean(np.array(rk) == 1)),
                         mean_rank=float(np.mean(rk)), mean_n=float(np.mean(n)),
                         ref_part_share={vocab[p]: v for p, v in enumerate(data[real_oid]["P"][i_r].sum(0).tolist())}))
        print(f"  {real_oid}/{g['gid']} ({g['intent']:<8}) on {len(hits):>2} objects: top-1 {np.mean(np.array(rk) == 1) * 100:5.1f}%  "
              f"mean rank {np.mean(rk):.2f} / mean n {np.mean(n):.1f}")
    if not ranks:
        print("  no transferred (virtual) grasps in this category")
    else:
        print(f"  overall: top-1 {np.mean(np.array(ranks) == 1) * 100:.1f}%  mean rank {np.mean(ranks):.2f}  ({len(ranks)} pairs)")
    json.dump(retr, open(os.path.join(out_dir, "retrieval.json"), "w"), indent=1)
    skipped = [o for o in dict.fromkeys(g["obj_id"] for g in all_grasps) if o not in data]
    json.dump(dict(vars(a), parts=vocab, n_objects=len(data), n_grasps=len(grasps), skipped=skipped),
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1, ensure_ascii=False)

    # 3) agreement with the per-object point-wise F4 scorer (same reference grasp, same object)
    if a.compare and os.path.exists(a.compare):
        old = json.load(open(a.compare))
        rhos = []
        for oid, d in data.items():
            o = {x["gid"]: x for x in old if x["obj_id"] == oid}
            if not any(x["is_ref"] for x in o.values()):
                continue
            ref_gid_o = next(x["gid"] for x in o.values() if x["is_ref"])
            r, _ = score_all(oid, ref_gid_o)
            g = [i for i, x in enumerate(d["gs"]) if x["gid"] != ref_gid_o]
            if len(g) > 2:
                rhos.append(spearmanr([o[d["gs"][i]["gid"]]["total"] for i in g], r[oid]["total"][g].numpy()).correlation)
        print(f"\nsame-object agreement with point-wise F4 scorer (same refs): Spearman mean {np.nanmean(rhos):.3f}, "
              f"min {np.nanmin(rhos):.3f} over {len(rhos)} objects")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape_dir", default=SHAPE_DIR)
    ap.add_argument("--oakbase", default=OAKBASE_DIR)
    ap.add_argument("--mano_root", default=MANO_ROOT)
    ap.add_argument("--cat", default="teapot")
    ap.add_argument("--ref", default="", help="<obj_id>/<grasp_id> of the single reference (default: first use grasp on a real object)")
    ap.add_argument("--f1_loosen", type=float, default=1.5, help="scales the F1 tolerance (FactorConfig.f1_temp)")
    ap.add_argument("--gate_tau", type=float, default=0.4, help="<=0: old total sqrt(F1*F4p)")
    ap.add_argument("--gate_width", type=float, default=0.05)
    ap.add_argument("--afford_parts", default="", help="comma list, e.g. handle: score contact on these parts only")
    ap.add_argument("--ref_part_min", type=float, default=0.10, help="parts with at least this share of the ref contact are required")
    ap.add_argument("--min_part_ratio", type=float, default=0.0, help="candidate share on a required part >= ratio * ref share; 0 disables")
    ap.add_argument("--n_obj_pts", type=int, default=2048)
    ap.add_argument("--compare", default="", help="factor_scores.json of an older point-wise F4 run for the same category")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="")
    main(ap.parse_args())
