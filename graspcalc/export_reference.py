"""Export one OakInk grasp as a self-contained reference file (hand state + labelled object; MANO not needed to load).

python -m graspcalc.export_reference --cat hammer --ref C22001/109708ca1a --side left --out ref_hammer_left.pt
--side left mirrors the right-hand OakInk grasp and its object through the x = 0 plane.
"""
import argparse

import torch

from .hands import MANOHand, mirror
from .oakink import load_real_grasps, stack
from .paths import MANO_ROOT, OAKBASE_DIR, SHAPE_DIR
from .score_parts import load_obj_oakbase, oakbase_cat, part_labels


def main(a):
    oid, gid = a.ref.split("/")
    g = [g for g in load_real_grasps(SHAPE_DIR) if g["obj_id"] == oid and g["gid"] == gid]
    hand = MANOHand(MANO_ROOT)
    with torch.no_grad():
        s = hand(stack("pose", g), stack("shape", g), stack("tsl", g))
    cat = oakbase_cat(OAKBASE_DIR, a.cat)
    obj, obj_n, name = load_obj_oakbase(SHAPE_DIR, OAKBASE_DIR, cat, oid, a.n_obj_pts)
    names, lab, _ = part_labels(OAKBASE_DIR, cat, name, obj[0].numpy())
    obj, obj_n, sign = obj[0], obj_n[0], hand.palm_normal_sign
    if a.side == "left":
        s, sign = mirror(s), -sign
        m = torch.tensor([-1.0, 1.0, 1.0])
        obj, obj_n = obj * m, obj_n * m
    torch.save(dict(points=s.points, part_ids=s.part_ids, wrist=s.wrist, fingers=s.fingers, sign=sign, side=a.side,
                    obj=obj, obj_n=obj_n, lab=torch.as_tensor(lab), vocab=names, ref=a.ref, cat=a.cat), a.out)
    print(f"saved {a.out}: {a.ref} side={a.side} parts={names}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True)
    ap.add_argument("--ref", required=True, help="obj_id/gid")
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--n_obj_pts", type=int, default=2048)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
