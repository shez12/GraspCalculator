"""Timing of the handle-only part scorer (score_parts.py): MANO forward + contact + F1 + F4p, reference cached."""
import argparse
import time

import torch

from .factors import FactorConfig, f1_shape, soft_iou
from .hands import MANOHand
from .oakink import load_real_grasps, stack
from .paths import MANO_ROOT, OAKBASE_DIR, SHAPE_DIR
from .score_parts import expand, index, load_obj_oakbase, part_labels, part_shares


def main(a):
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    dev = a.device
    cfg = FactorConfig()
    cfg.f1_temp *= 1.5
    hand = MANOHand(a.mano_root, device=dev)
    sign = hand.palm_normal_sign
    obj, obj_n, name = load_obj_oakbase(a.shape_dir, a.oakbase, a.cat, a.obj, 2048)
    names, lab, _ = part_labels(a.oakbase, a.cat, name, obj[0].numpy())
    onehot = torch.nn.functional.one_hot(torch.as_tensor(lab), len(names)).float().to(dev)
    aff = [names.index("handle")]
    obj, obj_n = obj.to(dev), obj_n.to(dev)
    gs = [g for g in load_real_grasps(a.shape_dir) if g["obj_id"] == a.obj]
    s_ref = hand(stack("pose", gs[:1]), stack("shape", gs[:1]), stack("tsl", gs[:1]))
    P_ref, _ = part_shares(s_ref, obj, obj_n, onehot, cfg)
    Pra = P_ref[:, :, aff]
    Pra = Pra / Pra.sum((1, 2), keepdim=True).clamp_min(1e-9)

    def run(B):
        idx = torch.arange(B) % len(gs)
        pose, shape, tsl = stack("pose", gs)[idx], stack("shape", gs)[idx], stack("tsl", gs)[idx]
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
        ts = {"mano": [], "score": []}
        for it in range(a.warmup + a.iters):
            sync(); t0 = time.perf_counter()
            st = hand(pose, shape, tsl)
            sync(); t1 = time.perf_counter()
            P, _ = part_shares(st, obj, obj_n, onehot, cfg, chunk=a.chunk)
            s1 = f1_shape(st, expand(s_ref, B), sign, sign, cfg)
            Pa = P[:, :, aff]
            s4 = soft_iou(Pa / Pa.sum((1, 2), keepdim=True).clamp_min(1e-9), Pra.expand(B, -1, -1), (1, 2))
            total = s1 * torch.sigmoid((s4 - 0.4) / 0.05)
            total.cpu()
            sync(); t2 = time.perf_counter()
            if it >= a.warmup:
                ts["mano"].append(t1 - t0)
                ts["score"].append(t2 - t1)
        m, s = [sorted(ts[k])[len(ts[k]) // 2] * 1000 for k in ("mano", "score")]
        print(f"{dev:<5} B={B:<5} total {m + s:8.2f} ms/batch  {(m + s) / B:7.3f} ms/grasp  {B / (m + s) * 1000:8.0f} grasp/s"
              f"   (MANO {m:.2f} ms, contact+F1+F4p {s:.2f} ms)")

    for B in a.batches:
        run(B)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape_dir", default=SHAPE_DIR)
    ap.add_argument("--oakbase", default=OAKBASE_DIR)
    ap.add_argument("--mano_root", default=MANO_ROOT)
    ap.add_argument("--cat", default="mug")
    ap.add_argument("--obj", default="C10001")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    main(ap.parse_args())
