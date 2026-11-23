"""Keypoint retargeting of a (MANO) HandState onto a URDFHand: wrist + finger chains, fingertips weighted x3."""
import torch
import torch.nn.functional as F

from .factors import palm_frame
from .hands import HandState, URDFHand


def _rot6d(x):
    a = F.normalize(x[:, :3], dim=-1)
    b = F.normalize(x[:, 3:] - (a * x[:, 3:]).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def retarget(hand: URDFHand, target: HandState, sign_target: float, steps=400, lr=0.02, tip_weight=3.0):
    """returns q (B, DoF), root_rot (B,3,3), root_tsl (B,3) and the final keypoint RMS error (B,) in metres."""
    B, dev = target.wrist.shape[0], target.wrist.device
    fingers = [f for f in target.fingers if f in hand.cfg.finger_chains]
    u = torch.zeros(B, len(hand.joint_names), device=dev, requires_grad=True)
    lo, hi = hand.q_lower.to(dev), hand.q_upper.to(dev)
    q_of = lambda: lo + (hi - lo) * torch.sigmoid(u - 1.0)
    with torch.no_grad():
        s0 = hand(q_of())
        Rs, _ = palm_frame(s0, hand.palm_normal_sign)
        Rt, _ = palm_frame(target, sign_target)
        R0 = Rt @ Rs.transpose(1, 2)
    r6 = R0[:, :, :2].transpose(1, 2).reshape(B, 6).clone().requires_grad_(True)
    t = (target.wrist - (R0 @ s0.wrist[..., None])[..., 0]).clone().requires_grad_(True)
    opt = torch.optim.Adam([u, r6, t], lr=lr)
    tgt = torch.cat([target.wrist[:, None]] + [target.fingers[f] for f in fingers], 1)
    w = torch.ones(tgt.shape[1], device=dev)
    k = 1
    for f in fingers:
        k += target.fingers[f].shape[1]
        w[k - 1] = tip_weight
    for i in range(steps):
        s = hand(q_of(), _rot6d(r6), t)
        kp = torch.cat([s.wrist[:, None]] + [s.fingers[f] for f in fingers], 1)
        err = ((kp - tgt) ** 2).sum(-1)
        loss = (err * w).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return q_of().detach(), _rot6d(r6).detach(), t.detach(), err.mean(1).sqrt().detach()
