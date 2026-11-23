"""One reference grasp -> batch scorer for any hand (MANO or URDF) on any object with part labels.

    total = F1 * sigmoid((F4p - gate_tau) / gate_width) * F4d      (contact_term="dist", default)
    total = F1 * sigmoid(...) * min(share / share_ref, 1)           (contact_term="share")
F1 is computed in each hand's own palm frame (so a left hand must be compared with a left-hand reference, e.g. a
mirrored right-hand grasp); F4p / F4d only use distances to labelled object parts, so the reference and the
candidate can be on different object instances in different frames.
"""
from typing import List, Optional

import torch

from .factors import FactorConfig, f1_shape, soft_iou
from .hands import HandState
from .score_parts import expand, part_shares


class GraspScorer:
    def __init__(self, s_ref: HandState, sign_ref: float, ref_obj, ref_obj_n, ref_lab, vocab: List[str],
                 afford_parts: List[str], f1_loosen=1.5, gate_tau=0.4, gate_width=0.05, contact_term="dist",
                 dist_delta=5000.0, dist_sigma=0.01, dist_clip=0.05, cfg: Optional[FactorConfig] = None, device="cpu"):
        self.cfg = cfg or FactorConfig()
        self.cfg.f1_temp *= f1_loosen
        self.vocab, self.device = list(vocab), device
        self.aff = [self.vocab.index(p) for p in afford_parts]
        self.gate_tau, self.gate_width, self.contact_term = gate_tau, gate_width, contact_term
        self.dist_delta, self.dist_sigma, self.dist_clip = dist_delta, dist_sigma, dist_clip
        self.s_ref, self.sign_ref = s_ref, sign_ref
        P, _, D, _ = self._contact(s_ref, ref_obj, ref_obj_n, ref_lab)
        self.P_ref, self.D_ref = P, D[:, :, self.aff].amin(-1).clamp(max=dist_clip)

    def _contact(self, s, obj, obj_n, lab):
        obj = torch.as_tensor(obj, dtype=torch.float32, device=self.device).view(1, -1, 3)
        obj_n = torch.as_tensor(obj_n, dtype=torch.float32, device=self.device).view(1, -1, 3)
        onehot = torch.nn.functional.one_hot(torch.as_tensor(lab, device=self.device).long(), len(self.vocab)).float()
        return part_shares(s, obj, obj_n, onehot, self.cfg)

    def score(self, s: HandState, sign: float, obj, obj_n, lab):
        """s: batch of candidate hands on one object (points (M,3), normals (M,3), part labels (M,) in vocab)."""
        B = s.points.shape[0]
        f1 = f1_shape(s, expand(self.s_ref, B), sign, self.sign_ref, self.cfg)
        P, _, D, _ = self._contact(s, obj, obj_n, lab)
        Pa, Pra = P[:, :, self.aff], self.P_ref[:, :, self.aff]
        share = (Pa.sum((1, 2)) / Pra.sum((1, 2)).clamp_min(1e-9)).clamp(max=1)
        f4p = soft_iou(Pa / Pa.sum((1, 2), keepdim=True).clamp_min(1e-9),
                       (Pra / Pra.sum((1, 2), keepdim=True).clamp_min(1e-9)).expand(B, -1, -1), (1, 2))
        gate = torch.sigmoid((f4p - self.gate_tau) / self.gate_width)
        Dc, Dr = D[:, :, self.aff].amin(-1).clamp(max=self.dist_clip), self.D_ref
        w = torch.maximum(torch.exp(-self.dist_delta * Dr ** 2), torch.exp(-self.dist_delta * Dc ** 2))
        f4d = torch.exp(-(w * (Dc - Dr) ** 2).sum(1) / w.sum(1).clamp_min(1e-9) / self.dist_sigma ** 2)
        disc = f4d if self.contact_term == "dist" else share
        return dict(total=f1 * gate * disc, f1=f1, f4p=f4p, gate=gate, f4d=f4d, share=share, dist=Dc)
