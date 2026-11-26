"""Hand-agnostic, reference-conditioned Point Transformer grasp scorer.

Distilled from the analytic scorer (total = F1 * gate(F4p) * F4d).
Operates strictly on HandState + object point clouds + binary affordance masks
(no MANO parameters), preserves left/right handedness via palm frame + sign
feature, and is SE(3)-invariant to global hand-object pose.

At RL training time, the reference grasp is encoded ONCE; scoring a batch of B
candidate grasps only encodes candidates and runs the lightweight comparison
head (~2 ms for B=1024 in fp16).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .factors import palm_frame
from .hands import PARTS, HandState

NUM_HAND_PARTS = len(PARTS)  # 16


def extract_grasp_tokens(
    s: HandState,
    sign: float,
    obj_pts: torch.Tensor,       # (B, M, 3) or (M, 3)
    afford_mask: torch.Tensor,   # (B, M) or (M,) float in {0, 1}
    n_obj_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build SE(3)-invariant hand-part tokens (B, 16, 8) and object tokens (B, K, 5).

    Supports both single shared object (M, 3) and per-candidate objects (B, M, 3).
    """
    dev = s.points.device
    B = s.points.shape[0]
    if obj_pts.ndim == 2:
        obj_pts = obj_pts.unsqueeze(0).expand(B, -1, -1)
    if afford_mask.ndim == 1:
        afford_mask = afford_mask.unsqueeze(0).expand(B, -1)
    obj_pts = obj_pts.to(dev, dtype=torch.float32)
    afford_mask = afford_mask.to(dev, dtype=torch.float32)

    # Subsample object points deterministically to n_obj_tokens (stratified)
    M = obj_pts.shape[1]
    if M > n_obj_tokens:
        idx = torch.linspace(0, M - 1, n_obj_tokens, device=dev).long()
        obj_sub = obj_pts[:, idx]       # (B, K, 3)
        aff_sub = afford_mask[:, idx]   # (B, K)
    else:
        obj_sub = obj_pts
        aff_sub = afford_mask

    # Palm frame (R: B,3,3; L: B)
    R, L = palm_frame(s, sign)
    L_safe = L.clamp(min=1e-4)[:, None, None]  # (B, 1, 1)
    wrist = s.wrist[:, None, :]                # (B, 1, 3)

    # Hand points in palm frame (normalized by palm length L)
    hand_palm = torch.einsum("bij,bni->bnj", R, s.points - wrist) / L_safe  # (B, N_h, 3)
    obj_palm = torch.einsum("bij,bki->bkj", R, obj_sub - wrist) / L_safe    # (B, K, 3)

    # Pairwise distances between hand points and subsampled object points: (B, N_h, K)
    dist_mat = torch.cdist(s.points.float(), obj_sub.float())  # (B, N_h, K)
    # Per-object-token min distance to hand
    d_obj_to_hand = dist_mat.min(dim=1).values.clamp(max=0.15) / 0.15  # (B, K)

    # Mask non-affordance points with large distance for affordance-specific min distance
    has_aff = aff_sub.sum(dim=1, keepdim=True) > 0.5  # (B, 1)
    aff_bool = aff_sub > 0.5                          # (B, K)
    big = torch.full_like(dist_mat, 1.0)
    dist_aff_mat = torch.where(aff_bool[:, None, :], dist_mat, big)

    # Aggregate to 16 semantic hand parts
    pid = s.part_ids.to(dev)  # (N_h,)
    centroids = torch.zeros(B, NUM_HAND_PARTS, 3, device=dev, dtype=torch.float32)
    d_part_obj = torch.full((B, NUM_HAND_PARTS), 0.15, device=dev, dtype=torch.float32)
    d_part_aff = torch.full((B, NUM_HAND_PARTS), 0.15, device=dev, dtype=torch.float32)

    for k in range(NUM_HAND_PARTS):
        mask = pid == k
        if not mask.any():
            continue
        centroids[:, k] = hand_palm[:, mask].mean(dim=1)
        d_part_obj[:, k] = dist_mat[:, mask].amin(dim=(1, 2)).clamp(max=0.15)
        d_aff_k = dist_aff_mat[:, mask].amin(dim=(1, 2)).clamp(max=0.15)
        d_part_aff[:, k] = torch.where(has_aff[:, 0], d_aff_k, d_part_obj[:, k])

    c_obj = torch.exp(-((d_part_obj / 0.02) ** 2))
    c_aff = torch.exp(-((d_part_aff / 0.02) ** 2))
    sign_col = torch.full((B, NUM_HAND_PARTS, 1), float(sign), device=dev, dtype=torch.float32)

    hand_feat = torch.cat(
        [
            centroids,                              # 3
            (d_part_aff / 0.15).unsqueeze(-1),      # 1
            (d_part_obj / 0.15).unsqueeze(-1),      # 1
            c_aff.unsqueeze(-1),                    # 1
            c_obj.unsqueeze(-1),                    # 1
            sign_col,                               # 1
        ],
        dim=-1,
    )  # (B, 16, 8)

    obj_feat = torch.cat(
        [
            obj_palm,                               # 3
            aff_sub.unsqueeze(-1),                  # 1
            d_obj_to_hand.unsqueeze(-1),            # 1
        ],
        dim=-1,
    )  # (B, K, 5)
    return hand_feat, obj_feat


class CrossSelfBlock(nn.Module):
    """Pre-LN Cross-Attention (query attends to kv) + Self-Attention + FFN."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_q2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q1(q)
        kvn = self.norm_kv(kv)
        q = q + self.cross_attn(qn, kvn, kvn, need_weights=False)[0]
        qn2 = self.norm_q2(q)
        q = q + self.self_attn(qn2, qn2, qn2, need_weights=False)[0]
        q = q + self.ffn(self.norm_ff(q))
        return q


class GraspTransformerScorer(nn.Module):
    """Reference-conditioned Point Transformer grasp scorer.

    Predicts (total, f1, f4p, f4d) in [0, 1].
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        n_obj_tokens: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_obj_tokens = n_obj_tokens

        self.part_emb = nn.Embedding(NUM_HAND_PARTS, d_model)
        self.hand_proj = nn.Sequential(
            nn.Linear(8, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.obj_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.enc_blocks = nn.ModuleList(
            [CrossSelfBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers)]
        )
        self.enc_norm = nn.LayerNorm(d_model)

        # Reference comparison: candidate parts attend to reference parts + elementwise diff
        self.cmp_cross = CrossSelfBlock(d_model, nhead, dropout=dropout)
        self.cmp_part_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),  # [total, f1, f4p, f4d]
        )

    def encode(
        self, hand_feat: torch.Tensor, obj_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = hand_feat.shape[0]
        part_ids = torch.arange(NUM_HAND_PARTS, device=hand_feat.device)
        h = self.hand_proj(hand_feat) + self.part_emb(part_ids)[None, :, :]
        o = self.obj_proj(obj_feat)
        for blk in self.enc_blocks:
            h = blk(h, o)
        h = self.enc_norm(h)
        z = h.mean(dim=1)
        return h, z

    def score_from_encoded(
        self,
        h_cand: torch.Tensor,
        z_cand: torch.Tensor,
        h_ref: torch.Tensor,
        z_ref: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = h_cand.shape[0]
        if h_ref.shape[0] == 1 and B > 1:
            h_ref = h_ref.expand(B, -1, -1)
            z_ref = z_ref.expand(B, -1)

        h_ctx = self.cmp_cross(h_cand, h_ref)
        part_pair = torch.cat([h_ctx, h_ref, (h_ctx - h_ref).abs(), h_ctx * h_ref], dim=-1)
        part_fused = self.cmp_part_mlp(part_pair).mean(dim=1)  # (B, d)

        glob = torch.cat([part_fused, z_cand, z_ref, (z_cand - z_ref).abs(), z_cand * z_ref], dim=-1)
        out = torch.sigmoid(self.head(glob))
        return {
            "total": out[:, 0],
            "f1": out[:, 1],
            "f4p": out[:, 2],
            "f4d": out[:, 3],
        }

    def forward(
        self,
        cand_hand: torch.Tensor,
        cand_obj: torch.Tensor,
        ref_hand: torch.Tensor,
        ref_obj: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_c, z_c = self.encode(cand_hand, cand_obj)
        h_r, z_r = self.encode(ref_hand, ref_obj)
        return self.score_from_encoded(h_c, z_c, h_r, z_r)


class TransformerGraspScorer:
    """Drop-in fast GPU scorer for SimToolReal RL training."""

    def __init__(
        self,
        model: GraspTransformerScorer,
        ref_hand_feat: torch.Tensor,
        ref_obj_feat: torch.Tensor,
        device: str | torch.device = "cuda",
        use_fp16: bool = True,
    ):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            rh = ref_hand_feat.to(self.device)
            ro = ref_obj_feat.to(self.device)
            self.h_ref, self.z_ref = self.model.encode(rh, ro)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        ref_path: str,
        afford_parts: list[str] | tuple[str, ...] = ("grip",),
        device: str | torch.device = "cuda",
        use_fp16: bool = True,
    ) -> "TransformerGraspScorer":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m_cfg = ckpt.get("model_cfg", {})
        model = GraspTransformerScorer(**m_cfg)
        model.load_state_dict(ckpt["state_dict"])

        ref = torch.load(ref_path, map_location="cpu", weights_only=False)
        s_ref = HandState(
            points=ref["points"].float(),
            part_ids=ref["part_ids"].long(),
            wrist=ref["wrist"].float(),
            fingers={k: v.float() for k, v in ref["fingers"].items()},
        )
        obj_ref = ref["obj"].float()
        if obj_ref.ndim == 3:
            obj_ref = obj_ref[0]
        part_names = ref.get("part_names", [])
        lab = ref["lab"].long()
        aff_ids = [i for i, n in enumerate(part_names) if n in afford_parts]
        if aff_ids:
            aff_mask = torch.isin(lab, torch.tensor(aff_ids)).float()
        else:
            aff_mask = torch.ones_like(lab, dtype=torch.float32)

        rh, ro = extract_grasp_tokens(
            s_ref, float(ref["sign"]), obj_ref, aff_mask, n_obj_tokens=model.n_obj_tokens
        )
        return cls(model, rh, ro, device=device, use_fp16=use_fp16)

    @torch.no_grad()
    def score(
        self,
        s: HandState,
        sign: float,
        obj_pts: torch.Tensor,
        afford_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hc_in, oc_in = extract_grasp_tokens(
            s, sign, obj_pts, afford_mask, n_obj_tokens=self.model.n_obj_tokens
        )
        if self.use_fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                h_c, z_c = self.model.encode(hc_in, oc_in)
                out = self.model.score_from_encoded(h_c, z_c, self.h_ref, self.z_ref)
            return {k: v.float() for k, v in out.items()}
        h_c, z_c = self.model.encode(hc_in, oc_in)
        return self.model.score_from_encoded(h_c, z_c, self.h_ref, self.z_ref)
