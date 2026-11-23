"""Hand-agnostic hand interface.

Every hand (MANO or any URDF dexterous hand) exposes the same things:
  * surface points (B, N, 3) with a semantic part id per point (shared PARTS vocabulary)
  * wrist position (B, 3)
  * per-finger keypoint chains {finger: (B, K, 3)}, ordered base -> tip
Scoring code only uses these, so a new hand only needs a part / finger mapping config.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

FINGERS = ["thumb", "index", "middle", "ring", "little"]
PARTS = ["palm"] + [f"{f}_{i}" for f in FINGERS for i in range(3)]  # 0 = proximal ... 2 = distal
GROUPS = ["palm"] + FINGERS


def part_group(part: str) -> str:
    return part.split("_")[0]


@dataclass
class HandState:
    points: torch.Tensor  # B, N, 3
    part_ids: torch.Tensor  # N (index into PARTS)
    wrist: torch.Tensor  # B, 3
    fingers: Dict[str, torch.Tensor]  # finger -> B, K, 3 (base ... tip)

    @property
    def parts(self) -> List[str]:
        return [PARTS[i] for i in sorted(set(self.part_ids.tolist()))]

    @property
    def groups(self) -> List[str]:
        return sorted({part_group(p) for p in self.parts}, key=GROUPS.index)


class HandModel:
    name = "hand"
    palm_normal_sign = 1.0  # flips palm-frame z so that +z points out of the palm (towards the object)

    def __call__(self, **params) -> HandState:
        raise NotImplementedError


# MANO joint index (original 16-joint order) -> part. The skinning weight of joint j moves the segment distal to j.
_MANO16_PART = {0: "palm"}
for _f, _base in [("index", 1), ("middle", 4), ("little", 7), ("ring", 10), ("thumb", 13)]:
    for _i in range(3):
        _MANO16_PART[_base + _i] = f"{_f}_{_i}"
# manotorch 21-joint (SNAP) order: 0 wrist, thumb 1-4, index 5-8, middle 9-12, ring 13-16, little 17-20
_MANO21_CHAIN = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
                 "little": [17, 18, 19, 20]}


class MANOHand(HandModel):
    name = "mano"

    def __init__(self, mano_root, side="right", device="cpu"):
        for k, v in dict(bool=bool, int=int, float=float, complex=complex, object=object, unicode=str, str=str).items():
            if k not in np.__dict__:  # chumpy (used to read the MANO pkl) needs numpy<1.24 aliases
                setattr(np, k, v)
        from manotorch.manolayer import ManoLayer
        self.layer = ManoLayer(center_idx=0, mano_assets_root=mano_root, side=side).to(device)
        self.device = device
        joint_of_vert = self.layer.th_weights.argmax(1).cpu().tolist()
        self.part_ids = torch.tensor([PARTS.index(_MANO16_PART[j]) for j in joint_of_vert])
        self.faces = self.layer.th_faces

    @torch.no_grad()
    def __call__(self, pose, shape, tsl) -> HandState:
        pose, shape, tsl = pose.to(self.device), shape.to(self.device), tsl.to(self.device)
        out = self.layer(pose, shape)
        verts = out.verts + tsl[:, None]
        joints = out.joints + tsl[:, None]
        return HandState(points=verts, part_ids=self.part_ids, wrist=joints[:, 0],
                         fingers={f: joints[:, idx] for f, idx in _MANO21_CHAIN.items()})


@dataclass
class URDFHandConfig:
    """Mapping from a robot hand's URDF to the shared vocabulary.

    link_parts:    link name -> part in PARTS (links not listed are ignored, e.g. mounting flanges)
    finger_chains: finger name (subset of FINGERS) -> link names from finger base to distal link
    tip_offsets:   finger -> xyz offset of the fingertip in the distal link frame
    wrist_link:    link whose origin is used as the wrist
    """
    urdf_path: str
    link_parts: Dict[str, str]
    finger_chains: Dict[str, List[str]]
    wrist_link: str
    tip_offsets: Dict[str, List[float]] = field(default_factory=dict)
    points_per_link: int = 128
    palm_normal_sign: float = 1.0


class URDFHand(HandModel):
    """Generic URDF hand via pytorch_kinematics. Not yet validated on a real robot hand."""

    def __init__(self, cfg: URDFHandConfig, device="cpu"):
        import os
        import pytorch_kinematics as pk
        import trimesh
        self.cfg = cfg
        self.name = os.path.splitext(os.path.basename(cfg.urdf_path))[0]
        self.palm_normal_sign = cfg.palm_normal_sign
        self.chain = pk.build_chain_from_urdf(open(cfg.urdf_path, "rb").read()).to(device=device)
        root = os.path.dirname(cfg.urdf_path)
        pts, pids, links = [], [], []
        for frame_name in self.chain.get_frame_names():
            link = self.chain.find_frame(frame_name).link
            if link.name not in cfg.link_parts:
                continue
            for vis in link.visuals:
                if vis.geom_type != "mesh":
                    continue
                fname, scale = vis.geom_param[0], vis.geom_param[1]
                fname = fname.replace("package://", "")
                mesh = trimesh.load(os.path.join(root, fname), force="mesh")
                if scale is not None:
                    mesh.apply_scale(np.asarray(scale))
                p, _ = trimesh.sample.sample_surface(mesh, cfg.points_per_link, seed=0)
                p = vis.offset.transform_points(torch.tensor(p, dtype=torch.float32, device=device))
                pts.append(p)
                pids += [PARTS.index(cfg.link_parts[link.name])] * len(p)
                links += [link.name] * len(p)
        self.local_points = torch.cat(pts)
        self.part_ids = torch.tensor(pids)
        self.point_links = links

    @torch.no_grad()
    def __call__(self, q, root_rot=None, root_tsl=None) -> HandState:
        """q: B, DoF joint angles; root_rot: B, 3, 3; root_tsl: B, 3 (hand base pose in object frame)."""
        B = q.shape[0]
        tf = self.chain.forward_kinematics(q)
        mats = {k: v.get_matrix() for k, v in tf.items()}  # B, 4, 4
        R = root_rot if root_rot is not None else torch.eye(3).expand(B, 3, 3)
        t = root_tsl if root_tsl is not None else torch.zeros(B, 3)

        def to_obj(m, p):  # p: n,3 in link frame -> B,n,3 in object frame
            p = p @ m[:, :3, :3].transpose(1, 2) + m[:, None, :3, 3]
            return p @ R.transpose(1, 2) + t[:, None]

        pts = torch.zeros(B, len(self.point_links), 3)
        for name in set(self.point_links):
            idx = [i for i, l in enumerate(self.point_links) if l == name]
            pts[:, idx] = to_obj(mats[name], self.local_points[idx])
        fingers = {}
        for f, chain_links in self.cfg.finger_chains.items():
            kps = [to_obj(mats[l], torch.zeros(1, 3))[:, 0] for l in chain_links]
            off = torch.tensor(self.cfg.tip_offsets.get(f, [0.0, 0.0, 0.0]))[None]
            kps.append(to_obj(mats[chain_links[-1]], off)[:, 0])
            fingers[f] = torch.stack(kps, 1)
        wrist = to_obj(mats[self.cfg.wrist_link], torch.zeros(1, 3))[:, 0]
        return HandState(points=pts, part_ids=self.part_ids, wrist=wrist, fingers=fingers)
