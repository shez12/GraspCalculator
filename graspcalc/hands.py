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
    normals: Optional[torch.Tensor] = None  # B, N, 3 outward surface normals (optional; enables the penetration term)

    @property
    def parts(self) -> List[str]:
        return [PARTS[i] for i in sorted(set(self.part_ids.tolist()))]

    @property
    def groups(self) -> List[str]:
        return sorted({part_group(p) for p in self.parts}, key=GROUPS.index)


def vertex_normals(verts, faces):
    """Outward vertex normals of a closed, consistently wound mesh: B, N, 3."""
    f = faces.long()
    v0, v1, v2 = verts[:, f[:, 0]], verts[:, f[:, 1]], verts[:, f[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vn = torch.zeros_like(verts)
    for i in range(3):
        vn.index_add_(1, f[:, i], fn)
    vn = torch.nn.functional.normalize(vn, dim=-1)
    out = ((verts - verts.mean(1, keepdim=True)) * vn).sum(-1).mean(1)  # flip if the winding points inwards
    return vn * torch.where(out < 0, -1.0, 1.0)[:, None, None]


class HandModel:
    name = "hand"
    side = "right"
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
        self.device, self.side = device, side
        self.palm_normal_sign = 1.0 if side == "right" else -1.0
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
                         fingers={f: joints[:, idx] for f, idx in _MANO21_CHAIN.items()},
                         normals=vertex_normals(verts, self.faces.to(verts.device)))


@dataclass
class URDFHandConfig:
    """Maps a URDF hand onto the shared hand interface.
    link_parts:    link name -> part in PARTS (links not listed are ignored, e.g. mounting flanges)
    finger_chains: finger name (subset of FINGERS) -> link names whose origins are the keypoints, base ... tip
    tip_offsets:   finger -> xyz offset of an extra fingertip keypoint in the last chain link's frame
    wrist_link:    link whose origin is used as the wrist
    side:          "left" or "right"; the palm frame of a left hand is mirrored, so its palm_normal_sign is negated
    """
    urdf_path: str
    link_parts: Dict[str, str]
    finger_chains: Dict[str, List[str]]
    wrist_link: str
    tip_offsets: Dict[str, List[float]] = field(default_factory=dict)
    points_per_link: int = 128
    side: str = "right"
    palm_normal_sign: float = 1.0
    joint_names: Optional[List[str]] = None  # order of q; default = URDF order of the movable joints


def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, :3] = _rpy_matrix(*[float(v) for v in o.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


def _axis_angle(axis, q):
    """axis: 3, q: B -> B,3,3 (Rodrigues)."""
    K = torch.zeros(3, 3, dtype=q.dtype, device=q.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -axis[2], axis[1], axis[2], -axis[0], -axis[1], axis[0]
    s, c = torch.sin(q)[:, None, None], torch.cos(q)[:, None, None]
    return torch.eye(3, dtype=q.dtype, device=q.device) + s * K + (1 - c) * (K @ K)


class URDFHand(HandModel):
    """Any URDF hand. Surface points are sampled per link (collision mesh, else visual mesh) and moved either by
    the built-in differentiable forward kinematics (__call__) or by link poses read from a simulator
    (from_link_poses), so the same scorer runs offline and inside Isaac Lab / MuJoCo."""

    def __init__(self, cfg: URDFHandConfig, device="cpu"):
        import os
        import xml.etree.ElementTree as ET
        import trimesh
        self.cfg, self.device, self.side = cfg, device, cfg.side
        self.name = os.path.splitext(os.path.basename(cfg.urdf_path))[0]
        self.palm_normal_sign = cfg.palm_normal_sign * (-1.0 if cfg.side == "left" else 1.0)
        root_dir = os.path.dirname(cfg.urdf_path)
        robot = ET.parse(cfg.urdf_path).getroot()
        links = {l.get("name"): l for l in robot.findall("link")}
        self.joints = []  # (name, type, parent, child, T_origin, axis)
        self.limits = {}
        for j in robot.findall("joint"):
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0.0, 0.0]
            lim = j.find("limit")
            if j.get("type") != "fixed":
                self.limits[j.get("name")] = (float(lim.get("lower", -np.pi)), float(lim.get("upper", np.pi))) \
                    if lim is not None else (-np.pi, np.pi)
            self.joints.append((j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                                torch.tensor(_origin(j), dtype=torch.float32, device=device),
                                torch.tensor(axis, dtype=torch.float32, device=device)))
        children = {c for _, _, _, c, _, _ in self.joints}
        self.root_link = next(n for n in links if n not in children)
        order, frontier = [], [self.root_link]  # joints in topological order
        while frontier:
            p = frontier.pop(0)
            for jt in self.joints:
                if jt[2] == p:
                    order.append(jt)
                    frontier.append(jt[3])
        self.joints = order
        movable = [jt[0] for jt in self.joints if jt[1] in ("revolute", "continuous", "prismatic")]
        self.joint_names = cfg.joint_names or movable
        self.q_lower = torch.tensor([self.limits[n][0] for n in self.joint_names], device=device)
        self.q_upper = torch.tensor([self.limits[n][1] for n in self.joint_names], device=device)
        pts, pids, plinks = [], [], []
        for lname, part in cfg.link_parts.items():
            el = links[lname]
            geoms = el.findall("collision") or el.findall("visual")
            for g in geoms:
                m = g.find("geometry/mesh")
                if m is None:
                    continue
                fname = m.get("filename").replace("package://", "")
                path = os.path.join(root_dir, fname)
                if not os.path.exists(path):  # package://<pkg>/meshes/... -> meshes/...
                    path = os.path.join(root_dir, *fname.split("/")[1:])
                mesh = trimesh.load(path, force="mesh")
                if m.get("scale"):
                    mesh.apply_scale(np.array([float(v) for v in m.get("scale").split()]))
                mesh.apply_transform(_origin(g))
                p, _ = trimesh.sample.sample_surface_even(mesh, cfg.points_per_link, seed=0)
                pts.append(torch.tensor(np.asarray(p), dtype=torch.float32))
                pids += [PARTS.index(part)] * len(p)
                plinks += [lname] * len(p)
        self.local_points = torch.cat(pts).to(device)
        self.part_ids = torch.tensor(pids)
        self.point_links = plinks
        self.point_link_names = list(dict.fromkeys(plinks))
        self.point_link_index = torch.tensor([self.point_link_names.index(l) for l in plinks], device=device)
        self.keypoint_links = sorted({l for c in cfg.finger_chains.values() for l in c} | {cfg.wrist_link})
        self.required_links = sorted(set(self.point_link_names) | set(self.keypoint_links))

    def fk(self, q, root_rot=None, root_tsl=None) -> Dict[str, torch.Tensor]:
        """q: B, DoF (order self.joint_names) -> {link: B,4,4} in the frame given by root_rot / root_tsl."""
        B = q.shape[0]
        T0 = torch.eye(4, dtype=q.dtype, device=q.device).repeat(B, 1, 1)
        if root_rot is not None:
            T0[:, :3, :3] = root_rot
        if root_tsl is not None:
            T0[:, :3, 3] = root_tsl
        T = {self.root_link: T0}
        col = {n: i for i, n in enumerate(self.joint_names)}
        for name, jtype, parent, child, To, axis in self.joints:
            Tj = T[parent] @ To
            if name in col and jtype in ("revolute", "continuous"):
                M = torch.eye(4, dtype=q.dtype, device=q.device).repeat(B, 1, 1)
                M[:, :3, :3] = _axis_angle(axis, q[:, col[name]])
                Tj = Tj @ M
            elif name in col and jtype == "prismatic":
                M = torch.eye(4, dtype=q.dtype, device=q.device).repeat(B, 1, 1)
                M[:, :3, 3] = axis * q[:, col[name], None]
                Tj = Tj @ M
            T[child] = Tj
        return T

    def fill_fixed_links(self, link_T: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds links a simulator merged away (fixed-joint children, e.g. pads / fingertip frames) from their parent."""
        out = dict(link_T)
        for name, jtype, parent, child, To, axis in self.joints:
            if child not in out and jtype == "fixed" and parent in out:
                out[child] = out[parent] @ To.to(out[parent])
        return out

    def from_link_poses(self, link_T: Dict[str, torch.Tensor]) -> HandState:
        """link_T: {link: B,4,4} (e.g. built from simulator body poses); needs self.required_links."""
        link_T = self.fill_fixed_links(link_T)
        Ts = torch.stack([link_T[l] for l in self.point_link_names], 1)  # B,L,4,4
        Tp = Ts[:, self.point_link_index]  # B,N,4,4
        pts = (Tp[..., :3, :3] @ self.local_points[None, :, :, None])[..., 0] + Tp[..., :3, 3]
        fingers = {}
        for f, chain_links in self.cfg.finger_chains.items():
            kps = [link_T[l][:, :3, 3] for l in chain_links]
            if f in self.cfg.tip_offsets:
                Tl = link_T[chain_links[-1]]
                off = torch.tensor(self.cfg.tip_offsets[f], dtype=Tl.dtype, device=Tl.device)
                kps.append(Tl[:, :3, :3] @ off + Tl[:, :3, 3])
            fingers[f] = torch.stack(kps, 1)
        return HandState(points=pts, part_ids=self.part_ids, wrist=link_T[self.cfg.wrist_link][:, :3, 3],
                         fingers=fingers)

    def __call__(self, q, root_rot=None, root_tsl=None) -> HandState:
        return self.from_link_poses(self.fk(q, root_rot, root_tsl))


def mirror(s: HandState, axis: int = 0) -> HandState:
    """Reflects a hand state through the plane normal to `axis` (a right-hand grasp becomes a left-hand grasp).
    Distances to an object reflected the same way are unchanged; the palm frame of the result is left-handed."""
    m = torch.ones(3, dtype=s.points.dtype, device=s.points.device)
    m[axis] = -1
    return HandState(points=s.points * m, part_ids=s.part_ids, wrist=s.wrist * m,
                     fingers={k: v * m for k, v in s.fingers.items()},
                     normals=None if s.normals is None else s.normals * m)
