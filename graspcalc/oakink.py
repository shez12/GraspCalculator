"""OakInk-Shape loading: real / transferred (virtual) grasps and object meshes."""
import glob
import json
import os
import pickle
import re

import numpy as np
import torch
import trimesh


INTENT_NAME = {"0001": "use", "0002": "hold", "0003": "liftup", "0004": "handover"}
SRC_RE = re.compile(r"(.+)/(.{6})_(.{4})_([_0-9]+)/([\-0-9]+)")


def load_real_grasps(shape_dir):
    """All real (recorded) grasps: oakink_shape_v2/<cat>/<obj_id>/<hash>/hand_param.pkl"""
    grasps = []
    for p in sorted(glob.glob(os.path.join(shape_dir, "oakink_shape_v2", "*", "*", "*", "hand_param.pkl"))):
        cat, obj_id, gid = p.split(os.sep)[-4:-1]
        src = open(os.path.join(os.path.dirname(p), "source.txt")).read()
        _, _, action_id, subject_id, _ = SRC_RE.findall(src)[0]
        hp = pickle.load(open(p, "rb"))
        grasps.append(dict(cat=cat, obj_id=obj_id, gid=gid, intent=INTENT_NAME[action_id], subject=subject_id,
                           pose=np.asarray(hp["pose"], np.float32), shape=np.asarray(hp["shape"], np.float32),
                           tsl=np.asarray(hp["tsl"], np.float32)))
    return grasps


def obj_dir(shape_dir, obj_id):
    """Real objects live in OakInkObjectsV2, virtual ones (o*) in OakInkVirtualObjectsV2 (same as oikit)."""
    for meta_f, root in (("object_id.json", "OakInkObjectsV2"), ("virtual_object_id.json", "OakInkVirtualObjectsV2")):
        meta = json.load(open(os.path.join(shape_dir, "metaV2", meta_f)))
        if obj_id in meta:
            return os.path.join(shape_dir, root, meta[obj_id]["name"], "align"), meta[obj_id]["name"]
    raise KeyError(obj_id)


def load_virtual_grasps(shape_dir, real):
    """Virtual grasps: oakink_shape_v2/<cat>/<real_obj>/<real_gid>/<virtual_obj>/hand_param.pkl,
    transferred from real grasp <real_gid>; intent/subject are inherited from it."""
    src = {(g["obj_id"], g["gid"]): g for g in real}
    out = []
    for p in sorted(glob.glob(os.path.join(shape_dir, "oakink_shape_v2", "*", "*", "*", "*", "hand_param.pkl"))):
        cat, robj, rgid, vobj = p.split(os.sep)[-5:-1]
        if (robj, rgid) not in src:
            continue
        hp = pickle.load(open(p, "rb"))
        g = src[(robj, rgid)]
        out.append(dict(cat=cat, obj_id=vobj, gid=rgid, intent=g["intent"], subject=g["subject"],
                        pose=np.asarray(hp["pose"], np.float32), shape=np.asarray(hp["shape"], np.float32),
                        tsl=np.asarray(hp["tsl"], np.float32)))
    return out


def strip_support_plate(mesh, obj_id="", min_frac=0.15, shrink=0.7):
    """Some virtual objects ship with a flat base plate / table sheet. Detect it as the largest planar patch
    (>min_frac of the area) lying at one extreme of the mesh, then drop the slab from that plane up to the height
    where the footprint shrinks below shrink x the plate span. The object frame is left unchanged."""
    n = mesh.face_normals.copy()
    c, a = mesh.triangles_center, mesh.area_faces
    k = np.argmax(np.abs(n), 1)
    n *= np.sign(n[np.arange(len(n)), k])[:, None]
    d = (n * c).sum(1)
    key = np.concatenate([np.round(n / 0.05), np.round(d / 0.003)[:, None]], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    area = np.bincount(inv, weights=a)
    j = np.argmax(area)
    if area[j] / mesh.area < min_frac:
        return mesh
    sel = inv == j
    pn = n[sel][0]
    h = c @ pn - d[sel].mean()
    if (h[~sel] > 0.001).mean() < (h[~sel] < -0.001).mean():
        h = -h
    if (h[~sel] > -0.001).mean() < 0.99:  # plane is not at an extreme -> a real flat face, keep
        return mesh
    u = np.linalg.svd(np.eye(3) - np.outer(pn, pn))[0][:, :2]
    span = lambda m: np.ptp(c[m] @ u, 0).max()
    span0, t = span(h < 0.001), 0.0
    for t in np.arange(0.001, 0.05, 0.001):
        band = (h >= t) & (h < t + 0.001)
        if band.any() and span(band) < shrink * span0:
            break
    keep = h >= t
    if span(keep) >= span0:  # the rest of the object is wider than the 'plate': it is the object's own flat bottom
        return mesh
    print(f"[plate] {obj_id}: removed support plate ({area[j] / mesh.area * 100:.0f}% of area, "
          f"{span0 * 1000:.0f} mm wide, {t * 1000:.0f} mm thick)")
    out = mesh.copy()
    out.update_faces(keep)
    out.remove_unreferenced_vertices()
    return out


def load_mesh(shape_dir, obj_id, strip=True):
    d, name = obj_dir(shape_dir, obj_id)
    files = glob.glob(os.path.join(d, "*.obj")) + glob.glob(os.path.join(d, "*.ply"))
    if len(files) > 1:
        files = [f for f in files if "align" in os.path.basename(f)]
    mesh = trimesh.load(files[0], process=False, force="mesh", skip_materials=True)
    mesh.vertices = mesh.vertices - (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2  # same as oikit
    return (strip_support_plate(mesh, obj_id) if strip else mesh), name


def load_obj(shape_dir, obj_id, n_pts, mesh=None, name=None):
    if mesh is None:
        mesh, name = load_mesh(shape_dir, obj_id)
    pts, fid = trimesh.sample.sample_surface_even(mesh, n_pts, seed=0)
    if len(pts) < n_pts:
        extra, fid2 = trimesh.sample.sample_surface(mesh, n_pts - len(pts), seed=0)
        pts, fid = np.concatenate([pts, extra]), np.concatenate([fid, fid2])
    nrm = mesh.face_normals[fid]
    return (torch.tensor(pts, dtype=torch.float32)[None], torch.tensor(nrm, dtype=torch.float32)[None], name)


def stack(key, gs):
    return torch.from_numpy(np.stack([g[key] for g in gs]))


GROUP_COLORS = {"palm": "#7f7f7f", "thumb": "#d62728", "index": "#ff7f0e", "middle": "#2ca02c", "ring": "#1f77b4",
                "little": "#9467bd"}
