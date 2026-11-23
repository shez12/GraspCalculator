"""URDF hand configs (link -> semantic part, finger keypoint chains)."""
import os

from .hands import URDFHandConfig

SIMTOOLREAL_ROOT = os.environ.get("SIMTOOLREAL_ROOT", "/home/szc/work/newsimtoolreal/simtoolreal")


def sharpa_ha4(side="left", urdf_path=None, points_per_link=128) -> URDFHandConfig:
    """SharpaWave HA4 (22 DoF). Pinky metacarpal counts as palm; elastomer pads belong to the distal part."""
    s = side
    parts = {f"{s}_hand_C_MC": "palm", f"{s}_pinky_MC": "palm",
             f"{s}_thumb_MC": "thumb_0", f"{s}_thumb_PP": "thumb_1",
             f"{s}_thumb_DP": "thumb_2", f"{s}_thumb_elastomer": "thumb_2"}
    chains = {"thumb": [f"{s}_thumb_MC", f"{s}_thumb_PP", f"{s}_thumb_DP", f"{s}_thumb_fingertip"]}
    for f, name in [("index", "index"), ("middle", "middle"), ("ring", "ring"), ("little", "pinky")]:
        parts.update({f"{s}_{name}_MCP_VL": f"{f}_0", f"{s}_{name}_PP": f"{f}_0", f"{s}_{name}_MP": f"{f}_1",
                      f"{s}_{name}_DP": f"{f}_2", f"{s}_{name}_elastomer": f"{f}_2"})
        chains[f] = [f"{s}_{name}_PP", f"{s}_{name}_MP", f"{s}_{name}_DP", f"{s}_{name}_fingertip"]
    urdf_path = urdf_path or os.path.join(SIMTOOLREAL_ROOT, f"assets/urdf/{s}_sharpa_ha4/{s}_sharpa_ha4_v2_1.urdf")
    return URDFHandConfig(urdf_path=urdf_path, link_parts=parts, finger_chains=chains, wrist_link=f"{s}_hand_C_MC",
                          points_per_link=points_per_link, side=side)
