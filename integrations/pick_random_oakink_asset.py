"""Pick a random OakInk asset from manifest.json and write a 1-asset manifest for fast rollout video."""
import json
import random
import sys
import torch

MANIFEST_IN = sys.argv[1]
MANIFEST_OUT = sys.argv[2]
CLOUDS_OUT = sys.argv[3]

m = json.load(open(MANIFEST_IN))
assets = m["assets"]
idx = random.randrange(len(assets))
a = assets[idx]

clouds = torch.load(m["clouds_pt"], map_location="cpu", weights_only=False)
single_clouds = {}
for k, v in clouds.items():
    if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == len(assets):
        single_clouds[k] = v[idx : idx + 1].clone()
    else:
        single_clouds[k] = v
torch.save(single_clouds, CLOUDS_OUT)

single_m = dict(m)
single_m["assets"] = [a]
single_m["clouds_pt"] = CLOUDS_OUT
json.dump(single_m, open(MANIFEST_OUT, "w"), indent=2)

cat = a.get("category", "obj")
oid = a.get("obj_id", str(idx))
print(f"{cat}_{oid}")
