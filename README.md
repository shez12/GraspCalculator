# graspcalculator

Hand-agnostic dexterous grasp scoring against a functional reference grasp.

A candidate grasp scores higher the closer it is to a reference grasp. References are OakInk MANO functional grasps.
MANO plugs in through the same hand interface as any URDF dexterous hand: the scoring code never touches MANO vertices directly.

This version:
- one reference grasp scores every object instance of a category;
- contact is compared on OakBase functional parts, optionally only on an affordance part such as `handle`;
- `total = F1 × gate(F4p)`.

---

## 1. Layout

```
graspcalc/
  hands.py        HandState, MANOHand, URDFHand (+ URDFHandConfig)
  factors.py      F1 hand shape, aligned-distance contact maps, soft IoU
  oakink.py       OakInk-Shape loaders: real / transferred grasps, object meshes, support-plate removal
  paths.py        default data paths (env vars OAKINK_SHAPE_DIR, OAKBASE_DIR, MANO_ROOT)
  score_parts.py  main scorer: one reference -> whole category, OakBase part-aware contact
  summarize.py    tables / box plots grouped by contacted parts and by intent
  report.py       PDF report of one or more runs
  bench.py        timing
```

## 2. Environment

GPU environment used for development: Python 3.10, PyTorch 2.12.1+cu130 (RTX 6000D / sm_120).

```bash
pip install -r requirements.txt
```

Notes:
- `chumpy` uses NumPy aliases removed in recent NumPy releases (`np.int` and others). `hands.py` adds them back.
- On the development machine, the `py310` conda environment borrows `manotorch`, `mano` and `chumpy` from the `oakink` environment through symlinks:

```bash
export PYTHONPATH=/home/szc/.cache/grasp_score_pypath
PY=/home/szc/.conda/envs/py310/bin/python
```

## 3. Data

| Data | Default path (environment variable) | Notes |
|---|---|---|
| OakInk-Shape | `/home/szc/data/OakInk/shape` (`OAKINK_SHAPE_DIR`) | `oakink_shape_v2/**/hand_param.pkl` (real and transferred grasps), `OakInkObjectsV2`, `OakInkVirtualObjectsV2`, `metaV2` |
| OakBase | `/home/szc/data/OakInk/OakBase` (`OAKBASE_DIR`) | part segmentation: `<category>/<object>/part_*.ply` plus `.json` |
| MANO | `/home/szc/work/OakInk/assets/mano_v1_2` (`MANO_ROOT`) | |

Download and check OakBase:

```bash
wget -c https://huggingface.co/datasets/oakink/OakInk-v1/resolve/main/OakBase.zip
# sha256 58a0ea15a39e3de036d2c7abbcaa81f4b91e840867258b7e70bdf91c2013f756
unzip OakBase.zip -d /home/szc/data/OakInk/
```

OakBase coverage:
- 29 categories, covering both real and virtual objects, matched by object name.
- Part clouds are in the same frame as the bbox-centred object meshes (nearest-neighbour error about 1 mm).
- teapot: all 22 objects have `body`, `spout`, `handle` and `lid`.
- mug: `body` and `handle`. S10002 and S10012 are misaligned and skipped automatically.

## 4. Method

### 4.1 Hand interface (`hands.py`)

`HandState` has three parts:
- hand surface points, each labelled with one of 16 semantic parts: `palm` plus `thumb/index/middle/ring/little` × 3 segments (`_0` proximal, `_2` distal);
- the wrist position;
- a keypoint chain for each finger, from base to fingertip.

Implementations:
- `MANOHand`: assigns vertices to parts by MANO skinning weights.
- `URDFHand` with `URDFHandConfig`: forward kinematics with pytorch_kinematics and points sampled on each link surface. The config sets `link_parts` (link to part), `finger_chains`, `wrist_link` and `tip_offsets`.

**`URDFHand` has not been validated on a real robot hand yet.**

### 4.2 F1 hand shape (`factors.f1_shape`)

F1 compares five kinds of features in a palm frame. The palm frame has its origin at the wrist, z along the palm normal and y towards the middle finger.

| Feature | Tolerance σ |
|---|---|
| fingertip positions (in palm lengths) | 0.15 |
| flexion angles and spread between adjacent fingers | 0.26 rad |
| straightness (\|tip − base\| / chain length) | 0.10 |
| thumb-tip to finger-tip distances (in palm lengths) | 0.15 |

`F1 = exp(-RMS(diff/σ) / T)`. The default T is 2.0; `--f1_loosen 1.5` makes it 3.0.

F1 depends only on hand shape, not on the hand's global pose or the object, so it can be compared across objects.

### 4.3 Objects (`oakink.load_mesh`, `score_parts.load_obj_oakbase`)

1. Load the object's `align` mesh (`*.obj` / `*.ply` only) and centre it on its bbox.
2. **Remove a support plate** (`strip_support_plate`). A planar patch is treated as a plate only if all of these hold:
   - it covers more than 15% of the surface area;
   - the rest of the object lies on one side of it;
   - the object's footprint shrinks above the plane;
   - the rest of the object is narrower than the patch. This last check keeps a mug's own flat bottom.

   Currently only teapot_2 (o13102) triggers this.
3. **Crop to OakBase**: drop faces more than 5 mm from the OakBase part clouds. If more than 20% of the surface is dropped, the mesh and OakBase are treated as misaligned and the object is skipped.
4. Sample 2048 surface points. Each point takes the OakBase part label of its nearest part point.

### 4.4 F4p part contact (`score_parts.py`)

1. **Aligned distance** (GenDexGrasp): `d_align = |h−o| · exp(2(1−|cos θ|))`, where θ is the angle between (h−o) and the object normal. It suppresses grazing side contacts.
2. **Soft contact**: `og[k, o] = exp(-(min_{h∈part k} d_align / 2cm)²)`, a 16 × M matrix (M object points).
3. **Per-part shares**: `A[k, p] = Σ_{o∈part p} og[k, o]`, normalised as `P = A / ΣA`. This 16 × P matrix does not depend on the instance, so it can be compared across objects.
4. **Affordance only** (`--afford_parts handle`): take `P[:, :, handle]`, renormalise it within the handle, and compare it with the reference by soft IoU, `Σmin / Σmax`. This measures which hand parts touch the handle and in what proportion. Contact on other parts is ignored.
   - **Share discount** (`--share_discount 1`, default on): the total is also multiplied by `discount = min(candidate share on the affordance / reference share, 1)`. Without it, a faint contact tail on the affordance (e.g. 0.04% of the contact, hand 36 mm from a trigger) is renormalised into a full distribution and can match the reference.
   - **F4d distance profile** (`--contact_term dist`, after DCGrasp): replaces the share discount. For each of the 16 hand parts, `D[k]` = distance from the part to the nearest affordance point (nearest object point without `--afford_parts`), clipped at 5 cm. With `w[k] = max(exp(-δ Dr[k]²), exp(-δ Dc[k]²))`, δ = 5000 m⁻², `F4d = exp(-Σ w (Dc − Dr)² / Σ w / σ_d²)`, σ_d = 1 cm. Absolute distances are kept, so a finger 36 mm from the trigger is not mistaken for a 4 mm contact; a transferred copy keeps its distances, so it is not penalised for a lower share.
5. **Without `--afford_parts`**:
   - `F4p = sqrt(F4p_a · F4p_b)`, where `F4p_a` is the soft IoU of the per-part shares and `F4p_b` the soft IoU of the full 16 × P matrix;
   - `--min_part_ratio` turns on a hard requirement (default 0, off): every part holding at least `--ref_part_min` of the reference contact must hold at least `ratio` × the reference share in the candidate, otherwise the total is 0.

### 4.5 Total score

```
gate  = sigmoid((F4p - gate_tau) / gate_width)     # defaults 0.4 / 0.05
total = F1 * gate * discount   # --contact_term share: discount = share discount (1 unless --afford_parts and --share_discount 1)
                               # --contact_term dist:  discount = F4d
```

- F4p decides whether the grasp is in the right place; F1 then ranks hand shape.
- `--gate_tau 0` restores the older total, `sqrt(F1·F4p)`.
- No object symmetry or mirroring is used; left and right hands stay distinct.

### 4.6 Reference

One reference per category scores every grasp on every object of that category. Intent is not used for scoring.
- `--ref <obj_id>/<grasp_id>` sets the reference. Without it, the first `use` grasp on a real object is used.
- References used so far:
  - teapot: `C13001/01267e054a` (handle 64%, body 36%);
  - mug: `C10001/7a99e06f35` (handle 41%, body 59%).

## 5. Usage

```bash
# 1) score: part_scores.json, retrieval.json, meta.json and one viz_<obj_id>.png per object;
#    the log contains full rankings and the retrieval check
$PY -m graspcalc.score_parts --cat teapot --ref C13001/01267e054a --afford_parts handle \
    --out_dir out_parts_teapot_handle > out_teapot.log
$PY -m graspcalc.score_parts --cat mug --afford_parts handle --out_dir out_parts_mug_handle > out_mug.log

# 2) statistics (tables + part_summary.png / intent_summary.png); --compare <other part_scores.json> adds Spearman
$PY -m graspcalc.summarize --scores out_parts_teapot_handle/part_scores.json --title "teapot, handle only"

# 3) PDF report (method, tables, box plots, all per-object figures)
$PY -m graspcalc.report --runs out_parts_teapot_handle out_parts_mug_handle --out grasp_score_report.pdf

# 4) timing
$PY -m graspcalc.bench --device cuda
$PY -m graspcalc.bench --device cpu --batches 1 16 64 256
```

Run time on the development machine (GPU):

| Category | Run time | Where the time goes |
|---|---|---|
| teapot | about 20 s | |
| mug (28,720 grasps) | about 14 min | mostly the retrieval check, which re-scores the whole category for each of 468 real references |

`score_parts` options:

| Option | Default | Description |
|---|---|---|
| `--cat` | teapot | category name, as used in OakInk-Shape and OakBase |
| `--ref` | auto | `<obj_id>/<grasp_id>` |
| `--afford_parts` | empty | comma-separated parts to score on, e.g. `handle` |
| `--f1_loosen` | 1.5 | multiplier on the F1 tolerance |
| `--gate_tau` / `--gate_width` | 0.4 / 0.05 | gate centre and width; `gate_tau ≤ 0` uses `sqrt(F1·F4p)` |
| `--share_discount` | 1 | affordance mode: multiply by `min(candidate / reference affordance share, 1)`; 0 disables |
| `--contact_term` | share | `share`: share discount; `dist`: multiply by F4d instead |
| `--dist_delta` / `--dist_sigma` / `--dist_clip` | 5000 / 0.01 / 0.05 | F4d weight (m⁻²), tolerance (m), distance clip (m) |
| `--penetration` | 0 | experimental, not evaluated: multiply by `exp(-(relu(pen − pen_tau)/pen_sigma)²)`, pen = deepest object point inside the hand (hand normals) |
| `--ref_part_min` / `--min_part_ratio` | 0.10 / 0 | part-share hard requirement; off when ratio is 0 |
| `--n_obj_pts` | 2048 | object sample points |
| `--device` | cuda if available | cuda or cpu |
| `--out_dir` | `out_parts_<cat>` | output directory |

## 6. Outputs

| File | Content |
|---|---|
| `part_scores.json` | one record per grasp: `total`, `s1_shape` (F1), `s4p`, `s4p_a`, `s4p_b`, `gate`, `discount`, `f4d`, `pen_mm`, `s_pen`, `part_share`, `intent`, `is_ref` |
| `retrieval.json` | per real reference: `top1`, `mean_rank`, `mean_n`, `ref_part_share` |
| `meta.json` | run configuration, part vocabulary, skipped objects |
| `viz_<obj_id>.png` | panel 1 is the reference (on its own object), then the best, median and worst grasp on this object. Object coloured by part, red = contact, hand coloured by part |

## 7. Results (handle only, no hard requirement)

| Category | Objects / grasps | Retrieval top-1 when the reference touches the handle | Mean rank |
|---|---|---|---|
| teapot | 22 / 168 | 87.5% (all references: 90.6%) | 1.15 |
| mug | 165 / 28,719 | 79.4% (all references: 72.2%) | 2.70 (about 202 candidates per object) |

Retrieval check: a real grasp is used as the reference, and its transferred copy on each virtual instance should rank first.

Speed (reference contact cached, MANO forward included):

| Device | Speed |
|---|---|
| CPU, single grasp | about 4–7 ms |
| CPU, batched | about 7–8 ms per grasp |
| GPU, batch 1024 | about 0.35 ms per grasp (about 2,850 grasps/s) |

## 8. Known issues

- **The share discount favours grasps with more affordance contact.** A transferred copy whose affordance share dropped below the reference's is discounted, so cross-object retrieval falls for references with little affordance contact (e.g. trigger_sprayer, whose reference has 12% on the trigger). `--share_discount 0` turns it off; `--contact_term dist` (F4d) avoids it.
- **The reference must touch the affordance part.** If it does not, affordance-only mode falls back to ranking by F1 alone.
- **Mug F4p sits near the gate centre.** It is around 0.4, so the gate is only 0.45–0.58. `--gate_tau 0.3` loosens it.
- **`URDFHand` is untested on real robot hands.**

## SimToolReal (Isaac Lab, left Sharpa HA4)

1. Export a left-hand reference (the right-hand OakInk grasp and its object are mirrored through x = 0):
   `python -m graspcalc.export_reference --cat hammer --ref C22001/109708ca1a --side left --out refs/hammer_C22001_109708ca1a_left.pt`
2. Copy `integrations/simtoolreal_eval_grasp_score_isaacsim.py` to `<simtoolreal>/dextoolbench/eval_grasp_score_isaacsim.py` and run it from the simtoolreal root:
   `OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD python dextoolbench/eval_grasp_score_isaacsim.py --object_category hammer --object_name claw_hammer --task_name swing_down --grasp_ref <ref.pt> --afford_part grip`
   It runs the DexToolBench policy evaluation and scores every step of the first `--score_window` steps after the object is lifted (`total = F1 x gate(F4p) x F4d`).
   Hand points come from the simulated link poses (`URDFHand.from_link_poses`); the robot URDF is used so links merged by `merge_fixed_joints` are recovered from fixed joints.
   DexToolBench meshes have no part labels; `graspcalc.objects.axis_part_labels` splits them into handle / head along the principal axis.
   Output: `outputs/grasp_score/<object>/<task>/grasp_scores.json` and `ep*_grasp.npz`.

`graspcalc.robots.sharpa_ha4` holds the Sharpa link -> part map; `graspcalc.retarget.retarget` fits a URDF hand to a MANO grasp by keypoints (offline checks).
