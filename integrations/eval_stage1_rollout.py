#!/usr/bin/env python3
"""Headless multi-object rollout evaluator for SimToolReal PointNet + 39D ref_feat policies.

Reports per-object lift rate, pos_near%, gated_near%, goals/trial, and live
`invariant_v2` scores (total, F1, F4p, F4d, F_loc) at gated success and pos_near frames.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to RL-Games .pth checkpoint")
    parser.add_argument("--manifest", required=True, help="Path to OakInk dataset manifest JSON")
    parser.add_argument(
        "--ref_dir",
        default=str(Path(__file__).resolve().parent.parent / "refs_by_cat"),
        help="Directory containing per-category mirrored left-hand references",
    )
    parser.add_argument("--task", default="Isaacsimenvs-SimToolReal-Direct-v0")
    parser.add_argument("--agent", default="rl_games_pointnet_sapg_cfg_entry_point")
    parser.add_argument("--trials_per_obj", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=0.20, help="Grasp-score gating threshold")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (default: stochastic)")
    parser.add_argument("--dump_best_frames", default="", help="Optional .pt path to save highest-v2 frame per object")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    la, _ = lp.parse_known_args([])
    la.headless = True
    la.enable_cameras = False
    app = AppLauncher(la).app

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from isaacsimenvs.tasks.simtoolreal.utils.grasp_score_utils import _score_single_group, _quat_to_mat
    from rl_games.torch_runner import Runner

    n_objs = len(json.loads(Path(args.manifest).read_text())["assets"])
    n_envs = args.trials_per_obj * n_objs

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = n_envs
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx"):
        px = env_cfg.sim.physx
        for attr, val in (
            ("gpu_max_rigid_contact_count", 2**19),
            ("gpu_max_rigid_patch_count", 2**17),
            ("gpu_found_lost_pairs_capacity", 2**18),
            ("gpu_found_lost_aggregate_pairs_capacity", 2**18),
            ("gpu_total_aggregate_pairs_capacity", 2**18),
            ("gpu_heap_capacity", 2**24),
            ("gpu_temp_buffer_capacity", 2**23),
            ("gpu_collision_stack_size", 2**24),
        ):
            if hasattr(px, attr):
                setattr(px, attr, val)

    env_cfg.assets.dataset_manifest = args.manifest
    env_cfg.reset.human_like_goals = True
    env_cfg.reset.human_like_scale_range = (1.0, 1.0)
    env_cfg.obs.use_pointcloud_obs = True
    env_cfg.obs.use_grasp_ref_feat = True
    env_cfg.reward.grasp_score_backend = "invariant_v2"
    env_cfg.reward.grasp_score_ref_dir = args.ref_dir
    env_cfg.reward.grasp_score_interval = 50
    env_cfg.reward.grasp_score_rew_scale = 100.0
    env_cfg.termination.use_grasp_score_curriculum = True
    env_cfg.termination.success_tolerance = 0.075
    env_cfg.termination.grasp_score_threshold_start = float(args.threshold)

    dr = env_cfg.domain_randomization
    dr.use_object_state_delay_noise = False
    dr.object_state_xyz_noise_std = 0.0
    dr.object_state_rotation_noise_degrees = 0.0
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0, 0.0)
    dr.torque_prob_range = (0.0, 0.0)

    spec = gym.spec(args.task)
    mod_name, cls_name = spec.entry_point.split(":")
    env = getattr(importlib.import_module(mod_name), cls_name)(cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(args.task, args.agent)
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device="cuda:0", clip_obs=clip_obs, clip_actions=clip_actions)
    agent_cfg["params"]["config"]["device"] = "cuda:0"
    agent_cfg["params"]["config"]["device_name"] = "cuda:0"
    if "expl_coef_block_size" in agent_cfg["params"]["config"]:
        agent_cfg["params"]["config"]["expl_coef_block_size"] = n_envs

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args.checkpoint)
    player.has_batch_dimension = True
    player.batch_size = n_envs
    player.reset()

    obs = player.env_reset(wrapped)
    env._current_grasp_score_threshold = float(args.threshold)
    if getattr(env, "_current_grasp_score_threshold_per_cat", None) is not None:
        env._current_grasp_score_threshold_per_cat.fill_(float(args.threshold))

    aidx = env._object_asset_index_per_env.cpu().numpy()
    assets = env._asset_manifest_assets

    lifted_ever = np.zeros(n_envs, dtype=bool)
    pos_near_cnt = np.zeros(n_envs, dtype=int)
    gated_near_cnt = np.zeros(n_envs, dtype=int)
    goals_hit = np.zeros(n_envs, dtype=int)

    near_by_obj = [[] for _ in range(n_objs)]
    succ_by_obj = [[] for _ in range(n_objs)]
    best_frame: dict[int, dict] = {}

    def eval_subset(env_ids_t: torch.Tensor):
        m = env_ids_t.numel()
        res = {k: torch.zeros(m, device=env.device) for k in ("total", "f1", "f4p", "f4d", "f_loc")}
        cids = env._grasp_env_ref_ids[env_ids_t]
        for cid in torch.unique(cids).tolist():
            mask = cids == cid
            r = _score_single_group(env, env._grasp_scorers_by_cat[int(cid)], env_ids_t[mask])
            for k in res:
                if k in r:
                    res[k][mask] = r[k].float()
        return {k: v.cpu().numpy() for k, v in res.items()}

    def capture_frame(ei: int, oi: int, scores_tup: tuple, is_succ: bool, is_pnear: bool, step_i: int):
        sub_ids = torch.tensor([ei], device=env.device, dtype=torch.long)
        pos = env.robot.data.body_link_pos_w[sub_ids]
        rot = _quat_to_mat(env.robot.data.body_link_quat_w[sub_ids])
        link_T = {}
        for idx_b, name in enumerate(env._grasp_body_names):
            T = torch.eye(4, device=env.device).unsqueeze(0)
            T[:, :3, :3], T[:, :3, 3] = rot[:, idx_b], pos[:, idx_b]
            link_T[name] = T
        hand = env._grasp_hand.from_link_poses(link_T)
        Ro = _quat_to_mat(env.object.data.root_quat_w[sub_ids])[0]
        po = env.object.data.root_pos_w[sub_ids][0]
        obj_w = (env._grasp_obj_pts[oi] @ Ro.T) + po[None]
        best_frame[oi] = {
            "category": assets[oi]["category"],
            "obj_id": assets[oi]["obj_id"],
            "urdf_path": assets[oi]["urdf_path"],
            "hand_pts": hand.points[0].detach().cpu(),
            "hand_pids": (hand.part_ids[0] if hand.part_ids.ndim == 2 else hand.part_ids).detach().cpu(),
            "obj_pts_w": obj_w.detach().cpu(),
            "obj_pts_local": env._grasp_obj_pts[oi].detach().cpu(),
            "obj_aff": env._grasp_obj_aff[oi].detach().cpu(),
            "Ro": Ro.detach().cpu(),
            "po": po.detach().cpu(),
            "v2": scores_tup[0],
            "f1": scores_tup[1],
            "f4p": scores_tup[2],
            "f4d": scores_tup[3],
            "f_loc": scores_tup[4],
            "is_success": bool(is_succ),
            "is_pos_near": bool(is_pnear),
            "step": int(step_i),
        }

    for step_i in range(args.steps):
        action = player.get_action(obs, is_deterministic=args.deterministic)
        obs, rew, dones, infos = player.env_step(wrapped, action)

        if getattr(player, "is_rnn", False) and getattr(player, "states", None) is not None:
            done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                for s in player.states:
                    s[:, done_idx, :] = 0.0

        lifted = env._lifted_object.cpu().numpy()
        pnear = env._pos_near_goal.cpu().numpy()
        gnear = env._near_goal.cpu().numpy()
        succ = env._is_success.cpu().numpy()

        lifted_ever |= lifted
        pos_near_cnt += pnear.astype(int)
        gated_near_cnt += gnear.astype(int)
        goals_hit += succ.astype(int)

        score_mask = pnear | succ | (lifted & (step_i % 15 == 0))
        if score_mask.any():
            eids = np.nonzero(score_mask)[0]
            sc = eval_subset(torch.as_tensor(eids, device=env.device, dtype=torch.long))
            for k_i, ei in enumerate(eids):
                oi = int(aidx[ei])
                tup = (
                    float(sc["total"][k_i]),
                    float(sc["f1"][k_i]),
                    float(sc["f4p"][k_i]),
                    float(sc["f4d"][k_i]),
                    float(sc["f_loc"][k_i]),
                )
                if pnear[ei]:
                    near_by_obj[oi].append(tup)
                if succ[ei]:
                    succ_by_obj[oi].append(tup)
                if args.dump_best_frames and (pnear[ei] or succ[ei]):
                    prev = best_frame.get(oi)
                    if prev is None or tup[0] > prev["v2"]:
                        capture_frame(int(ei), oi, tup, bool(succ[ei]), bool(pnear[ei]), step_i)

    mode_str = "deterministic" if args.deterministic else "stochastic"
    print(f"\n=== Rollout Summary ({mode_str}, {n_objs} objs x {args.trials_per_obj} trials, {args.steps} steps, thr={args.threshold:.2f}) ===")
    print(f"{'Category':16s} {'ObjID':8s} {'Lift':>6s} {'PosNear%':>9s} {'Gate%':>7s} {'Goals':>6s} {'Succ v2 (mean/max)':>20s} {'F1/F4p/F4d/Floc':>22s}")
    for oi, a in enumerate(assets):
        m = aidx == oi
        lift_n = int(lifted_ever[m].sum())
        pn_pct = 100.0 * pos_near_cnt[m].mean() / args.steps
        gn_pct = 100.0 * gated_near_cnt[m].mean() / args.steps
        g_mean = float(goals_hit[m].mean())
        rows = succ_by_obj[oi] if succ_by_obj[oi] else near_by_obj[oi]
        if rows:
            arr = np.array(rows)
            tag = "" if succ_by_obj[oi] else "*"
            v2_str = f"{arr[:,0].mean():.3f}/{arr[:,0].max():.3f}{tag}"
            comp_str = f"{arr[:,1].mean():.2f}/{arr[:,2].mean():.2f}/{arr[:,3].mean():.2f}/{arr[:,4].mean():.2f}"
        else:
            v2_str, comp_str = "-", "-"
        print(f"{a['category']:16s} {a['obj_id']:8s} {lift_n:2d}/{int(m.sum()):<3d} {pn_pct:8.1f}% {gn_pct:6.1f}% {g_mean:6.2f} {v2_str:>20s} {comp_str:>22s}")

    if args.dump_best_frames:
        out_p = Path(args.dump_best_frames)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_frame, out_p)
        print(f"\nSaved {len(best_frame)} best frames -> {out_p}")

    app.close()


if __name__ == "__main__":
    main()
