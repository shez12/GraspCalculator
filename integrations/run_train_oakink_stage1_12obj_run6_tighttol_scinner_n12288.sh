#!/usr/bin/env bash
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="/home/szc/work/newsimtoolreal/simtoolreal:/home/szc/work/graspcalculator:${PYTHONPATH:-}"
cd /home/szc/work/newsimtoolreal/simtoolreal

CKPT="/home/szc/work/newsimtoolreal/simtoolreal/outputs/2026-11-26/16-54-00/0_simtoolreal_pointnet_sapg/last/model.pth"
THR_JSON="/home/szc/.cache/stage1_run6_thresholds_snapshot.json"

exec /home/szc/.conda/envs/simtoolreal_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_pointnet_sapg_cfg_entry_point \
  --headless \
  --checkpoint "${CKPT}" \
  --checkpoint_load_mode weights \
  --wandb_activate \
  --clearml_activate \
  --clearml_project simtoolreal_grasp_score \
  --clearml_name oakink_stage1_run6_tighttol_consec_scinner_n12288 \
  --wandb_project simtoolreal_grasp_score \
  --wandb_name oakink_stage1_run6_tighttol_consec_scinner_n12288 \
  env.scene.num_envs=12288 \
  env.assets.dataset_manifest=/home/szc/data/simtoolreal_oakink_assets/manifest_12cat_1each.json \
  env.reset.human_like_goals=True \
  env.reset.human_like_scale_range=[1.0,1.0] \
  env.obs.use_pointcloud_obs=True \
  env.obs.use_grasp_ref_feat=True \
  env.domain_randomization.use_object_state_delay_noise=False \
  env.domain_randomization.object_state_xyz_noise_std=0.0 \
  env.domain_randomization.object_state_rotation_noise_degrees=0.0 \
  env.domain_randomization.object_scale_noise_multiplier_range=[1.0,1.0] \
  env.domain_randomization.joint_velocity_obs_noise_std=0.0 \
  env.domain_randomization.force_scale=0.0 \
  env.domain_randomization.torque_scale=0.0 \
  env.domain_randomization.force_prob_range=[0.0,0.0] \
  env.domain_randomization.torque_prob_range=[0.0,0.0] \
  env.termination.use_grasp_score_curriculum=True \
  env.termination.success_tolerance=0.06 \
  env.termination.grasp_score_threshold_start=0.0 \
  env.termination.grasp_score_threshold_step=0.05 \
  env.termination.grasp_score_threshold_max=0.60 \
  env.termination.grasp_score_threshold_init_by_cat="${THR_JSON}" \
  env.reward.grasp_score_backend=invariant_v2 \
  env.reward.grasp_score_ref_dir=/home/szc/work/graspcalculator/refs_by_cat \
  env.reward.grasp_score_chunk=256 \
  env.reward.grasp_score_interval=50 \
  env.reward.grasp_score_rew_scale=100.0 \
  env.reward.grasp_score_pre_lift_mult=1.0 \
  agent.params.config.per_cat_advantage_norm=True \
  agent.params.config.per_cat_advantage_floor_ratio=0.2 \
  agent.params.config.per_cat_advantage_clip=8.0 \
  agent.params.config.save_frequency=500 \
  agent.params.config.expl_coef_block_size=2048 \
  agent.params.config.minibatch_size=49152 \
  agent.params.config.central_value_config.minibatch_size=49152
