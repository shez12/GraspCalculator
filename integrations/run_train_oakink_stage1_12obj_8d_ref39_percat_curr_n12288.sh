#!/usr/bin/env bash
# Stage 1 training launcher: 12 representative OakInk objects (1 per category),
# 8D PointNet (+ s_ax) + 39D cross-hand ref_feat + per-category score_threshold[cat] curriculum.
#
# Override paths via environment variables:
#   SIMTOOLREAL_DIR=/path/to/simtoolreal
#   GRASPCALC_DIR=/path/to/GraspCalculator
#   OAKINK_ASSETS_DIR=/path/to/simtoolreal_oakink_assets
#   PYTHON=python
#   CKPT=/path/to/model.pth   # optional: omit to train from scratch, set to warm-start weights
#   WANDB_ACTIVATE=0          # set to 1 to log to Weights & Biases
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRASPCALC_DIR="${GRASPCALC_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SIMTOOLREAL_DIR="${SIMTOOLREAL_DIR:-${PWD}}"
OAKINK_ASSETS_DIR="${OAKINK_ASSETS_DIR:-${HOME}/data/simtoolreal_oakink_assets}"
MANIFEST="${MANIFEST:-${OAKINK_ASSETS_DIR}/manifest_12cat_1each.json}"
REF_DIR="${REF_DIR:-${GRASPCALC_DIR}/refs_by_cat}"
PYTHON="${PYTHON:-python}"
NUM_ENVS="${NUM_ENVS:-12288}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-49152}"
EXPL_BLOCK_SIZE="${EXPL_BLOCK_SIZE:-2048}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-simtoolreal_grasp_score}"
WANDB_NAME="${WANDB_NAME:-oakink_stage1_12obj_8d_ref39_percat_curr_n${NUM_ENVS}}"
CKPT="${CKPT:-}"
CKPT_MODE="${CKPT_MODE:-weights}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${SIMTOOLREAL_DIR}:${GRASPCALC_DIR}:${PYTHONPATH:-}"
cd "${SIMTOOLREAL_DIR}"

EXTRA_ARGS=()
if [ -n "${CKPT}" ]; then
  EXTRA_ARGS+=(--checkpoint "${CKPT}" --checkpoint_load_mode "${CKPT_MODE}")
fi
if [ "${WANDB_ACTIVATE}" = "1" ]; then
  EXTRA_ARGS+=(--wandb_activate --wandb_project "${WANDB_PROJECT}" --wandb_name "${WANDB_NAME}")
fi

exec "${PYTHON}" isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_pointnet_sapg_cfg_entry_point \
  --headless \
  "${EXTRA_ARGS[@]}" \
  env.scene.num_envs="${NUM_ENVS}" \
  env.assets.dataset_manifest="${MANIFEST}" \
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
  env.termination.success_tolerance=0.075 \
  env.termination.grasp_score_threshold_start=0.0 \
  env.termination.grasp_score_threshold_step=0.05 \
  env.termination.grasp_score_threshold_max=0.40 \
  env.reward.grasp_score_backend=invariant_v2 \
  env.reward.grasp_score_ref_dir="${REF_DIR}" \
  env.reward.grasp_score_chunk=256 \
  env.reward.grasp_score_interval=50 \
  env.reward.grasp_score_rew_scale=100.0 \
  env.reward.grasp_score_pre_lift_mult=1.0 \
  agent.params.config.save_frequency=500 \
  agent.params.config.expl_coef_block_size="${EXPL_BLOCK_SIZE}" \
  agent.params.config.minibatch_size="${MINIBATCH_SIZE}" \
  agent.params.config.central_value_config.minibatch_size="${MINIBATCH_SIZE}"
