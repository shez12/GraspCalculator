#!/usr/bin/env bash
# Periodic rollout video recorder aligned with training (stochastic policy,
# 8D PointNet + 39D ref_feat, grasp-score gate, LSTM reset on dones).
set -u

RUN_DIR="${1:?pass run dir as first arg}"
INTERVAL_SEC="${2:-900}"
FULL_MANIFEST="${3:?pass dataset manifest json as third arg}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRASPCALC_DIR="${GRASPCALC_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SIMTOOLREAL_DIR="${SIMTOOLREAL_DIR:-${PWD}}"
REF_DIR="${REF_DIR:-${GRASPCALC_DIR}/refs_by_cat}"
PYTHON="${PYTHON:-python}"
GRASP_SCORE_THRESHOLD="${GRASP_SCORE_THRESHOLD:-0.20}"

CKPT_DIR="${RUN_DIR}/0_simtoolreal_pointnet_sapg"
VID_DIR="${RUN_DIR}/videos"
TMP_MANIFEST="/tmp/rollout_asset_manifest_$$.json"
TMP_CLOUDS="/tmp/rollout_asset_clouds_$$.pt"
mkdir -p "${VID_DIR}"

LAST_MTIME=0
echo "[watcher] watching ${CKPT_DIR} every ${INTERVAL_SEC}s -> ${VID_DIR} (manifest=${FULL_MANIFEST})"

while true; do
  CKPT=""
  if [ -f "${CKPT_DIR}/last/model.pth" ]; then
    CKPT="${CKPT_DIR}/last/model.pth"
  else
    CKPT=$(ls -t "${CKPT_DIR}"/nn/*.pth 2>/dev/null | head -n 1 || true)
  fi

  if [ -n "${CKPT}" ] && [ -f "${CKPT}" ]; then
    MTIME=$(stat -c %Y "${CKPT}" 2>/dev/null || echo 0)
    if [ "${MTIME}" -gt "${LAST_MTIME}" ]; then
      ASSET_TAG=$("${PYTHON}" "${SCRIPT_DIR}/pick_random_oakink_asset.py" "${FULL_MANIFEST}" "${TMP_MANIFEST}" "${TMP_CLOUDS}")
      TS=$(date +%Y%m%d_%H%M%S)
      OUT="${VID_DIR}/rollout_${TS}_${ASSET_TAG}.mp4"
      echo "[watcher] $(date '+%F %T') asset=${ASSET_TAG} recording ${CKPT} -> ${OUT}"
      cd "${SIMTOOLREAL_DIR}"
      OMNI_KIT_ACCEPT_EULA=YES \
      PYTHONPATH="${SIMTOOLREAL_DIR}:${GRASPCALC_DIR}:${PYTHONPATH:-}" \
      "${PYTHON}" -u isaacsimenvs/play_video.py \
        --checkpoint "${CKPT}" \
        --task Isaacsimenvs-SimToolReal-Direct-v0 \
        --agent rl_games_pointnet_sapg_cfg_entry_point \
        --num_envs 1 \
        --steps 300 \
        --dataset_manifest "${TMP_MANIFEST}" \
        --human_like_goals \
        --use_pointcloud_obs \
        --use_grasp_ref_feat \
        --grasp_score_ref_dir "${REF_DIR}" \
        --grasp_score_backend invariant_v2 \
        --use_grasp_score_curriculum \
        --grasp_score_threshold "${GRASP_SCORE_THRESHOLD}" \
        --clean_env \
        --out "${OUT}"
      if [ -f "${OUT}" ]; then
        LAST_MTIME="${MTIME}"
        echo "[watcher] saved ${OUT} ($(du -h "${OUT}" | cut -f1))"
      fi
    fi
  fi
  sleep "${INTERVAL_SEC}"
done
