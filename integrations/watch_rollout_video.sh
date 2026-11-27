#!/usr/bin/env bash
set -u
RUN_DIR="${1:?pass run dir as first arg}"
INTERVAL_SEC="${2:-900}"  # default: every 15 minutes
FULL_MANIFEST="${3:-/home/szc/data/simtoolreal_oakink_assets/manifest_12cat_1each.json}"
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
      ASSET_TAG=$(/home/szc/.conda/envs/simtoolreal_isaacsim/bin/python /home/szc/.cache/pick_random_oakink_asset.py "${FULL_MANIFEST}" "${TMP_MANIFEST}" "${TMP_CLOUDS}")
      TS=$(date +%Y%m%d_%H%M%S)
      OUT="${VID_DIR}/rollout_${TS}_${ASSET_TAG}.mp4"
      echo "[watcher] $(date '+%F %T') asset=${ASSET_TAG} recording ${CKPT} -> ${OUT}"
      cd /home/szc/work/newsimtoolreal/simtoolreal
      OMNI_KIT_ACCEPT_EULA=YES \
      PYTHONPATH="/home/szc/work/newsimtoolreal/simtoolreal:/home/szc/work/graspcalculator:${PYTHONPATH:-}" \
      /home/szc/.conda/envs/simtoolreal_isaacsim/bin/python -u isaacsimenvs/play_video.py \
        --checkpoint "${CKPT}" \
        --task Isaacsimenvs-SimToolReal-Direct-v0 \
        --agent rl_games_pointnet_sapg_cfg_entry_point \
        --num_envs 1 \
        --steps 300 \
        --dataset_manifest "${TMP_MANIFEST}" \
        --human_like_goals \
        --use_pointcloud_obs \
        --use_grasp_ref_feat \
        --grasp_score_ref_dir /home/szc/work/graspcalculator/refs_by_cat \
        --grasp_score_backend invariant_v2 \
        --use_grasp_score_curriculum \
        --grasp_score_threshold 0.20 \
        --deterministic \
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
