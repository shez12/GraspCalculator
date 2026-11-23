cd /home/szc/work/newsimtoolreal/simtoolreal
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/szc/work/newsimtoolreal/simtoolreal:$PYTHONPATH
/home/szc/.conda/envs/simtoolreal_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 --agent rl_games_sapg_cfg_entry_point --headless \
  --checkpoint pretrained_policy/model.pth --checkpoint_load_mode weights \
  --wandb_activate --wandb_project simtoolreal_grasp_score --wandb_name sapg_sharpa_ref_w300_n6144 \
  env.scene.num_envs=6144 \
  env.reward.grasp_score_rew_scale=300.0 \
  env.reward.grasp_score_ref=/home/szc/work/graspcalculator/refs/sharpa_mallet_hammer_swing_side_lift.pt \
  agent.params.config.expl_coef_block_size=1024
