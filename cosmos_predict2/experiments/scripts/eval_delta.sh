# PRED_DIR="outputs_eval/action_conditioned/basic/cosmos_predict_v2p5/000150000/20/model"
# PRED_DIR="outputs_eval/action_conditioned/basic/OF2/000001500/20/model"
# PRED_DIR="outputs_eval/action_conditioned/basic/vjepa2/000001500/20/model"
# PRED_DIR="outputs_eval/action_conditioned/basic/cotracker_tau=0/000001500/20/model"
# PRED_DIR="outputs_eval/action_conditioned/basic/cotracker_4/000001500/20/model"
# PRED_DIR="outputs_eval/action_conditioned/basic/opd_fixed_teacher_kl_grad/000001000/20/model"
PRED_DIR="outputs_eval/action_conditioned/basic/mixed_reward_0.7of_0.3vjepa1/000000900/20/model"


# GT_DIR="outputs_eval/action_conditioned/basic/cosmos_predict_v2p5/000150000/20/model"
# GT_DIR="outputs_eval/action_conditioned/basic/OF2/000001500/20/model"
# GT_DIR="outputs_eval/action_conditioned/basic/vjepa2/000001500/20/model"
# GT_DIR="outputs_eval/action_conditioned/basic/cotracker_tau=0/000001500/20/model"
# GT_DIR="outputs_eval/action_conditioned/basic/cotracker_4/000001500/20/model"
GT_DIR="outputs_eval/action_conditioned/basic/opd_fixed_teacher_kl_grad/000001000/20/model"
# GT_DIR="assets/action_conditioned/basic/bridge1/gt_video_renamed"
# GT_DIR="outputs_eval/action_conditioned/basic/OF2/000001000/20/model"

python cosmos_predict2/_src/predict2/action/eval/delta_LIPIS.py \
  "$PRED_DIR" \
  "$GT_DIR" \
  -n 3 \
  --max-frames 36 \
  --net alex \
  --device cuda \
  --batch-size 256