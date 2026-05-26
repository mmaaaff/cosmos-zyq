# CHECKPOINT_DIR="output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_bridge_action_conditioned/checkpoints/iter_000150000"
# CHECKPOINT_DIR="output/cosmos_predict2_action_conditioned_grpo/OF2/2b_bridge_action_conditioned_grpo_optical_flow/checkpoints/iter_000001500"
# CHECKPOINT_DIR="output/cosmos_predict2_action_conditioned_grpo/vjepa2/2b_bridge_action_conditioned_grpo_vjepa/checkpoints/iter_000001500"
# CHECKPOINT_DIR="output/cosmos_predict2_action_conditioned_grpo/cotracker_4/2b_bridge_action_conditioned_grpo_cotracker/checkpoints/iter_000001500"
CHECKPOINT_DIR="output/cosmos_predict2_action_conditioned_grpo/cotracker_tau=0/2b_bridge_action_conditioned_grpo_cotracker/checkpoints/iter_000001500"

# experiment="ac_reason_embeddings_rectified_flow_2b_256_320"
# experiment="ac_reason_embeddings_rectified_flow_2b_256_320_grpo_optical_flow"
# experiment="ac_reason_embeddings_rectified_flow_2b_256_320_grpo_vjepa"
# experiment="ac_reason_embeddings_rectified_flow_2b_256_320_grpo_cotracker"
experiment="ac_reason_embeddings_rectified_flow_2b_256_320_grpo_cotracker"

MODEL="model.pt" # model.pt | model_ema_bf16.pt | model_ema_fp32.pt
num_steps=20
n=3
GPU_ID=3
save_dir="outputs_eval"

if [[ "$CHECKPOINT_DIR" =~ cosmos_predict2_action_conditioned(_grpo)?/([^/]+)/.*/checkpoints/iter_([0-9]+)$ ]]; then
    group="${BASH_REMATCH[2]}"
    iter="${BASH_REMATCH[3]}"

    MODEL_TAG="${MODEL%.pt}"
    SAVE_ROOT="./${save_dir}/action_conditioned/basic/${group}/${iter}/${num_steps}/${n}_chunks/${MODEL_TAG}"

    echo "group: $group"
    echo "iter: $iter"
    echo "SAVE_ROOT: $SAVE_ROOT"
else
    echo "Path format not matched"
    exit 1
fi

if [[ -f "$CHECKPOINT_DIR/model_ema_bf16.pt" && -f "$CHECKPOINT_DIR/model_ema_fp32.pt" && -f "$CHECKPOINT_DIR/model.pt" ]]; then
    echo "Converted checkpoint files already exist, skip conversion."
else
    python ./scripts/convert_distcp_to_pt.py "$CHECKPOINT_DIR/model" "$CHECKPOINT_DIR"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python examples/action_conditioned.py \
    -i cosmos_predict2/_src/predict2/action/eval/forward_reverse_inference_params.json \
    -o "$SAVE_ROOT" \
    --save-root "$SAVE_ROOT" \
    --config-file cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py \
    --checkpoint-path "$CHECKPOINT_DIR/${MODEL}" \
    --experiment "$experiment" \
    --num-steps "$num_steps" \
    --save-fps 3 \
    --eval-reverse-action-num-chunks "$n"