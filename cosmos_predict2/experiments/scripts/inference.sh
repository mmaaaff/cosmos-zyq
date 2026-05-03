CHECKPOINT_DIR="/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/cosmos_predict2_action_conditioned_grpo/cotracker3/2b_bridge_action_conditioned_grpo_cotracker/checkpoints/iter_000000300"
experiment="ac_reason_embeddings_rectified_flow_2b_256_320_grpo_cotracker"
MODEL="model.pt" # model.pt | model_ema_bf16.pt | model_ema_fp32.pt
num_steps=20
GPU_ID=0

if [[ "$CHECKPOINT_DIR" =~ cosmos_predict2_action_conditioned_grpo/([^/]+)/.*/checkpoints/iter_([0-9]+)$ ]]; then
    group="${BASH_REMATCH[1]}"
    iter="${BASH_REMATCH[2]}"

    MODEL_TAG="${MODEL%.pt}"
    SAVE_ROOT="./outputs/action_conditioned/basic/${group}/${iter}/${num_steps}/${MODEL_TAG}"

    echo "group: $group"
    echo "iter: $iter"
    echo "SAVE_ROOT: $SAVE_ROOT"
else
    echo "Path format not matched"
    exit 1
fi

python ./scripts/convert_distcp_to_pt.py "$CHECKPOINT_DIR/model" "$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" python examples/action_conditioned.py \
    -i assets/action_conditioned/basic/inference_params.json \
    -o "$SAVE_ROOT" \
    --save-root "$SAVE_ROOT" \
    --config-file cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py \
    --checkpoint-path "$CHECKPOINT_DIR/${MODEL}" \
    --experiment "$experiment" \
    --num-steps "$num_steps" \
    --save-fps 3

python /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/assets/action_conditioned/basic/concate_videos.py \
  --dir_a $SAVE_ROOT \
  --dir_b /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/original/iter_000150000_model_ema_fp32.pt \
  --dir_c /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/gt \
  --output_dir ${SAVE_ROOT}/compare \
  --fps 3 \
  --overwrite