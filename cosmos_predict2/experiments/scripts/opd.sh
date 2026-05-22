cd /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq

set -e
set -o pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cosmos

cd /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq

torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_grpo_opd_fixed_teacher ~dataloader_train.dataloaders \
    job.wandb_mode=offline