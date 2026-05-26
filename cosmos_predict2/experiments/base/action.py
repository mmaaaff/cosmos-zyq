# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.callbacks.rollout_reward_validation import ActionRolloutRewardValidation
from cosmos_predict2._src.predict2.action.configs.action_conditioned.reward import (
    OpticalFlowRewardConfig,
    CoTrackerCenteredVelocityRewardConfig,
    VJEPA2RewardConfig,
)
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# Use the post-trained checkpoint which has the correct experiment reference
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]  # This uses post_trained=True by default


"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320 ~dataloader_train.dataloaders \
    job.wandb_mode=offline
"""
ac_reason_embeddings_rectified_flow_2b_256_320 = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "bridge_13frame_480_640_train"},
            {"override /data_val": "bridge_13frame_480_640_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name="2b_bridge_action_conditioned",
            # zyq
            wandb_reuse_id=False,
            wandb_resume="never",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=2_000,
            # pyrefly: ignore  # missing-attribute
            load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        trainer=dict(
            max_iter=1_000_000,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                heart_beat=dict(
                    save_s3=False,
                ),
                iter_speed=dict(
                    hit_thres=100,
                    save_s3=False,
                ),
                device_monitor=dict(
                    save_s3=False,
                ),
                wandb=dict(
                    save_s3=False,
                ),
                wandb_10x=dict(
                    save_s3=False,
                ),
                dataloader_speed=dict(
                    save_s3=False,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                # NOTE: this should be 1 for the action conditioned model
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                # overwrite the probs to disable random num of conditional frames
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=2,
            sampler=dict(
                dataset=dict(fps_downsample_ratio=1, video_size=[256, 320]),
            ),
            dataset=dict(fps_downsample_ratio=1, video_size=[256, 320]),
        ),
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_grpo ~dataloader_train.dataloaders \
    job.wandb_mode=offline \
    2>&1 | tee /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/train.log
"""
"""
inference

CHECKPOINTS_DIR=/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/cosmos_predict2_action_conditioned_grpo/cosmos_predict_v2p5/2b_bridge_action_conditioned_grpo_vjepa/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

SAVE_ROOT=outputs/action_conditioned/basic/CT2/iter_000000990_model.pt
python examples/action_conditioned.py \
-i assets/action_conditioned/basic/inference_params.json -o $SAVE_ROOT \
--save-root $SAVE_ROOT \
--config-file cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py \
--checkpoint-path $CHECKPOINT_DIR/model.pt \
--experiment ac_reason_embeddings_rectified_flow_2b_256_320_grpo_cotracker
"""
rollout_n=4
ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base = LazyDict(
    dict(
        defaults=[
            f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
            # NOTE: GRPO 版本的模型配置组（由 action_conditioned/configs/action_conditioned/model.py 注册）
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow_grpo"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "bridge_13frame_480_640_train"},
            {"override /data_val": "bridge_13frame_480_640_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned_grpo",
            group="cosmos_predict_v2p5",
            wandb_reuse_id=False,  # True / False
            wandb_resume="never",  # "allow" / "must" / "never"
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=10,
            # pyrefly: ignore  # missing-attribute
            load_path="/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_bridge_action_conditioned/checkpoints/iter_000150000/model_ema_fp32.pt",  # 直接使用 post-train 过的模型
            #load_path=get_checkpoint_path("s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320/checkpoints/iter_000016000/model"),
            load_training_state=True,
            strict_resume=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            logging_iter=5,
            grad_accum_iter=rollout_n,  # note: 与 rollout_num_batches 一致
            # resume_iteration=0,  # 强制 trainer_grpo 的 iteration 起点。设置为非 None 值则禁止加载 optimizer, scheduler, grad_scaler 状态，设为 None 则视作继续训练，加载这些状态
            # NOTE: GRPO 训练通常更慢；可以视情况把采样 callback 频率调低
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=1,  # following Dance-GRPO
                ),
                every_n_sample_reg=dict(
                    every_n=200,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=200,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                heart_beat=dict(
                    save_s3=False,
                ),
                iter_speed=dict(
                    hit_thres=200,
                    save_s3=False,
                ),
                device_monitor=dict(
                    save_s3=False,
                ),
                wandb=dict(
                    save_s3=False,
                ),
                wandb_10x=None,
                dataloader_speed=dict(
                    save_s3=False,
                ),
                # rollout_reward_validation=L(ActionRolloutRewardValidation)(
                #     every_n=200,
                #     max_eval_episodes=8,
                #     max_chunks_per_episode=4,
                #     save_video_count=4,
                #     sampler_type="unipc",
                #     num_steps=None,
                #     guidance=None,
                #     shift=None,
                #     seed=0,
                #     save_fps=4,
                #     use_ema=False,
                # ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                # ---------------- action-conditioned base config ----------------
                # use_kerras_sigma_at_inference=False,
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                ),
                # ---------------- GRPO hyperparameters (placeholders) ----------------
                # NOTE: 这些字段由 GRPO 模型的 Config 定义；在 dummy reward 阶段先给一个可跑通的默认值。
                grpo=dict(
                    num_steps=20,
                    shift=5.0,  # Cosmos 原始代码 5.0，但感觉应该没用？因为似乎 use_kerras_sigma_at_inference 是 True（不过 grpo 这里我已经改成 flase）
                    eta=0.3,  # GRPO: 0.3
                    guidance=0.0,  # 若 guidance > 0, 则 num_updates 应该要降低
                    seed=1,
                    use_group_adv=True,
                    num_generations=12,  # 一个 prompt 生成多少个样本，即 group size
                    init_same_noise=True,
                    timestep_fraction=0.6,
                    rollout_num_batches=rollout_n,  # 一次 rollout 多少个 batch，注意这里实际值要乘以 GPU 数量再乘以 batch_size 才得到 prompts per iter
                    num_updates=4,  # 用一组 rollout 训练多少轮
                    clip_range=1e-4,
                    adv_clip_max=5.0,
                ),
            ),
        ),
        dataloader_train=dict(
            # NOTE: GRPO online rollout 非常吃算力，先用小 batch_size
            batch_size=1,
            sampler=dict(
                dataset=dict(fps_downsample_ratio=1, video_size=[256, 320]),
            ),
            dataset=dict(fps_downsample_ratio=1, video_size=[256, 320]),
        ),  # dataloader 定义位于 cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py line 111
    ),
    flags={"allow_objects": True},
)

multichunk_rollout_chunks = 2
multichunk_action_chunk_size = 12
ac_reason_embeddings_rectified_flow_2b_256_320_multichunk_grpo_base = LazyDict(
    dict(
        defaults=[
            {"/experiment/grpo_base": "ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base"},
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow_multichunk_grpo"},
            "_self_",
        ],
        job=dict(
            group="multichunk_grpo",
            name="2b_bridge_action_conditioned_multichunk_grpo_base",
        ),
        model=dict(
            config=dict(
                # Keep the model native chunk fixed at 12 actions.
                state_t=1 + multichunk_action_chunk_size // 4,
                net=dict(
                    num_action_per_chunk=multichunk_action_chunk_size,
                ),
                grpo=dict(
                    num_rollout_chunks=multichunk_rollout_chunks,
                    action_chunk_size=multichunk_action_chunk_size,
                ),
            ),
        ),
        dataloader_train=dict(
            sampler=dict(
                dataset=dict(num_action_per_chunk=multichunk_rollout_chunks * multichunk_action_chunk_size),
            ),
            dataset=dict(num_action_per_chunk=multichunk_rollout_chunks * multichunk_action_chunk_size),
        ),
        dataloader_val=dict(
            sampler=dict(
                dataset=dict(num_action_per_chunk=multichunk_rollout_chunks * multichunk_action_chunk_size),
            ),
            dataset=dict(num_action_per_chunk=multichunk_rollout_chunks * multichunk_action_chunk_size),
        ),
    ),
    flags={"allow_objects": True},
)


ac_reason_embeddings_rectified_flow_2b_256_320_multichunk_grpo_ssim = LazyDict(
    dict(
        defaults=[
            {"/experiment/grpo_base": "ac_reason_embeddings_rectified_flow_2b_256_320_multichunk_grpo_base"},
            {"override /reward": "ssim"},
            "_self_",
        ],
        job=dict(
            group="multichunk_grpo",
            name="2b_bridge_action_conditioned_multichunk_grpo_ssim",
        ),
    ),
    flags={"allow_objects": True},
)


"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_grpo_ssim ~dataloader_train.dataloaders \
    job.wandb_mode=offline \
    2>&1 | tee /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/train.log
"""
ac_reason_embeddings_rectified_flow_2b_256_320_grpo_ssim = LazyDict(
    dict(
        defaults=[
            # 到 /experiment/grpo_base 去找 grpo_base 这个 experiment, 因为它在 cs.store 时是注册在这个 group 里的。
            # 要用绝对路径 /experiment/grpo_base 是因为目前本来就已经在 experiment group 里，不用绝对路径会变成 /experiment/experiment/...
            {"/experiment/grpo_base": "ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base"},
            {"override /reward": "ssim"},
            "_self_",
        ],
        job=dict(
            name="2b_bridge_action_conditioned_grpo_ssim",
        ),
    ),
    flags={"allow_objects": True},
)

ac_reason_embeddings_rectified_flow_2b_256_320_grpo_vjepa = LazyDict(
    dict(
        defaults=[
            {"/experiment/grpo_base": "ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base"},
            {"override /reward": "vjepa2"},
            "_self_",
        ],
        job=dict(
            group="vjepa2",
            name="2b_bridge_action_conditioned_grpo_vjepa",
        ),
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_grpo_optical_flow ~dataloader_train.dataloaders \
    job.wandb_mode=offline \
    2>&1 | tee /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/train_OF.log 
"""
ac_reason_embeddings_rectified_flow_2b_256_320_grpo_optical_flow = LazyDict(
    dict(
        defaults=[
            {"/experiment/grpo_base": "ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base"},
            {"override /reward": "optical_flow"},
            "_self_",
        ],
        job=dict(
            group="OF_5steps",
            name="2b_bridge_action_conditioned_grpo_optical_flow",
        ),
        model=dict(
            config=dict(
                grpo=dict(
                    num_steps=5
                )
            )
        )
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_grpo_cotracker ~dataloader_train.dataloaders \
    job.wandb_mode=offline \
    2>&1 | tee /inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/train_cotracker.log 
"""
ac_reason_embeddings_rectified_flow_2b_256_320_grpo_cotracker = LazyDict(
    dict(
        defaults=[
            {"/experiment/grpo_base": "ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base"},
            {"override /reward": "cotracker_centered_velocity"},
            "_self_",
        ],
        job=dict(
            group="cotracker_tau=0",
            name="2b_bridge_action_conditioned_grpo_cotracker",
        ),
        model=dict(
            config=dict(
                reward=dict(
                    checkpoint_path="ckpt/cotracker/scaled_offline.pth",
                    input_resolution=[224, 224],
                    patch_size=8,
                    temporal_radius=2,
                    tau=0.0,
                    window_batch_size=32,
                    score_mode="charbonnier",
                    eps=1e-3,
                    min_active_points=8,
                    invisibility_penalty=0.0,
                    offline=True,
                    fps_downsample_ratio="${dataloader_train.sampler.dataset.fps_downsample_ratio}",
                ),
                grpo=dict(
                    num_updates=2,
                )
            ),
        ),
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_grpo_mixed_reward ~dataloader_train.dataloaders \
    job.wandb_mode=offline
"""
ac_reason_embeddings_rectified_flow_2b_256_320_grpo_mixed_reward = LazyDict(
    dict(
        defaults=[
            {"/experiment/grpo_base": "ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base"},
            {"override /reward": "mixed"},
            "_self_",
        ],
        job=dict(
            group ="mixed_reward_0.7of_0.3vjepa1",
            name="2b_bridge_action_conditioned_grpo_mixed_reward",
        ),
        model=dict(
            config=dict(
                reward=dict(
                    components=dict(
                        # cotracker=dict(
                        #     weight=0.7,
                        #     reward=dict(
                        #         CoTrackerCenteredVelocityRewardConfig,
                        #         fps_downsample_ratio="${dataloader_train.sampler.dataset.fps_downsample_ratio}",  # OmegaConf/Hydra 写法
                        #     ),
                        # ),
                        optical_flow=dict(
                            weight=0.7,
                            reward=OpticalFlowRewardConfig,
                            ),
                        vjepa2=dict(
                            weight=0.3,
                            reward=VJEPA2RewardConfig,
                        ),
                    ),
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_grpo_opd_fixed_teacher ~dataloader_train.dataloaders \
    job.wandb_mode=offline
"""
ac_reason_embeddings_rectified_flow_2b_256_320_grpo_opd_fixed_teacher = LazyDict(
    dict(
        defaults=[
            {"/experiment/grpo_base": "ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base"},
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow_opd"},
            "_self_",
        ],
        job=dict(
            group="opd_fixed_teacher_kl_grad",
            name="2b_bridge_action_conditioned_grpo_opd_fixed_teacher",
        ),
        model=dict(
            config=dict(
                reward=None,
                grpo=dict(
                    eta=0.3,
                    sigma_dependent_eta=False,
                    opd_teacher_checkpoint_path="/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/cosmos_predict2_action_conditioned_grpo/OF2/2b_bridge_action_conditioned_grpo_optical_flow/checkpoints/iter_000001000/model.pt",
                    opd_kl_scale=1.0,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

cs.store(
    group="experiment/grpo_base",
    package="_global_",
    name="ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base",
    node=ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base,
)
cs.store(
    group="experiment/grpo_base",
    package="_global_",
    name="ac_reason_embeddings_rectified_flow_2b_256_320_multichunk_grpo_base",
    node=ac_reason_embeddings_rectified_flow_2b_256_320_multichunk_grpo_base,
)

for _item in [
    ac_reason_embeddings_rectified_flow_2b_256_320,
    ac_reason_embeddings_rectified_flow_2b_256_320_grpo_ssim,
    ac_reason_embeddings_rectified_flow_2b_256_320_multichunk_grpo_ssim,
    ac_reason_embeddings_rectified_flow_2b_256_320_grpo_vjepa,
    ac_reason_embeddings_rectified_flow_2b_256_320_grpo_optical_flow,
    ac_reason_embeddings_rectified_flow_2b_256_320_grpo_cotracker,
    ac_reason_embeddings_rectified_flow_2b_256_320_grpo_mixed_reward,
    ac_reason_embeddings_rectified_flow_2b_256_320_grpo_opd_fixed_teacher,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(group="experiment", package="_global_", name=f"{experiment_name}", node=_item)
