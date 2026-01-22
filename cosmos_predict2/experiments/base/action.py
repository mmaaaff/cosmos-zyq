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

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# Use the post-trained checkpoint which has the correct experiment reference
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]  # This uses post_trained=True by default


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
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
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config_grpo.py  \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_grpo ~dataloader_train.dataloaders \
    job.wandb_mode=offline
"""
ac_reason_embeddings_rectified_flow_2b_256_320_grpo = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
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
            name="2b_bridge_action_conditioned_grpo",
            reuse_id=False,
            wandb_resume="never",
        ),
        optimizer=dict(
            lr=1e-5,
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
            straggler_detection=dict(enabled=False),
            logging_iter=1,
            # NOTE: GRPO 训练通常更慢；可以视情况把采样 callback 频率调低
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=1,  # following Dance-GRPO
                ),
                every_n_sample_reg=dict(
                    every_n=10_000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=10_000,
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
                # ---------------- action-conditioned base config ----------------
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
                    num_steps=16,
                    shift=5.0,  # 这个数据来源于 Cosmos 原始代码，可查找 set_timesteps
                    eta=0.3,  # follows GRPO
                    guidance=3.0,
                    seed=1,
                    use_group_adv=True,
                    num_generations=12,  # 一个 prompt 生成多少个样本
                    init_same_noise=True,
                    timestep_fraction=0.6,
                    rollout_num_batches=4,  # 一次 rollout 多少个 batch，注意这里实际值要乘以 GPU 数量
                    num_updates=2,  # 用一组 rollout 训练多少轮
                    clip_range=1e-4,
                    adv_clip_max=5.0,
                ),
                reward=dict(
                    # NOTE: reward model 还未定稿，先用 dummy reward 打通链路
                    type="ssim",
                ),
            ),
        ),
        dataloader_train=dict(
            # NOTE: GRPO online rollout 非常吃算力，先用更小 batch 打通
            batch_size=1,
            sampler=dict(
                dataset=dict(fps_downsample_ratio=1, video_size=[256, 320]),
            ),
            dataset=dict(fps_downsample_ratio=1, video_size=[256, 320]),
        ),  # dataloader 定义位于 cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py line 111
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

for _item in [ac_reason_embeddings_rectified_flow_2b_256_320, ac_reason_embeddings_rectified_flow_2b_256_320_grpo]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(group="experiment", package="_global_", name=f"{experiment_name}", node=_item)
