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
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_model import (
    ActionConditionedVideo2WorldConfig,
    ActionConditionedVideo2WorldModel,
)
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_grpo_model import (
    ActionVideo2WorldModelRectifiedFlowGRPO,
    ActionVideo2WorldModelRectifiedFlowGRPOConfig,
)
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_multichunk_grpo_model import (
    ActionVideo2WorldModelRectifiedFlowMultiChunkGRPO,
)
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_opd_model import (
    ActionVideo2WorldModelRectifiedFlowOPD,
)

# EDM model
DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(ActionConditionedVideo2WorldModel)(
        config=ActionConditionedVideo2WorldConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionConditionedVideo2WorldModel)(
        config=ActionConditionedVideo2WorldConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

# rectified flow model
FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionVideo2WorldModelRectifiedFlow)(
        config=Video2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)

# rectified flow model (GRPO)
FSDP_RECTIFIED_FLOW_GRPO_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionVideo2WorldModelRectifiedFlowGRPO)(
        config=ActionVideo2WorldModelRectifiedFlowGRPOConfig(
            fsdp_shard_size=8,
            state_t=24,
            # NOTE: 默认值仅用于保证可实例化；实验配置会覆盖 `model.config.grpo`
            grpo=dict(
                num_steps=16,
                shift=5.0,
                eta=0.3,
                guidance=3.0,
                seed=1,
                timestep_fraction=1.0,
                rollout_num_batches=1,  # 一次 rollout 多少个 batch
                num_updates=4,  # 用 rollout 做多少轮更新
                use_group_adv=True,
                num_generations=4,
                adv_clip_max=5.0,
                clip_range=1e-4,
                sigma_dependent_eta=False,
            ),
        ),
        _recursive_=False,
    ),
)

# rectified flow model (OPD)
FSDP_RECTIFIED_FLOW_OPD_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionVideo2WorldModelRectifiedFlowOPD)(
        config=ActionVideo2WorldModelRectifiedFlowGRPOConfig(
            fsdp_shard_size=8,
            state_t=24,
            grpo=dict(
                num_steps=16,
                shift=5.0,
                eta=0.3,
                guidance=3.0,
                seed=1,
                timestep_fraction=1.0,
                rollout_num_batches=1,
                num_updates=4,
                use_group_adv=True,
                num_generations=4,
                adv_clip_max=5.0,
                clip_range=1e-4,
                sigma_dependent_eta=False,
                opd_teacher_checkpoint_path="",
                opd_kl_scale=1.0,
            ),
        ),
        _recursive_=False,
    ),
)

# rectified flow model (multi-chunk GRPO)
FSDP_RECTIFIED_FLOW_MULTICHUNK_GRPO_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionVideo2WorldModelRectifiedFlowMultiChunkGRPO)(
        config=ActionVideo2WorldModelRectifiedFlowGRPOConfig(
            fsdp_shard_size=8,
            state_t=24,
            grpo=dict(
                num_steps=16,
                shift=5.0,
                eta=0.3,
                guidance=3.0,
                seed=1,
                timestep_fraction=1.0,
                rollout_num_batches=1,
                num_updates=4,
                use_group_adv=True,
                num_generations=4,
                adv_clip_max=5.0,
                clip_range=1e-4,
                sigma_dependent_eta=False,
                num_rollout_chunks=2,
                action_chunk_size=12,
            ),
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="action_conditioned_video2world_ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="action_conditioned_video2world_fsdp", node=FSDP_CONFIG)
    cs.store(
        group="model",
        package="_global_",
        name="action_conditioned_video2world_fsdp_rectified_flow",
        node=FSDP_RECTIFIED_FLOW_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="action_conditioned_video2world_fsdp_rectified_flow_grpo",
        node=FSDP_RECTIFIED_FLOW_GRPO_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="action_conditioned_video2world_fsdp_rectified_flow_opd",
        node=FSDP_RECTIFIED_FLOW_OPD_CONFIG,
    )
    cs.store(
        group="model",
        package="_global_",
        name="action_conditioned_video2world_fsdp_rectified_flow_multichunk_grpo",
        node=FSDP_RECTIFIED_FLOW_MULTICHUNK_GRPO_CONFIG,
    )
