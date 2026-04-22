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
from cosmos_predict2._src.predict2.rl.reward import (
    CompositeRewardModel,
    CoTrackerCenteredVelocityReward,
    DummyRewardModel,
    OpticalFlowReward,
    SSIM_Reward,
    VJEPA2Reward,
)

DummyRewardConfig: LazyDict = L(DummyRewardModel)()

SSIMRewardConfig: LazyDict = L(SSIM_Reward)()

OpticalFlowRewardConfig: LazyDict = L(OpticalFlowReward)(
    estimator="farneback",
    score_mode="robust",
    resize_hw=[128, 160],
    frame_stride=1,
    max_frames=0,
    eps=1e-3,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

CoTrackerCenteredVelocityRewardConfig: LazyDict = L(CoTrackerCenteredVelocityReward)(
    checkpoint_path="checkpoints/cotracker/scaled_offline.pth",
    input_resolution=[224, 224],
    patch_size=8,
    temporal_radius=2,
    tau=1.0,
    window_batch_size=32,
    score_mode="charbonnier",
    eps=1e-3,
    min_active_points=16,
    invisibility_penalty=1.0,
    offline=True,
)

VJEPA2RewardConfig: LazyDict = L(VJEPA2Reward)(
    model_name="facebook/vjepa2-vitg-fpc64-384",
    num_frames=64,
    image_size=384,
    stride=1,
)

MixedRewardConfig: LazyDict = L(CompositeRewardModel)(
    components={},
    record_component_metrics=True,
)


def build_mixed_reward_config(
    components: dict[str, dict[str, object]],
    *,
    record_component_metrics: bool = True,
) -> LazyDict:
    return L(CompositeRewardModel)(
        components=components,
        record_component_metrics=record_component_metrics,
    )


def register_reward():
    cs = ConfigStore.instance()
    cs.store(group="reward", package="model.config.reward", name="dummy", node=DummyRewardConfig)
    cs.store(group="reward", package="model.config.reward", name="ssim", node=SSIMRewardConfig)
    cs.store(group="reward", package="model.config.reward", name="mixed", node=MixedRewardConfig)
    cs.store(group="reward", package="model.config.reward", name="optical_flow", node=OpticalFlowRewardConfig)
    cs.store(
        group="reward",
        package="model.config.reward",
        name="cotracker_centered_velocity",
        node=CoTrackerCenteredVelocityRewardConfig,
    )
    cs.store(group="reward", package="model.config.reward", name="vjepa2", node=VJEPA2RewardConfig)
