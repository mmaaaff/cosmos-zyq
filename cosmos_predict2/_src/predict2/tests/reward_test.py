# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cosmos_predict2._src.predict2.rl.reward import (
    BaseRewardModel,
    CompositeRewardModel,
    CoTrackerCenteredVelocityReward,
    RewardInput,
)


def _make_video(batch: int = 1, frames: int = 5, height: int = 8, width: int = 8) -> torch.Tensor:
    return torch.zeros(batch, 3, frames, height, width, dtype=torch.uint8)


class _ConstantReward(BaseRewardModel):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)
        self.unloaded = False

    def forward(self, inp: RewardInput) -> torch.Tensor:
        assert inp.video is not None
        return torch.full((inp.video.shape[0],), self.value, dtype=torch.float32, device=inp.video.device)

    def unload(self, clear_cuda_cache: bool = True) -> None:
        del clear_cuda_cache
        self.unloaded = True


class _WrongShapeReward(BaseRewardModel):
    def forward(self, inp: RewardInput) -> torch.Tensor:
        assert inp.video is not None
        return torch.zeros((inp.video.shape[0], 1), dtype=torch.float32, device=inp.video.device)


def test_composite_reward_weighted_sum_and_metrics():
    reward = CompositeRewardModel(
        components={
            "a": {"weight": 0.25, "reward": _ConstantReward(2.0)},
            "b": {"weight": 0.75, "reward": _ConstantReward(4.0)},
        }
    )

    video = _make_video(batch=2)
    out = reward(RewardInput(video=video, text=None, action=None, metadata={"gt_video": video.clone()}))

    assert out.shape == (2,)
    assert torch.allclose(out, torch.full((2,), 3.5, dtype=torch.float32))
    metrics = reward.get_last_metrics()
    assert set(metrics.keys()) == {"a", "b"}
    assert torch.allclose(metrics["a"], torch.full((2,), 2.0, dtype=torch.float32))
    assert torch.allclose(metrics["b"], torch.full((2,), 4.0, dtype=torch.float32))


def test_composite_reward_requires_non_empty_components():
    with pytest.raises(ValueError, match="at least one reward component"):
        CompositeRewardModel(components={})


def test_composite_reward_rejects_wrong_shape():
    reward = CompositeRewardModel(
        components={
            "ok": {"weight": 1.0, "reward": _ConstantReward(1.0)},
            "bad": {"weight": 1.0, "reward": _WrongShapeReward()},
        }
    )
    video = _make_video(batch=1)

    with pytest.raises(ValueError, match="must return shape \\[B\\]"):
        reward(RewardInput(video=video, text=None, action=None, metadata={"gt_video": video.clone()}))


def test_composite_reward_unload_calls_children():
    reward_a = _ConstantReward(1.0)
    reward_b = _ConstantReward(2.0)
    reward = CompositeRewardModel(
        components={
            "a": {"weight": 1.0, "reward": reward_a},
            "b": {"weight": 1.0, "reward": reward_b},
        }
    )

    reward.unload()

    assert reward_a.unloaded is True
    assert reward_b.unloaded is True
    assert reward.get_last_metrics() == {}


def test_cotracker_centered_reward_zero_when_tracks_match(monkeypatch):
    reward = CoTrackerCenteredVelocityReward(
        checkpoint_path="dummy.pth",
        input_resolution=[8, 8],
        patch_size=4,
        temporal_radius=1,
        tau=0.1,
        min_active_points=1,
    )

    tracks = torch.tensor(
        [
            [
                [[1.0, 1.0], [5.0, 5.0]],
                [[2.0, 1.0], [6.0, 5.0]],
                [[3.0, 1.0], [7.0, 5.0]],
            ],
            [
                [[1.5, 1.0], [5.5, 5.0]],
                [[2.5, 1.0], [6.5, 5.0]],
                [[3.5, 1.0], [7.5, 5.0]],
            ],
            [
                [[2.0, 1.0], [6.0, 5.0]],
                [[3.0, 1.0], [7.0, 5.0]],
                [[4.0, 1.0], [8.0, 5.0]],
            ],
        ],
        dtype=torch.float32,
    )
    visibility = torch.ones(3, 3, 2, dtype=torch.bool)

    calls = iter([(tracks, visibility), (tracks, visibility)])
    monkeypatch.setattr(reward, "_run_tracker", lambda _: next(calls))

    video = _make_video(frames=5)
    out = reward(RewardInput(video=video, text=None, action=None, metadata={"gt_video": video.clone()}))
    assert out.shape == (1,)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_cotracker_centered_reward_returns_zero_when_no_valid_points(monkeypatch):
    reward = CoTrackerCenteredVelocityReward(
        checkpoint_path="dummy.pth",
        input_resolution=[8, 8],
        patch_size=4,
        temporal_radius=1,
        tau=0.1,
        min_active_points=1,
    )

    tracks = torch.zeros(3, 3, 2, 2, dtype=torch.float32)
    invisible = torch.zeros(3, 3, 2, dtype=torch.bool)

    calls = iter([(tracks, invisible), (tracks, invisible)])
    monkeypatch.setattr(reward, "_run_tracker", lambda _: next(calls))

    video = _make_video(frames=5)
    out = reward(RewardInput(video=video, text=None, action=None, metadata={"gt_video": video.clone()}))
    assert torch.equal(out, torch.zeros_like(out))


def test_cotracker_centered_reward_requires_enough_frames():
    reward = CoTrackerCenteredVelocityReward(
        checkpoint_path="dummy.pth",
        input_resolution=[8, 8],
        patch_size=4,
        temporal_radius=2,
    )
    video = _make_video(frames=3)

    with pytest.raises(ValueError, match="requires T >="):
        reward(RewardInput(video=video, text=None, action=None, metadata={"gt_video": video.clone()}))
