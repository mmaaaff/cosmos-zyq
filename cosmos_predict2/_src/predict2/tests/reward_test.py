# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
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


def _read_video_as_bcthw(video_path: str | os.PathLike[str], device: torch.device) -> torch.Tensor:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import imageio

    frames = np.asarray(imageio.mimread(video_path), dtype=np.uint8)  # [T, H, W, C]
    if frames.size == 0:
        raise ValueError(f"Video file has no readable frames: {video_path}")
    if frames.ndim == 3:
        frames = frames[..., None]
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif frames.shape[-1] > 3:
        frames = frames[..., :3]

    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(device=device)


"""
python - <<'PY'
from cosmos_predict2._src.predict2.tests.reward_test import run_cotracker_forward_test

reward = run_cotracker_forward_test(
    pred_video_path="/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/OF/000001000/20/model/0_chunk.mp4",
    gt_video_path="/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/OF/000001000/20/model/0_chunk.mp4",
    output_dir="/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/outputs/action_conditioned/basic/test",
    checkpoint_path="ckpt/cotracker/scaled_offline.pth",
    device="cuda",
)

print("reward:", reward)
PY
"""
def run_cotracker_forward_test(
    pred_video_path: str,
    gt_video_path: str,
    output_dir: str,
    checkpoint_path: str = "ckpt/cotracker/scaled_offline.pth",
    device: str = "cuda",
    fps_downsample_ratio: int = 1,
) -> torch.Tensor:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CoTracker checkpoint not found: {checkpoint_path}")

    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for CoTracker forward_test, but torch.cuda.is_available() is False.")

    pred = _read_video_as_bcthw(pred_video_path, device=device_obj)
    gt = _read_video_as_bcthw(gt_video_path, device=device_obj)
    reward = CoTrackerCenteredVelocityReward(
        checkpoint_path=checkpoint_path,
        input_resolution=[224, 224],
        patch_size=4,
        temporal_radius=2,
        tau=1.0,
        window_batch_size=32,
        score_mode="charbonnier",
        eps=1e-3,
        min_active_points=16,
        invisibility_penalty=1.0,
        offline=True,
        fps_downsample_ratio=fps_downsample_ratio,
    )
    return reward.forward_test(
        RewardInput(video=pred, text=None, action=None, metadata={"gt_video": gt}),
        output_dir=output_dir,
    )


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
    weights = reward.get_component_weights()
    assert weights == {"a": 0.25, "b": 0.75}
    weights["a"] = 99.0
    assert reward.get_component_weights() == {"a": 0.25, "b": 0.75}


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


# python -m pytest cosmos_predict2/_src/predict2/tests/reward_test.py::test_cotracker_forward_test_writes_velocity_visualization -o addopts=''
def test_cotracker_forward_test_writes_velocity_visualization(monkeypatch, tmp_path):
    reward = CoTrackerCenteredVelocityReward(
        checkpoint_path="dummy.pth",
        input_resolution=[8, 8],
        patch_size=4,
        temporal_radius=1,
        tau=0.1,
        min_active_points=0,
    )

    points = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]], dtype=torch.float32)
    offsets = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
    tracks = torch.stack([points, points + offsets * 0.5, points + offsets], dim=0).unsqueeze(0).repeat(3, 1, 1, 1)
    visibility = torch.ones(3, 3, 4, dtype=torch.bool)

    calls = iter([(tracks, visibility), (tracks, visibility)])
    monkeypatch.setattr(reward, "_run_tracker", lambda _: next(calls))

    saved = {}

    import imageio

    def fake_mimwrite(path, frames, fps, quality):
        saved["path"] = path
        saved["frames"] = np.asarray(frames)
        saved["fps"] = fps
        saved["quality"] = quality

    monkeypatch.setattr(imageio, "mimwrite", fake_mimwrite)

    video = _make_video(frames=5)
    out = reward.forward_test(
        RewardInput(video=video, text=None, action=None, metadata={"gt_video": video.clone()}),
        output_dir=tmp_path,
        fps=7,
    )

    assert out.shape == (1,)
    assert os.path.basename(saved["path"]) == "cotracker_velocity_b0.mp4"
    assert saved["frames"].shape == (5, 8, 16, 3)
    assert saved["fps"] == 7
    assert ((saved["frames"] == np.array([255, 0, 0])).all(axis=-1)).any()
    assert ((saved["frames"] == np.array([0, 0, 255])).all(axis=-1)).any()


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
