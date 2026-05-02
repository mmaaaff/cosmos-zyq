from types import SimpleNamespace

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.predict2.action.callbacks.rollout_reward_validation import ActionRolloutRewardValidation
from cosmos_predict2.experiments.base import action as action_experiments


class _MockActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        self.config = SimpleNamespace(
            grpo={"num_steps": 5, "guidance": 0.0, "shift": 5.0},
            text_encoder_config=None,
            ema=SimpleNamespace(enabled=False),
        )
        self.generate_calls = []

    def generate_samples_from_batch(self, data_batch, **kwargs):
        self.generate_calls.append((data_batch, kwargs))
        idx = len(self.generate_calls)
        _, _, t, h, w = data_batch["video"].shape
        return torch.full((1, 3, t, h, w), 0.1 * idx, dtype=torch.float32)

    def decode(self, latents):
        return latents

    def collect_rollout_and_rewards(self, *args, **kwargs):
        raise AssertionError("validation must not use GRPO SDE rollout collection")


def test_action_rollout_validation_uses_unipc_generation_not_grpo_rollout():
    callback = ActionRolloutRewardValidation(
        every_n=1,
        chunk_size=2,
        max_chunks_per_episode=2,
        save_video_count=0,
        num_steps=5,
        guidance=0.0,
        shift=5.0,
    )
    model = _MockActionModel()
    gt_video = np.zeros((7, 2, 2, 3), dtype=np.uint8)
    actions = np.zeros((6, 7), dtype=np.float32)

    result = callback._run_rollout_from_arrays(model, gt_video, actions, key="episode", episode_seed=123)

    assert len(model.generate_calls) == 2
    assert result.pred_video.shape == (1, 3, 5, 2, 2)
    assert result.gt_video.shape == (1, 3, 5, 2, 2)
    assert result.action.shape == (1, 4, 7)
    for data_batch, kwargs in model.generate_calls:
        assert data_batch["action"].shape == (1, 2, 7)
        assert kwargs["is_negative_prompt"] is False
        assert kwargs["num_steps"] == 5
        assert kwargs["guidance"] == 0.0
        assert kwargs["shift"] == 5.0


def test_action_rollout_validation_skips_repeated_condition_frame_between_chunks():
    callback = ActionRolloutRewardValidation(every_n=1, chunk_size=2, max_chunks_per_episode=2)
    model = _MockActionModel()
    gt_video = np.zeros((7, 2, 2, 3), dtype=np.uint8)
    actions = np.zeros((6, 7), dtype=np.float32)

    result = callback._run_rollout_from_arrays(model, gt_video, actions, key="episode", episode_seed=123)
    temporal_values = result.pred_video[0, 0, :, 0, 0].tolist()

    assert len(temporal_values) == 5
    assert temporal_values[:3] == [26, 26, 26]
    assert temporal_values[3:] == [51, 51]


def test_action_rollout_validation_aggregates_reward_metrics():
    callback = ActionRolloutRewardValidation(every_n=1)

    metrics = callback._aggregate_reward_metrics(
        rewards=[torch.tensor([1.0, 3.0])],
        component_values={"ssim": [torch.tensor([0.2, 0.4])]},
    )

    assert metrics["val_rollout/reward_count"] == 2.0
    assert metrics["val_rollout/reward_mean"] == 2.0
    assert metrics["val_rollout/reward_std"] == 1.0
    assert abs(metrics["val_rollout/reward_component_ssim_mean"] - 0.3) < 1e-6
    assert abs(metrics["val_rollout/reward_component_ssim_std"] - 0.1) < 1e-6


def test_grpo_base_experiment_instantiates_rollout_reward_validation_callback():
    callback_cfg = action_experiments.ac_reason_embeddings_rectified_flow_2b_256_320_grpo_base["trainer"][
        "callbacks"
    ]["rollout_reward_validation"]

    callback = instantiate(callback_cfg)

    assert isinstance(callback, ActionRolloutRewardValidation)
    assert callback.max_eval_episodes == 8
    assert callback.max_chunks_per_episode == 4
    assert callback.save_video_count == 4
    assert callback.sampler_type == "unipc"
