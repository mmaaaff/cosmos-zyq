# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.config_helper import override
from cosmos_predict2._src.predict2.action.configs.action_conditioned import config as action_config_module
from cosmos_predict2._src.predict2.action.configs.action_conditioned import config_grpo as action_grpo_config_module
from cosmos_predict2._src.predict2.action.configs.action_conditioned.reward import (
    DummyRewardConfig,
    SSIMRewardConfig,
    build_mixed_reward_config,
)
from cosmos_predict2._src.predict2.action.models import action_conditioned_video2world_rectified_flow_grpo_model as grpo_model_module
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
)
from cosmos_predict2._src.predict2.rl.grpo_sde_sampler import grpo_sde_step
from cosmos_predict2._src.predict2.rl.reward import (
    CompositeRewardModel,
    CoTrackerCenteredVelocityReward,
    DummyRewardModel,
    SSIM_Reward,
)


def _stub_rectified_flow_init(self, config):
    torch.nn.Module.__init__(self)
    self.config = config


def test_grpo_config_default_reward_instantiates_ssim(monkeypatch):
    monkeypatch.setattr(ActionVideo2WorldModelRectifiedFlow, "__init__", _stub_rectified_flow_init)

    cfg = action_grpo_config_module.make_config()

    model = instantiate(cfg.model)

    assert isinstance(model._reward_model, SSIM_Reward)


def test_grpo_reward_override_instantiates_dummy(monkeypatch):
    monkeypatch.setattr(ActionVideo2WorldModelRectifiedFlow, "__init__", _stub_rectified_flow_init)

    cfg = action_grpo_config_module.make_config()
    cfg = override(cfg, ["--", "reward=dummy"])

    model = instantiate(cfg.model)

    assert isinstance(model._reward_model, DummyRewardModel)


def test_grpo_reward_override_supports_fine_grained_params(monkeypatch):
    monkeypatch.setattr(ActionVideo2WorldModelRectifiedFlow, "__init__", _stub_rectified_flow_init)

    cfg = action_grpo_config_module.make_config()
    cfg = override(
        cfg,
        ["--", "reward=cotracker_centered_velocity", "model.config.reward.temporal_radius=3"],
    )

    model = instantiate(cfg.model)

    assert isinstance(model._reward_model, CoTrackerCenteredVelocityReward)
    assert model._reward_model.temporal_radius == 3


def test_grpo_model_requires_reward_config(monkeypatch):
    monkeypatch.setattr(ActionVideo2WorldModelRectifiedFlow, "__init__", _stub_rectified_flow_init)

    cfg = action_config_module.make_config()
    cfg = override(cfg, ["--", "model=action_conditioned_video2world_fsdp_rectified_flow_grpo"])

    with pytest.raises(ValueError, match="GRPO reward config is required"):
        instantiate(cfg.model)


def test_grpo_mixed_reward_instantiates_composite(monkeypatch):
    monkeypatch.setattr(ActionVideo2WorldModelRectifiedFlow, "__init__", _stub_rectified_flow_init)

    cfg = action_grpo_config_module.make_config()
    cfg = override(cfg, ["--", "reward=mixed"])
    cfg.model.config.reward = build_mixed_reward_config(
        {
            "dummy": {"weight": 0.25, "reward": DummyRewardConfig},
            "ssim": {"weight": 0.75, "reward": SSIMRewardConfig},
        }
    )

    model = instantiate(cfg.model)

    assert isinstance(model._reward_model, CompositeRewardModel)
    assert model._reward_model.reward_weights == {"dummy": 0.25, "ssim": 0.75}


class _MockRewardModel:
    def __call__(self, inp):
        del inp
        return torch.tensor([1.0], dtype=torch.float32)

    def get_last_metrics(self):
        return {"ssim": torch.tensor([0.25], dtype=torch.float32)}


class _MockScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([], dtype=torch.int64)
        self.sigmas = torch.tensor([], dtype=torch.float32)

    def set_timesteps(self, num_steps, device, shift, use_kerras_sigma):
        del num_steps, shift, use_kerras_sigma
        self.timesteps = torch.tensor([2, 1], dtype=torch.int64, device=device)
        self.sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32, device=device)


def test_collect_rollout_and_rewards_caches_reward_metrics(monkeypatch):
    model = object.__new__(grpo_model_module.ActionVideo2WorldModelRectifiedFlowGRPO)
    model.config = SimpleNamespace(
        grpo={},
        text_encoder_config=None,
        state_ch=1,
        use_kerras_sigma_at_inference=False,
    )
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model.input_data_key = "video"
    model.input_image_key = "images"
    model.tokenizer = SimpleNamespace(get_latent_num_frames=lambda t: t, spatial_compression_factor=1)
    model.net = SimpleNamespace(is_context_parallel_enabled=False)
    model.sample_scheduler = _MockScheduler()
    model._reward_model = _MockRewardModel()
    model._normalize_video_databatch_inplace = lambda data_batch: None
    model._augment_image_dim_inplace = lambda data_batch: None
    model.is_image_batch = lambda data_batch: False
    model.get_velocity_fn_from_batch = lambda data_batch, guidance, is_negative_prompt=False: (
        lambda init_noise, latents, t: torch.zeros_like(latents)
    )
    model.decode = lambda latents: latents
    model._compute_advantage = lambda rewards, hp: rewards
    monkeypatch.setattr(
        model,
        "_get_grpo_params",
        lambda: SimpleNamespace(
            num_steps=2,
            guidance=1.0,
            shift=1.0,
            eta=0.0,
            seed=1,
            num_generations=1,
            use_group_adv=False,
            adv_clip_max=5.0,
        ),
    )
    monkeypatch.setattr(
        grpo_model_module,
        "grpo_sde_step",
        lambda **kwargs: SimpleNamespace(
            next_latents=kwargs["latents"],
            log_prob=torch.zeros(kwargs["latents"].shape[0], dtype=torch.float32, device=kwargs["latents"].device),
        ),
    )

    samples = model.collect_rollout_and_rewards({"video": torch.zeros(1, 1, 2, 2, 2, dtype=torch.float32)})

    assert set(samples.reward_metrics.keys()) == {"ssim"}
    assert torch.allclose(samples.reward_metrics["ssim"], torch.tensor([0.25], dtype=torch.float32))
    assert samples.timestep_tokens.tolist() == [2]
    assert samples.latents.shape[1] == 1
    assert samples.next_latents.shape[1] == 1
    assert samples.old_log_probs.shape[1] == 1


def test_compute_grpo_loss_logs_reward_component_metrics(monkeypatch):
    model = object.__new__(grpo_model_module.ActionVideo2WorldModelRectifiedFlowGRPO)
    monkeypatch.setattr(
        model,
        "_get_grpo_params",
        lambda: SimpleNamespace(
            timestep_fraction=1.0,
            seed=1,
            eta=0.0,
            clip_range=0.1,
        ),
    )
    monkeypatch.setattr(
        grpo_model_module,
        "grpo_sde_step",
        lambda **kwargs: SimpleNamespace(
            log_prob=torch.zeros(kwargs["latents"].shape[0], dtype=torch.float32, device=kwargs["latents"].device),
            next_latents=kwargs["latents"],
        ),
    )

    samples = grpo_model_module.GrpoRolloutSamples(
        latents=torch.zeros(1, 1, 1, 1, 1, 1, dtype=torch.float32),
        next_latents=torch.zeros(1, 1, 1, 1, 1, 1, dtype=torch.float32),
        old_log_probs=torch.zeros(1, 1, dtype=torch.float32),
        timestep_tokens=torch.tensor([1], dtype=torch.int64),
        sigmas=torch.tensor([1.0, 0.0], dtype=torch.float32),
        init_noise=torch.zeros(1, 1, 1, 1, 1, dtype=torch.float32),
        rewards=torch.tensor([1.0], dtype=torch.float32),
        reward_metrics={
            "ssim": torch.tensor([0.25], dtype=torch.float32),
            "vjepa2": torch.tensor([0.75], dtype=torch.float32),
        },
        advantages=torch.tensor([0.5], dtype=torch.float32),
        velocity_fn=lambda init_noise, latents, t: torch.zeros_like(latents),
        is_image_batch=False,
    )

    output_batch, loss = model.compute_grpo_loss(samples, update_seed=0)

    assert loss.ndim == 0
    assert "reward_component_ssim_mean" in output_batch
    assert "reward_component_ssim_std" in output_batch
    assert "reward_component_vjepa2_mean" in output_batch
    assert "reward_component_vjepa2_std" in output_batch


def test_grpo_sde_step_uses_dancegrpo_score_correction():
    latents = torch.tensor([[1.2]], dtype=torch.float32)
    velocity = torch.tensor([[0.5]], dtype=torch.float32)
    sigma = torch.tensor(0.8, dtype=torch.float32)
    sigma_next = torch.tensor(0.6, dtype=torch.float32)
    eta = 0.3

    out = grpo_sde_step(
        latents=latents,
        velocity=velocity,
        sigma=sigma,
        sigma_next=sigma_next,
        eta=eta,
        noise=torch.zeros_like(latents),
    )

    dsigma = sigma_next - sigma
    pred_x0 = latents - sigma * velocity
    score = -(latents - pred_x0 * (1.0 - sigma)) / (sigma * sigma)
    expected_mean = latents + dsigma * velocity + (-0.5 * eta * eta * score) * dsigma

    assert torch.allclose(out.pred_x0, pred_x0)
    assert torch.allclose(out.next_latents, expected_mean)
