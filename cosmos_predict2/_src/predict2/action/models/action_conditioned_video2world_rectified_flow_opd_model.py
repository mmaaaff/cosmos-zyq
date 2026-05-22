"""
OPD-specialized Action-conditioned Video2World Rectified Flow model.

This class keeps the DanceGRPO rollout/training scaffold from
`ActionVideo2WorldModelRectifiedFlowGRPO`, but computes OPD's KL advantage during
the policy update from fixed rollout transitions, matching Flow-OPD's structure:
rollout is no-grad; training recomputes student/reference transition means.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor
from cosmos_predict2._src.imaginaire.utils.fsdp_helper import hsdp_device_mesh
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_grpo_model import (
    ActionVideo2WorldModelRectifiedFlowGRPO,
    ActionVideo2WorldModelRectifiedFlowGRPOConfig,
    GrpoRolloutSamples,
    _get_data_parallel_rank_for_seed,
    _GrpoHyperParams,
)
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
)
from cosmos_predict2._src.predict2.checkpointer.dcp import DefaultLoadPlanner, ModelWrapper
from cosmos_predict2._src.predict2.rl.grpo_sde_sampler import grpo_sde_step
from cosmos_predict2._src.predict2.utils.model_loader import load_model_state_dict_from_checkpoint


@dataclass
class _OpdHyperParams(_GrpoHyperParams):
    opd_teacher_checkpoint_path: str = ""
    opd_kl_scale: float = 1.0


@dataclass
class OpdRolloutSamples(GrpoRolloutSamples):
    teacher_velocity_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None


class ActionVideo2WorldModelRectifiedFlowOPD(ActionVideo2WorldModelRectifiedFlowGRPO):
    """
    OPD training model.

    Rollout remains no-grad and only caches fixed transitions plus logging metrics.
    The differentiable KL advantage is recomputed in `compute_grpo_loss()`.
    """

    config: ActionVideo2WorldModelRectifiedFlowGRPOConfig

    def _requires_reward_model(self) -> bool:
        return False

    def __init__(self, config: ActionVideo2WorldModelRectifiedFlowGRPOConfig):
        super().__init__(config)
        object.__setattr__(self, "_opd_teacher_model", None)

    def _get_grpo_params(self) -> _OpdHyperParams:
        d = dict(self.config.grpo or {})
        hp = _OpdHyperParams()
        for k, v in d.items():
            if hasattr(hp, k):
                setattr(hp, k, v)
        hp.num_steps = max(int(hp.num_steps), 2)
        hp.rollout_num_batches = max(int(hp.rollout_num_batches), 1)
        hp.num_updates = max(int(hp.num_updates), 1)
        hp.num_generations = max(int(hp.num_generations), 1)
        hp.timestep_fraction = min(max(float(hp.timestep_fraction), 0.0), 1.0)
        return hp

    def _load_opd_teacher_from_pt(
        self,
        teacher_config: ActionVideo2WorldModelRectifiedFlowGRPOConfig,
        checkpoint_path: str,
    ) -> ActionVideo2WorldModelRectifiedFlow:
        fsdp_shard_size = teacher_config.fsdp_shard_size
        teacher_config.fsdp_shard_size = 1

        log.info(f"Loading OPD teacher .pt checkpoint from {checkpoint_path}")
        teacher = ActionVideo2WorldModelRectifiedFlow(teacher_config)
        teacher = load_model_state_dict_from_checkpoint(
            model=teacher,
            config=None,
            s3_checkpoint_dir=checkpoint_path,
        )
        if fsdp_shard_size > 1:
            teacher_config.fsdp_shard_size = fsdp_shard_size
            teacher.apply_fsdp(hsdp_device_mesh(sharding_group_size=fsdp_shard_size))
        return teacher

    def _load_opd_teacher_from_dcp(
        self,
        teacher: ActionVideo2WorldModelRectifiedFlow,
        checkpoint_path: str,
    ) -> None:
        model_checkpoint_path = checkpoint_path.rstrip("/")
        if os.path.basename(model_checkpoint_path) != "model":
            model_checkpoint_path = os.path.join(model_checkpoint_path, "model")

        if model_checkpoint_path.startswith("s3://"):
            raise NotImplementedError(
                "OPD teacher DCP loading currently supports local filesystem paths. "
                f"Got object-store path: {model_checkpoint_path}"
            )

        log.info(f"Loading OPD teacher DCP checkpoint from {model_checkpoint_path}")
        teacher_wrapper = ModelWrapper(teacher)
        teacher_state_dict = teacher_wrapper.state_dict()
        dcp.load(
            teacher_state_dict,
            storage_reader=FileSystemReader(model_checkpoint_path),
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        teacher_wrapper.load_state_dict(teacher_state_dict)

    def _load_opd_teacher_model(
        self,
        checkpoint_path: str,
        memory_format: torch.memory_format,
    ) -> ActionVideo2WorldModelRectifiedFlow:
        teacher_config = copy.deepcopy(self.config)
        teacher_config.ema.enabled = False
        if checkpoint_path.endswith(".pt"):
            teacher = self._load_opd_teacher_from_pt(teacher_config, checkpoint_path)
        else:
            teacher = ActionVideo2WorldModelRectifiedFlow(teacher_config)
            self._load_opd_teacher_from_dcp(teacher, checkpoint_path)

        teacher.to(device=self.tensor_kwargs["device"], memory_format=memory_format)
        teacher.on_train_start(memory_format)
        teacher.requires_grad_(False)
        teacher.eval()
        return teacher

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """
        往 callback 里加一个加载教师模型的逻辑
        """
        super().on_train_start(memory_format)
        hp = self._get_grpo_params()
        teacher = self._load_opd_teacher_model(hp.opd_teacher_checkpoint_path, memory_format)
        object.__setattr__(self, "_opd_teacher_model", teacher)
        log.info(f"Loaded OPD teacher checkpoint from {hp.opd_teacher_checkpoint_path}")

    def _get_opd_teacher_velocity_fn(
        self,
        data_batch: Dict[str, torch.Tensor],
        hp: _OpdHyperParams,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        teacher = self._opd_teacher_model
        if teacher is None:
            raise RuntimeError("OPD teacher is not loaded. `on_train_start()` must run before rollout.")
        teacher.eval()
        return teacher.get_velocity_fn_from_batch(data_batch, guidance=float(hp.guidance), is_negative_prompt=False)

    @torch.no_grad()
    def collect_rollout_and_rewards(
        self, data_batch: Dict[str, torch.Tensor], rollout_seed_offset: int = 0
    ) -> OpdRolloutSamples:
        hp = self._get_grpo_params()
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)

        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            if self.text_encoder is None:
                raise RuntimeError("text_encoder is None but text_encoder_config.compute_online=True")
            text_embeddings = self.text_encoder.compute_text_embeddings_online(
                data_batch, self.input_caption_key
            )  # [B0, L, D]
            data_batch["t5_text_embeddings"] = text_embeddings.to(**self.tensor_kwargs)  # [B0, L, D]
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device=self.tensor_kwargs["device"]
            )  # [B0, L]

        data_batch = self._maybe_repeat_batch_for_group(data_batch, hp.num_generations)  # B = B0 * G
        self._ensure_action_dtype_inplace(data_batch)
        self._ensure_rollout_input_dtypes_inplace(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        b = data_batch[input_key].shape[0]  # B = B0 * G
        _T, _H, _W = data_batch[input_key].shape[-3:]  # video/image frames, height, width

        rollout_data_batch = data_batch
        if not is_image_batch and _T > 1:
            rollout_data_batch = dict(data_batch)
            video = data_batch[input_key]  # [B, C, T, H, W]
            rollout_data_batch[input_key] = torch.cat(
                [video[:, :, :1], torch.full_like(video[:, :, 1:], -1.0)], dim=2
            ).contiguous()  # [B, C, T, H, W]
            rollout_data_batch["num_conditional_frames"] = 1

        state_shape = (
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        )  # [C, T_lat, H_lat, W_lat]

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        dp_rank = _get_data_parallel_rank_for_seed()
        generator.manual_seed(int(hp.seed) + int(rollout_seed_offset) + int(dp_rank))
        if bool(hp.init_same_noise) and int(hp.num_generations) > 1:
            assert b % int(hp.num_generations) == 0, "Batch size must be divisible by num_generations."
            n_groups = b // int(hp.num_generations)  # B0
            base_noise = torch.randn(
                (n_groups,) + state_shape,
                device=self.tensor_kwargs["device"],
                dtype=self.tensor_kwargs["dtype"],
                generator=generator,
            )  # [B0, C, T_lat, H_lat, W_lat]
            init_noise = base_noise.repeat_interleave(int(hp.num_generations), dim=0)  # [B, C, T_lat, H_lat, W_lat]
        else:
            init_noise = torch.randn(
                (b,) + state_shape,
                device=self.tensor_kwargs["device"],
                dtype=self.tensor_kwargs["dtype"],
                generator=generator,
            )  # [B, C, T_lat, H_lat, W_lat]

        velocity_fn = self.get_velocity_fn_from_batch(
            rollout_data_batch, guidance=float(hp.guidance), is_negative_prompt=False
        )
        teacher_velocity_fn = self._get_opd_teacher_velocity_fn(rollout_data_batch, hp)

        self.sample_scheduler.set_timesteps(
            hp.num_steps,
            device=self.tensor_kwargs["device"],
            shift=float(hp.shift),
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )
        sigmas = self.sample_scheduler.sigmas.to(device=self.tensor_kwargs["device"], dtype=torch.float32)  # [S+1]
        timesteps = self.sample_scheduler.timesteps.to(device=self.tensor_kwargs["device"])  # [S]

        latents = init_noise  # [B, C, T_lat, H_lat, W_lat]
        init_noise_local = init_noise  # [B, C, T_lat, H_lat, W_lat]
        if self.net.is_context_parallel_enabled:
            cp_group = self.get_context_parallel_group()
            init_noise_local = broadcast_split_tensor(init_noise_local, seq_dim=2, process_group=cp_group)
            latents = broadcast_split_tensor(latents, seq_dim=2, process_group=cp_group)

        all_latents = []
        all_next_latents = []
        all_old_log_probs = []
        rollout_opd_kls = []
        rollout_opd_advantages = []

        for i, t_tok in enumerate(timesteps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            t_B_1 = torch.stack([t_tok]).unsqueeze(0).repeat(b, 1)  # [B, 1]

            v_pred = velocity_fn(init_noise_local, latents, t_B_1)  # [B, C, T_lat, H_lat, W_lat]
            eps = torch.randn(
                latents.shape, dtype=torch.float32, device=latents.device, generator=generator
            )  # [B, C, T_lat, H_lat, W_lat]
            step_out = grpo_sde_step(
                latents=latents,
                velocity=v_pred,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=float(hp.eta),
                noise=eps,
                sigma_max=sigmas[1],
                sigma_dependent_eta=bool(hp.sigma_dependent_eta),
                fixed_next_latents=None,
            )

            v_teacher = teacher_velocity_fn(init_noise_local, latents, t_B_1)  # [B, C, T_lat, H_lat, W_lat]
            teacher_step_out = grpo_sde_step(
                latents=latents,
                velocity=v_teacher,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=float(hp.eta),
                noise=torch.zeros_like(eps),
                sigma_max=sigmas[1],
                sigma_dependent_eta=bool(hp.sigma_dependent_eta),
                fixed_next_latents=step_out.next_latents,
            )
            reduce_dims = tuple(range(1, step_out.transition_mean.ndim))
            opd_kl = (
                (step_out.transition_mean.detach() - teacher_step_out.transition_mean.detach()) ** 2
                / (2.0 * step_out.transition_std.detach() ** 2)
            ).mean(dim=reduce_dims)  # [B]
            rollout_opd_kls.append(opd_kl.to(torch.float32))  # each entry: [B]
            rollout_opd_advantages.append((-float(hp.opd_kl_scale) * opd_kl).to(torch.float32))  # each entry: [B]

            next_latents = step_out.next_latents  # [B, C, T_lat, H_lat, W_lat]
            all_latents.append(latents)  # each entry: [B, C, T_lat, H_lat, W_lat]
            all_next_latents.append(next_latents)  # each entry: [B, C, T_lat, H_lat, W_lat]
            all_old_log_probs.append(step_out.log_prob.to(torch.float32))  # each entry: [B]
            latents = next_latents

        latents_s = torch.stack(all_latents, dim=1)  # [B, S, C, T_lat, H_lat, W_lat]
        next_latents_s = torch.stack(all_next_latents, dim=1)  # [B, S, C, T_lat, H_lat, W_lat]
        old_log_probs_s = torch.stack(all_old_log_probs, dim=1)  # [B, S]
        rollout_opd_advantages_s = torch.stack(rollout_opd_advantages, dim=1)  # [B, S]
        rollout_opd_kls_s = torch.stack(rollout_opd_kls, dim=1)  # [B, S]

        rewards = rollout_opd_advantages_s.mean(dim=1).to(
            device=self.tensor_kwargs["device"], dtype=torch.float32
        )  # [B]
        reward_metrics = {"opd_kl": rollout_opd_kls_s.mean(dim=1).detach()}  # [B]
        advantages = rewards  # [B], logging-compatible summary; OPD loss recomputes dense advantages

        return OpdRolloutSamples(
            latents=latents_s[:, :-1].detach(),  # [B, S-1, C, T_lat, H_lat, W_lat]
            next_latents=next_latents_s[:, :-1].detach(),  # [B, S-1, C, T_lat, H_lat, W_lat]
            old_log_probs=old_log_probs_s[:, :-1].detach(),  # [B, S-1]
            timestep_tokens=timesteps[:-1].detach(),  # [S-1]
            sigmas=sigmas.detach(),  # [S+1]
            init_noise=init_noise_local.detach(),  # [B, C, T_lat, H_lat, W_lat]
            rewards=rewards.detach(),
            reward_metrics={k: v.detach() for k, v in reward_metrics.items()},
            advantages=advantages.detach(),
            velocity_fn=velocity_fn,
            is_image_batch=is_image_batch,
            teacher_velocity_fn=teacher_velocity_fn,
        )

    def compute_grpo_loss(
        self,
        samples: GrpoRolloutSamples,
        update_seed: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        hp = self._get_grpo_params()
        if not isinstance(samples, OpdRolloutSamples) or samples.teacher_velocity_fn is None:
            raise TypeError("OPD loss expects OpdRolloutSamples with `teacher_velocity_fn`.")

        S = int(samples.timestep_tokens.shape[0])  # rollout train transitions
        train_T = max(1, int(S * hp.timestep_fraction))
        device = samples.latents.device
        B = int(samples.latents.shape[0])

        gen = torch.Generator(device=device)
        gen.manual_seed(int(hp.seed) + int(update_seed))
        perms = torch.stack([torch.randperm(S, device=device, generator=gen) for _ in range(B)], dim=0)  # [B, S]
        idx = perms[:, :train_T]  # [B, train_T]
        arange_b = torch.arange(B, device=device)  # [B]

        latents_i = samples.latents[arange_b[:, None], idx]  # [B, train_T, C, T_lat, H_lat, W_lat]
        next_latents_i = samples.next_latents[arange_b[:, None], idx]  # [B, train_T, C, T_lat, H_lat, W_lat]
        old_log_probs_i = samples.old_log_probs[arange_b[:, None], idx].to(torch.float32)  # [B, train_T]

        new_log_probs_list = []
        advantages_list = []
        opd_kls_list = []
        for j in range(train_T):
            idx_b = idx[:, j]  # [B]
            t_tok_b = samples.timestep_tokens.index_select(dim=0, index=idx_b)  # [B]
            sigma_b = samples.sigmas.index_select(dim=0, index=idx_b)  # [B]
            sigma_next_b = samples.sigmas.index_select(dim=0, index=(idx_b + 1))  # [B]
            t_B_1 = t_tok_b.view(B, 1)  # [B, 1]

            v_pred = samples.velocity_fn(
                samples.init_noise, latents_i[:, j], t_B_1
            )  # [B, C, T_lat, H_lat, W_lat]
            step_out = grpo_sde_step(
                latents=latents_i[:, j],  # [B, C, T_lat, H_lat, W_lat]
                velocity=v_pred,
                sigma=sigma_b,
                sigma_next=sigma_next_b,
                eta=float(hp.eta),
                noise=torch.zeros_like(latents_i[:, j]),
                sigma_max=samples.sigmas[1],
                sigma_dependent_eta=bool(hp.sigma_dependent_eta),
                fixed_next_latents=next_latents_i[:, j],
            )
            new_log_probs_list.append(step_out.log_prob.to(torch.float32))  # each entry: [B]

            with torch.no_grad():
                v_teacher = samples.teacher_velocity_fn(
                    samples.init_noise, latents_i[:, j], t_B_1
                )  # [B, C, T_lat, H_lat, W_lat]
                teacher_step_out = grpo_sde_step(
                    latents=latents_i[:, j],  # [B, C, T_lat, H_lat, W_lat]
                    velocity=v_teacher,
                    sigma=sigma_b,
                    sigma_next=sigma_next_b,
                    eta=float(hp.eta),
                    noise=torch.zeros_like(latents_i[:, j]),
                    sigma_max=samples.sigmas[1],
                    sigma_dependent_eta=bool(hp.sigma_dependent_eta),
                    fixed_next_latents=next_latents_i[:, j],
                )

            reduce_dims = tuple(range(1, step_out.transition_mean.ndim))
            opd_kl = (
                (step_out.transition_mean - teacher_step_out.transition_mean.detach()) ** 2
                / (2.0 * step_out.transition_std ** 2)
            ).mean(dim=reduce_dims)  # [B]
            opd_kls_list.append(opd_kl.to(torch.float32))  # each entry: [B]
            advantages_list.append((-float(hp.opd_kl_scale) * opd_kl).to(torch.float32))  # each entry: [B]

        new_log_probs = torch.stack(new_log_probs_list, dim=1)  # [B, train_T]
        ratio = torch.exp(new_log_probs - old_log_probs_i)  # [B, train_T]
        adv = torch.stack(advantages_list, dim=1)  # [B, train_T]
        opd_kls = torch.stack(opd_kls_list, dim=1)  # [B, train_T]

        unclipped = -adv * ratio  # [B, train_T]
        clipped = -adv * torch.clamp(ratio, 1.0 - hp.clip_range, 1.0 + hp.clip_range)  # [B, train_T]
        loss = torch.mean(torch.maximum(unclipped, clipped))

        approx_kl = torch.mean(old_log_probs_i - new_log_probs).detach()
        clip_frac = torch.mean(((ratio - 1.0).abs() > hp.clip_range).to(torch.float32)).detach()
        output_batch: Dict[str, torch.Tensor] = {
            "grpo_loss": loss.detach(),
            "reward_mean": samples.rewards.mean().detach(),
            "reward_std": samples.rewards.std().detach(),
            "adv_mean": adv.mean().detach(),
            "adv_std": adv.std().detach(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "opd_train_kl_mean": opd_kls.mean().detach(),
            "opd_train_kl_std": opd_kls.std().detach(),
        }
        for name, metric in samples.reward_metrics.items():
            output_batch[self._reward_metric_log_key(name, "mean")] = metric.mean().detach()
            output_batch[self._reward_metric_log_key(name, "std")] = metric.std().detach()
        output_batch["edm_loss"] = output_batch["grpo_loss"]
        return output_batch, loss
