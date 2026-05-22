"""
Multi-chunk GRPO Action-conditioned Video2World Rectified Flow model.

This variant keeps the single-chunk policy/loss representation used by
`ActionVideo2WorldModelRectifiedFlowGRPO`, but collects rollout samples by
autoregressively generating several native action chunks and flattening the
chunk dimension into the batch dimension.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_grpo_model import (
    ActionVideo2WorldModelRectifiedFlowGRPO,
    ActionVideo2WorldModelRectifiedFlowGRPOConfig,
    GrpoRolloutSamples,
    _get_data_parallel_rank_for_seed,
)
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import IS_PREPROCESSED_KEY
from cosmos_predict2._src.predict2.rl.grpo_sde_sampler import grpo_sde_step
from cosmos_predict2._src.predict2.rl.reward import RewardInput


class ActionVideo2WorldModelRectifiedFlowMultiChunkGRPO(ActionVideo2WorldModelRectifiedFlowGRPO):
    """
    GRPO model that rolls out multiple native 12-action chunks autoregressively.

    The collected chunk trajectories are flattened to `[K * B, S, ...]`, so the
    inherited `compute_grpo_loss()` can train on them as ordinary single-chunk
    GRPO samples.

    Shape comments use `B0` for the incoming dataloader batch size, `G` for
    `num_generations`, and `B = B0 * G` after group repeat.
    """

    config: ActionVideo2WorldModelRectifiedFlowGRPOConfig

    def _get_action_chunk_size(self) -> int:
        grpo_cfg = dict(self.config.grpo or {})
        chunk_size = grpo_cfg.get("action_chunk_size", None)
        if chunk_size is None:
            chunk_size = getattr(self.config.net, "num_action_per_chunk", 12)
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"action_chunk_size must be positive, got {chunk_size}")
        return chunk_size

    def _get_expected_num_rollout_chunks(self) -> int | None:
        """
        从实验配置里面拿 num_rollout_chunks
        """
        grpo_cfg = dict(self.config.grpo or {})
        value = grpo_cfg.get("num_rollout_chunks", None)
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"num_rollout_chunks must be positive when set, got {value}")
        return value

    def _infer_num_rollout_chunks(
        self,
        data_batch: Dict[str, Any],
        *,
        input_key: str,
        action_chunk_size: int,
    ) -> int:
        """
        根据当前 batch 帧数以及 chunk_size 推断 num_rollout_batches
        """
        if "action" not in data_batch or not torch.is_tensor(data_batch["action"]):
            raise KeyError("Multi-chunk GRPO requires data_batch['action'] as a tensor.")

        total_actions = int(data_batch["action"].shape[1])  # data_batch["action"]: [B, K*A, D], B=B0*G
        if total_actions % action_chunk_size != 0:
            raise ValueError(
                f"Long action length must be divisible by action_chunk_size. "
                f"Got total_actions={total_actions}, action_chunk_size={action_chunk_size}."
            )

        num_chunks = total_actions // action_chunk_size
        expected_chunks = self._get_expected_num_rollout_chunks()
        if expected_chunks is not None and expected_chunks != num_chunks:
            raise ValueError(
                f"Configured grpo.num_rollout_chunks={expected_chunks} does not match "
                f"the inferred value from action length: {num_chunks}."
            )

        expected_frames = 1 + num_chunks * action_chunk_size
        actual_frames = int(data_batch[input_key].shape[2])  # data_batch[input_key]: [B, C, 1+K*A, H, W], B=B0*G
        if actual_frames != expected_frames:
            raise ValueError(
                f"Long video/action lengths are inconsistent. Expected video T={expected_frames} "
                f"from action length, got T={actual_frames}."
            )
        return num_chunks

    def _copy_conditioning_fields_for_chunk(
        self,
        data_batch: Dict[str, Any],
        *,
        batch_size: int,
        input_key: str,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in data_batch.items():
            if key in {input_key, "action", IS_PREPROCESSED_KEY, "num_conditional_frames", "num_frames"}:
                continue
            if torch.is_tensor(value):
                out[key] = value
            elif isinstance(value, list):
                out[key] = list(value)
            elif isinstance(value, tuple):
                out[key] = tuple(value)
            else:
                out[key] = value

        out["num_conditional_frames"] = 1
        out["num_frames"] = torch.full(  # [B], B=B0*G
            (batch_size,),
            fill_value=int(self._get_action_chunk_size() + 1),
            device=self.tensor_kwargs["device"],
            dtype=torch.int64,
        )
        out[IS_PREPROCESSED_KEY] = True
        return out

    def _make_chunk_batch(
        self,
        data_batch: Dict[str, Any],
        *,
        input_key: str,
        condition_frame: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> Dict[str, Any]:
        b, c, _, h, w = condition_frame.shape  # condition_frame: [B, C, 1, H, W], B=B0*G
        action_chunk_size = int(action_chunk.shape[1])  # action_chunk: [B, A, D], B=B0*G
        chunk_batch = self._copy_conditioning_fields_for_chunk(
            data_batch, batch_size=b, input_key=input_key
        )
        # Future frames are placeholders. Only the first latent frame is used as video condition.
        future_frames = torch.zeros(  # [B, C, A, H, W], B=B0*G
            b,
            c,
            action_chunk_size,
            h,
            w,
            device=condition_frame.device,
            dtype=condition_frame.dtype,
        )
        chunk_batch[input_key] = torch.cat([condition_frame, future_frames], dim=2).contiguous()  # [B, C, 1+A, H, W], B=B0*G
        chunk_batch["action"] = action_chunk.contiguous()  # [B, A, D], B=B0*G
        return chunk_batch

    def _concat_chunk_batches(self, chunk_batches: list[Dict[str, Any]], *, batch_size: int) -> Dict[str, Any]:
        assert len(chunk_batches) > 0
        keys = set().union(*[b.keys() for b in chunk_batches])
        out: Dict[str, Any] = {}
        for key in keys:
            values = [b[key] for b in chunk_batches if key in b]
            first = values[0]
            if torch.is_tensor(first):
                if first.ndim > 0 and int(first.shape[0]) == batch_size:
                    out[key] = torch.cat(values, dim=0)  # [K*B, ...]
                else:
                    out[key] = first
            elif isinstance(first, list):
                merged = []
                for value in values:
                    merged.extend(value)
                out[key] = merged
            elif isinstance(first, tuple):
                merged_tuple = []
                for value in values:
                    merged_tuple.extend(list(value))
                out[key] = tuple(merged_tuple)
            else:
                out[key] = first
        return out

    @staticmethod
    def _repeat_chunk_major(x: torch.Tensor, num_chunks: int) -> torch.Tensor:
        return x.repeat(int(num_chunks))  # [K*B], B=B0*G

    @torch.no_grad()
    def collect_rollout_and_rewards(
        self, data_batch: Dict[str, torch.Tensor], rollout_seed_offset: int = 0
    ) -> GrpoRolloutSamples:
        hp = self._get_grpo_params()

        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)

        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            if self.text_encoder is None:
                raise RuntimeError("text_encoder is None but text_encoder_config.compute_online=True")
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)  # [B0, L, D]
            data_batch["t5_text_embeddings"] = text_embeddings.to(**self.tensor_kwargs)  # [B0, L, D]
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0],
                text_embeddings.shape[1],
                device=self.tensor_kwargs["device"],
            )  # [B0, L]

        data_batch = self._maybe_repeat_batch_for_group(data_batch, hp.num_generations)  # [B, ...], B=B0*G
        self._ensure_action_dtype_inplace(data_batch)
        self._ensure_rollout_input_dtypes_inplace(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        if is_image_batch:
            raise ValueError("Multi-chunk GRPO expects video batches, not image batches.")
        input_key = self.input_data_key  # 一般是 "video"
        b = int(data_batch[input_key].shape[0])  # data_batch[input_key]: [B, C, 1+K*A, H, W], B=B0*G
        action_chunk_size = self._get_action_chunk_size()
        num_chunks = self._infer_num_rollout_chunks(
            data_batch, input_key=input_key, action_chunk_size=action_chunk_size
        )

        _, _, _, height, width = data_batch[input_key].shape  # [B, C, 1+K*A, H, W], B=B0*G
        state_shape = (
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(action_chunk_size + 1),
            height // self.tokenizer.spatial_compression_factor,
            width // self.tokenizer.spatial_compression_factor,
        )

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        dp_rank = _get_data_parallel_rank_for_seed()
        generator.manual_seed(int(hp.seed) + int(rollout_seed_offset) + int(dp_rank))

        self.sample_scheduler.set_timesteps(
            hp.num_steps,
            device=self.tensor_kwargs["device"],
            shift=float(hp.shift),
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )
        sigmas = self.sample_scheduler.sigmas.to(device=self.tensor_kwargs["device"], dtype=torch.float32)  # [S+1]
        timesteps = self.sample_scheduler.timesteps.to(device=self.tensor_kwargs["device"])  # [S]

        chunk_batches: list[Dict[str, Any]] = []
        chunk_latents: list[torch.Tensor] = []
        chunk_next_latents: list[torch.Tensor] = []
        chunk_old_log_probs: list[torch.Tensor] = []
        chunk_init_noise: list[torch.Tensor] = []
        pred_video_chunks: list[torch.Tensor] = []

        condition_frame = data_batch[input_key][:, :, :1].contiguous()  # [B, C, 1, H, W], B=B0*G
        for chunk_idx in range(num_chunks):
            action_start = chunk_idx * action_chunk_size
            action_end = action_start + action_chunk_size
            action_chunk = data_batch["action"][:, action_start:action_end]  # [B, A, D], B=B0*G
            chunk_batch = self._make_chunk_batch(
                data_batch,
                input_key=input_key,
                condition_frame=condition_frame,
                action_chunk=action_chunk,
            )
            # chunk_batch:
            # {
            #     "video": [B, C, 1+A, H, W],  # 除了第一帧条件帧以外其他帧是 0
            #     "action": [B, A, D],
            #     "num_conditional_frames": 1,
            #     "num_frames": [B],
            #     "t5_text_embeddings": ...,
            #     "t5_text_mask": ...,
            #     "fps": ...,
            #     "padding_mask": ...,
            #     ...
            # }

            chunk_batches.append(chunk_batch)

            if bool(hp.init_same_noise) and int(hp.num_generations) > 1:
                assert b % int(hp.num_generations) == 0, "Batch size must be divisible by num_generations."
                n_groups = b // int(hp.num_generations)  # B0
                base_noise = torch.randn(  # [B0, C, T_lat, H_lat, W_lat]
                    (n_groups,) + state_shape,
                    device=self.tensor_kwargs["device"],
                    dtype=self.tensor_kwargs["dtype"],
                    generator=generator,
                )
                init_noise = base_noise.repeat_interleave(int(hp.num_generations), dim=0)  # [B, C, T_lat, H_lat, W_lat], B=B0*G
            else:
                init_noise = torch.randn(  # [B, C, T_lat, H_lat, W_lat], B=B0*G
                    (b,) + state_shape,
                    device=self.tensor_kwargs["device"],
                    dtype=self.tensor_kwargs["dtype"],
                    generator=generator,
                )

            velocity_fn = self.get_velocity_fn_from_batch(
                chunk_batch, guidance=float(hp.guidance), is_negative_prompt=False
            )
            latents = init_noise  # [B, C, T_lat, H_lat, W_lat], B=B0*G
            init_noise_local = init_noise  # [B, C, T_lat, H_lat, W_lat], B=B0*G
            if self.net.is_context_parallel_enabled:
                cp_group = self.get_context_parallel_group()
                init_noise_local = broadcast_split_tensor(init_noise_local, seq_dim=2, process_group=cp_group)  # [B, C, T_lat/CP, H_lat, W_lat], B=B0*G
                latents = broadcast_split_tensor(latents, seq_dim=2, process_group=cp_group)  # [B, C, T_lat/CP, H_lat, W_lat], B=B0*G

            all_latents = []
            all_next_latents = []
            all_old_log_probs = []
            for i, t_tok in enumerate(timesteps):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                t_B_1 = torch.stack([t_tok]).unsqueeze(0).repeat(b, 1)  # [B, 1], B=B0*G
                v_pred = velocity_fn(init_noise_local, latents, t_B_1)  # [B, C, T_lat, H_lat, W_lat], B=B0*G
                eps = torch.randn(latents.shape, dtype=torch.float32, device=latents.device, generator=generator)  # [B, C, T_lat, H_lat, W_lat], B=B0*G
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

                next_latents = step_out.next_latents  # [B, C, T_lat, H_lat, W_lat], B=B0*G
                all_latents.append(latents)  # list[S] of [B, C, T_lat, H_lat, W_lat], B=B0*G
                all_next_latents.append(next_latents)  # list[S] of [B, C, T_lat, H_lat, W_lat], B=B0*G
                all_old_log_probs.append(step_out.log_prob.to(torch.float32))  # list[S] of [B], B=B0*G
                latents = next_latents

            latents_s = torch.stack(all_latents, dim=1)  # [B, S, C, T_lat, H_lat, W_lat], B=B0*G
            next_latents_s = torch.stack(all_next_latents, dim=1)  # [B, S, C, T_lat, H_lat, W_lat], B=B0*G
            old_log_probs_s = torch.stack(all_old_log_probs, dim=1)  # [B, S], B=B0*G

            chunk_latents.append(latents_s[:, :-1].detach())  # list[K] of [B, S-1, C, T_lat, H_lat, W_lat], B=B0*G
            chunk_next_latents.append(next_latents_s[:, :-1].detach())  # list[K] of [B, S-1, C, T_lat, H_lat, W_lat], B=B0*G
            chunk_old_log_probs.append(old_log_probs_s[:, :-1].detach())  # list[K] of [B, S-1], B=B0*G
            chunk_init_noise.append(init_noise_local.detach())  # list[K] of [B, C, T_lat, H_lat, W_lat], B=B0*G

            final_latents = latents  # [B, C, T_lat, H_lat, W_lat], B=B0*G
            if self.net.is_context_parallel_enabled:
                final_latents = cat_outputs_cp(final_latents, seq_dim=2, cp_group=self.get_context_parallel_group())  # [B, C, T_lat, H_lat, W_lat], B=B0*G
            pred_video_pixels = self.decode(final_latents.to(torch.float32)).detach()  # [B, C, 1+A, H, W], B=B0*G
            pred_video_chunks.append(pred_video_pixels)  # list[K] of [B, C, 1+A, H, W], B=B0*G
            condition_frame = pred_video_pixels[:, :, -1:].detach()  # [B, C, 1, H, W], B=B0*G

        pred_full_video = torch.cat(  # [B, C, 1+K*A, H, W], B=B0*G
            [pred_video_chunks[0]] + [chunk[:, :, 1:] for chunk in pred_video_chunks[1:]],
            dim=2,
        ).contiguous()

        gt_video = data_batch[input_key]  # [B, C, 1+K*A, H, W], B=B0*G
        if torch.is_tensor(gt_video) and gt_video.device != pred_full_video.device:
            gt_video = gt_video.to(device=pred_full_video.device)

        assert self._reward_model is not None
        rewards = self._reward_model(
            RewardInput(
                video=pred_full_video,  # [B, C, 1+K*A, H, W], B=B0*G
                text=None,
                action=data_batch.get("action", None),
                metadata={
                    "is_image_batch": False,
                    "gt_video": gt_video,  # [B, C, 1+K*A, H, W], B=B0*G
                    "num_rollout_chunks": torch.tensor(num_chunks, device=pred_full_video.device),
                },
            )
        ).to(device=self.tensor_kwargs["device"], dtype=torch.float32)  # [B], B=B0*G
        reward_metrics = self._extract_reward_metrics(batch_size=int(rewards.shape[0]))  # each [B], B=B0*G
        advantages, advantage_metrics = self._compute_rollout_advantages(rewards, reward_metrics, hp)  # [B], each [B], B=B0*G

        flat_chunk_batch = self._concat_chunk_batches(chunk_batches, batch_size=b)  # tensor fields [K*B, ...], B=B0*G
        self._ensure_action_dtype_inplace(flat_chunk_batch)
        self._ensure_rollout_input_dtypes_inplace(flat_chunk_batch)
        flat_velocity_fn = self.get_velocity_fn_from_batch(
            flat_chunk_batch, guidance=float(hp.guidance), is_negative_prompt=False
        )

        train_timesteps = timesteps[:-1]  # [S-1]
        flat_rewards = self._repeat_chunk_major(rewards.detach(), num_chunks)  # [K*B], B=B0*G
        flat_advantages = self._repeat_chunk_major(advantages.detach(), num_chunks)  # [K*B], B=B0*G
        flat_reward_metrics = {
            k: self._repeat_chunk_major(v.detach(), num_chunks) for k, v in reward_metrics.items()  # each [K*B], B=B0*G
        }
        flat_advantage_metrics = {
            k: self._repeat_chunk_major(v.detach(), num_chunks) for k, v in advantage_metrics.items()  # each [K*B], B=B0*G
        }

        log.info(
            f"Collected multi-chunk GRPO rollout: chunks={num_chunks}, "
            f"flat_batch={b * num_chunks}, reward_mean={rewards.mean().item():.4f}"
        )

        return GrpoRolloutSamples(
            latents=torch.cat(chunk_latents, dim=0),  # [K*B, S-1, C, T_lat, H_lat, W_lat], B=B0*G
            next_latents=torch.cat(chunk_next_latents, dim=0),  # [K*B, S-1, C, T_lat, H_lat, W_lat], B=B0*G
            old_log_probs=torch.cat(chunk_old_log_probs, dim=0),  # [K*B, S-1], B=B0*G
            timestep_tokens=train_timesteps.detach(),
            sigmas=sigmas.detach(),
            init_noise=torch.cat(chunk_init_noise, dim=0),  # [K*B, C, T_lat, H_lat, W_lat], B=B0*G
            rewards=flat_rewards,
            reward_metrics=flat_reward_metrics,
            advantages=flat_advantages,
            velocity_fn=flat_velocity_fn,
            is_image_batch=False,
            advantage_metrics=flat_advantage_metrics,
        )
