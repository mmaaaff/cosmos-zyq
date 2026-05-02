"""
GRPO-enabled Action-conditioned Video2World Rectified Flow model.

Annotation:
- This model integrates DanceGRPO-style online rollout and clipped policy optimization into COSMOS by overriding
  `training_step`, while leaving the high-level training loop (`ImaginaireTrainer`) unchanged.
- Sigma/timestep scheduling is unified with UniPC's scheduler (`FlowUniPCMultistepScheduler.set_timesteps`) output.
"""

# 说明（重要）：
# - 本文件中的 GRPO rollout 只优化“单个模型原生时间窗口”的生成，不负责长视频的外层多 chunk 自回归 rollout（这是 inference 阶段会干的事情）。
# - 对当前 action-conditioned 配置，这个原生窗口通常就是 12 个 action -> 13 帧视频。
# - 这里看起来像“整段 action 一次性条件化并整段去噪”，是因为模型本来就是按一个 action chunk 对应一个
#   video chunk 来设计的；这里的 rollout 轨迹主要是 diffusion 时间步上的 latent trajectory。
# - 训练样本是否分 chunk，主要由上游 dataloader 决定：Dataset_3D 会先把长轨迹切成固定窗口，每个 sample
#   只包含一个 sequence_length = 1 + num_action_per_chunk 的片段；因此这里并不是在对任意长整段视频直接生成。
# - `examples/action_conditioned.py` 里的长视频推理会在模型外层再套一层 chunk loop，把多个 12-action 窗口串起来。
#   当前文件没有显式建模那层长时闭环过程，所以它更适合优化单个 13-frame chunk 的质量，而不是直接优化长时 rollout。

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, Optional, Tuple

import attrs
import torch
import torch.distributed as dist

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.rl.grpo_sde_sampler import grpo_sde_step
from cosmos_predict2._src.predict2.rl.reward import CompositeRewardModel, RewardInput


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


# ---------- 跟 Megatron 多 DP 子组相关的设置，实际上不一定用到 ----------
# Megatron-core is optional in this repo; we mirror the pattern used in `imaginaire.trainer`.
try:
    from megatron.core import parallel_state  # type: ignore

    _USE_MEGATRON = True
except Exception:  # pragma: no cover
    parallel_state = None  # type: ignore
    _USE_MEGATRON = False


def _get_data_parallel_group_with_cp():
    """
    Get the same data-parallel process group used by the DDP wrapper in this codebase.

    Why:
    - In Megatron setups (TP/PP/CP), the global WORLD group can contain multiple DP subgroups.
    - DDP is created with `parallel_state.get_data_parallel_group(with_context_parallel=True)`; we must use the
      *same* group for all_gather/reductions to keep semantics consistent and avoid cross-group mixing.
    """
    if _USE_MEGATRON and parallel_state is not None and parallel_state.is_initialized():
        return parallel_state.get_data_parallel_group(with_context_parallel=True)
    return None


def _get_data_parallel_rank_for_seed() -> int:
    """
    DP rank used for randomness control.

    We intentionally use Megatron's *data-parallel* rank (without context-parallel) when available so that:
    - Different DP replicas get different rollout noise.
    - Ranks within the same context-parallel group (CP shards of the same sample) share the same seed.
    """
    if _USE_MEGATRON and parallel_state is not None and parallel_state.is_initialized():
        try:
            return int(parallel_state.get_data_parallel_rank())
        except TypeError:
            # Some versions require explicit kwarg; fall back to the simplest call.
            return int(parallel_state.get_data_parallel_rank())
    if _dist_is_initialized():
        return int(dist.get_rank())
    return 0
# -------------------------------------------------------------


def _all_gather_concat_1d(x: torch.Tensor, group=None) -> torch.Tensor:
    """
    All-gather a 1D tensor across data-parallel ranks and concatenate on dim=0.
    """

    if not _dist_is_initialized():
        return x
    world_size = dist.get_world_size(group=group)
    chunks = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x.contiguous(), group=group)
    return torch.cat(chunks, dim=0)


@dataclass
class _GrpoHyperParams:
    # Rollout
    num_steps: int = 16
    shift: float = 5.0
    eta: float = 0.0
    guidance: float = 3.0
    seed: int = 1
    timestep_fraction: float = 1.0
    # 如果启用 group（num_generations>1），是否让同一个 prompt 的一组样本共享相同的 init_noise
    # 参考 Dance-GRPO/Flux 代码中的 `--init_same_noise`：提升训练稳定性。
    init_same_noise: bool = False
    # Outer loop: rollout batch formation / inner loop: multi-updates
    rollout_num_batches: int = 1
    num_updates: int = 4
    # Advantage
    use_group_adv: bool = True
    num_generations: int = 4
    adv_clip_max: float = 5.0
    # PPO/GRPO clipping
    clip_range: float = 1e-4


@dataclass
class GrpoRolloutSamples:
    """
    Cached rollout samples for GRPO multi-update training.

    Annotation:
    - `velocity_fn` is created once per rollout by `get_velocity_fn_from_batch` to keep conditioning fixed.
      It will use the current model parameters when invoked (so `new_log_probs` change after each update).
    """

    latents: torch.Tensor  # [B, S, C, T, H, W] (T may be local under context parallel)
    next_latents: torch.Tensor  # [B, S, C, T, H, W]
    old_log_probs: torch.Tensor  # [B, S]
    timestep_tokens: torch.Tensor  # [S] int64
    sigmas: torch.Tensor  # [S+1] float32
    init_noise: torch.Tensor  # [B, C, T, H, W] float32 (may be local under CP)
    rewards: torch.Tensor  # [B]
    reward_metrics: Dict[str, torch.Tensor]
    advantages: torch.Tensor  # [B]
    velocity_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    is_image_batch: bool
    advantage_metrics: Dict[str, torch.Tensor] = field(default_factory=dict)


@attrs.define(slots=False)
class ActionVideo2WorldModelRectifiedFlowGRPOConfig(Video2WorldModelRectifiedFlowConfig):
    """
    Extends the base action-conditioned rectified-flow config with GRPO-specific knobs.

    Annotation:
    - `grpo` stays as a plain dictionary for override friendliness.
    - `reward` is a LazyDict so Hydra can select/instantiate reward implementations like other subcomponents.
    """

    grpo: Dict[str, Any] = attrs.field(factory=dict)
    reward: LazyDict | None = None


class ActionVideo2WorldModelRectifiedFlowGRPO(ActionVideo2WorldModelRectifiedFlow):
    """
    GRPO training model.
    """

    config: ActionVideo2WorldModelRectifiedFlowGRPOConfig

    def __init__(self, config: ActionVideo2WorldModelRectifiedFlowGRPOConfig):
        super().__init__(config)
        if config.reward is None:
            raise ValueError(
                "GRPO reward config is required. Provide it via Hydra group override `/reward=...` "
                "or by setting `model.config.reward`."
            )
        self._reward_model = lazy_instantiate(config.reward)

    # ----------------------------- utilities -----------------------------
    def _get_grpo_params(self) -> _GrpoHyperParams:
        d = dict(self.config.grpo or {})
        hp = _GrpoHyperParams()
        for k, v in d.items():
            if hasattr(hp, k):
                setattr(hp, k, v)
        # Defensive clamps
        hp.num_steps = int(hp.num_steps)
        # We drop the final denoising transition from the GRPO loss to match DanceGRPO,
        # so at least two rollout steps are needed to leave one trainable transition.
        hp.num_steps = max(hp.num_steps, 2)
        hp.rollout_num_batches = int(hp.rollout_num_batches)
        hp.rollout_num_batches = max(hp.rollout_num_batches, 1)
        hp.num_updates = int(hp.num_updates)
        hp.num_updates = max(hp.num_updates, 1)
        hp.num_generations = int(hp.num_generations)
        hp.num_generations = max(hp.num_generations, 1)
        hp.timestep_fraction = float(hp.timestep_fraction)
        hp.timestep_fraction = min(max(hp.timestep_fraction, 0.0), 1.0)
        return hp

    def unload_reward_model(self) -> None:
        if hasattr(self._reward_model, "unload"):
            self._reward_model.unload(clear_cuda_cache=True)

    def _extract_reward_metrics(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self._reward_model, "get_last_metrics"):
            return {}

        raw_metrics = self._reward_model.get_last_metrics()
        metrics: Dict[str, torch.Tensor] = {}
        for name, value in raw_metrics.items():
            if not torch.is_tensor(value):
                log.warning(f"Skipping reward metric '{name}' because it is not a tensor: {type(value)}")
                continue

            value = value.detach().to(device=self.tensor_kwargs["device"], dtype=torch.float32)
            if value.ndim != 1:
                log.warning(
                    f"Skipping reward metric '{name}' because it must have shape [B], got {tuple(value.shape)}"
                )
                continue
            if value.shape[0] != batch_size:
                log.warning(
                    f"Skipping reward metric '{name}' because batch size {value.shape[0]} != expected {batch_size}"
                )
                continue

            metrics[name] = value
        return metrics

    @staticmethod
    def _safe_metric_name(name: str) -> str:
        safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
        if not safe_name:
            safe_name = "unnamed"
        return safe_name

    @staticmethod
    def _reward_metric_log_key(name: str, stat: str) -> str:
        return f"reward_component_{ActionVideo2WorldModelRectifiedFlowGRPO._safe_metric_name(name)}_{stat}"

    @staticmethod
    def _advantage_metric_log_key(name: str, stat: str) -> str:
        return f"adv_component_{ActionVideo2WorldModelRectifiedFlowGRPO._safe_metric_name(name)}_{stat}"

    def _maybe_repeat_batch_for_group(self, data_batch: Dict[str, torch.Tensor], num_generations: int) -> Dict[str, torch.Tensor]:
        """
        Repeat a batch along dim=0 for group-based GRPO (num_generations per prompt). 用一个 prompt 重复多次，作为一个 group。

        Annotation:
        - This matches the common GRPO setup: one prompt rolled out multiple times to compute group-normalized advantage.
        - We only repeat tensor entries whose first dimension matches batch size.
        """

        if num_generations <= 1:
            return data_batch
        # infer batch size from video/image key
        if self.input_data_key in data_batch:
            b0 = data_batch[self.input_data_key].shape[0]
        elif self.input_image_key in data_batch:
            b0 = data_batch[self.input_image_key].shape[0]
        else:
            raise KeyError("Expected video or image key in data_batch for GRPO.")

        out: Dict[str, torch.Tensor] = {}
        for k, v in data_batch.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == b0:
                # shape: [B, ...] -> [B * num_generations, ...]
                out[k] = v.repeat_interleave(num_generations, dim=0)
            else:
                out[k] = v
        return out

    def _compute_advantage(self, rewards: torch.Tensor, hp: _GrpoHyperParams) -> torch.Tensor:
        """
        Compute advantages, supporting group-normalized mode.
        """
        adv = self._compute_advantage_unclipped(rewards, hp)
        adv = torch.clamp(adv, -hp.adv_clip_max, hp.adv_clip_max)  # shape: [B]
        return adv

    def _compute_advantage_unclipped(self, rewards: torch.Tensor, hp: _GrpoHyperParams) -> torch.Tensor:
        """
        Compute normalized advantages without applying the final advantage clip.
        """

        # rewards shape: [B]
        rewards_f = rewards.to(torch.float32)

        if hp.use_group_adv and hp.num_generations > 1:
            # Group by contiguous chunks of size num_generations
            assert rewards_f.shape[0] % hp.num_generations == 0, "Batch size must be divisible by num_generations."
            n_groups = rewards_f.shape[0] // hp.num_generations
            rewards_g = rewards_f.view(n_groups, hp.num_generations)  # shape: [B] -> [G, K]
            mean = rewards_g.mean(dim=1, keepdim=True)  # shape: [G, K] -> [G, 1]
            std = rewards_g.std(dim=1, keepdim=True).clamp_min(1e-8)  # shape: [G, K] -> [G, 1]
            adv = ((rewards_g - mean) / std).view_as(rewards_f)  # shape: [G, K] -> [B]
        else:
            # Global normalization across data-parallel ranks
            dp_group = _get_data_parallel_group_with_cp()
            gathered = _all_gather_concat_1d(rewards_f, group=dp_group)  # shape: [B] -> [B_total]
            mean = gathered.mean()
            std = gathered.std().clamp_min(1e-8)
            adv = (rewards_f - mean) / std  # shape: [B]

        return adv

    def _compute_rollout_advantages(
        self,
        rewards: torch.Tensor,
        reward_metrics: Dict[str, torch.Tensor],
        hp: _GrpoHyperParams,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rollout advantages.

        For mixed reward, follow DanceGRPO by normalizing each component reward into an advantage first,
        then combining component advantages with configured weights. The raw weighted reward remains available
        for logging as `reward_mean` / `reward_std`.
        """

        if not isinstance(self._reward_model, CompositeRewardModel):
            advantages = self._compute_advantage(rewards, hp).to(
                device=self.tensor_kwargs["device"], dtype=torch.float32
            )
            return advantages, {}

        component_weights = self._reward_model.get_component_weights()
        weight_names = set(component_weights.keys())
        metric_names = set(reward_metrics.keys())
        if metric_names != weight_names:
            missing = sorted(weight_names - metric_names)
            unexpected = sorted(metric_names - weight_names)
            raise ValueError(
                "CompositeRewardModel component rewards must match configured weights exactly. "
                f"Missing component rewards: {missing}; unexpected component rewards: {unexpected}."
            )

        advantage_metrics: Dict[str, torch.Tensor] = {}
        total_advantage: torch.Tensor | None = None
        for name, weight in component_weights.items():
            component_advantage = self._compute_advantage_unclipped(reward_metrics[name], hp).to(
                device=self.tensor_kwargs["device"], dtype=torch.float32
            )
            advantage_metrics[name] = component_advantage
            weighted_advantage = float(weight) * component_advantage
            total_advantage = weighted_advantage if total_advantage is None else total_advantage + weighted_advantage

        assert total_advantage is not None
        total_advantage = torch.clamp(total_advantage, -hp.adv_clip_max, hp.adv_clip_max)
        return total_advantage, advantage_metrics

    def _ensure_action_dtype_inplace(self, data_batch: Dict[str, torch.Tensor]) -> None:
        """
        Ensure action tensor dtype matches the model compute dtype (bf16/fp16).

        Why:
        - In the original (non-GRPO) training pipeline, inputs are typically cast via `.to(**self.tensor_kwargs)`
          before entering the network.
        - In GRPO rollout we call `get_velocity_fn_from_batch()` which captures `condition` containing `action`
          taken directly from `data_batch["action"]`. If that tensor is float32 while the network weights are bf16,
          `nn.Linear(action)` will fail with: mat1 Float vs mat2 BFloat16.
        - This fix is localized to the GRPO path (new code) and avoids modifying the base network implementation.
        """

        if "action" in data_batch and torch.is_tensor(data_batch["action"]):
            data_batch["action"] = data_batch["action"].to(
                device=self.tensor_kwargs["device"], dtype=self.tensor_kwargs["dtype"]
            )

    def _ensure_rollout_input_dtypes_inplace(self, data_batch: Dict[str, torch.Tensor]) -> None:
        """
        Ensure rollout-time conditioner inputs do not accidentally upcast the main latent input to float32.

        Root cause we are preventing:
        - In `minimal_v4_dit.py::prepare_embedded_sequence`, when `self.concat_padding_mask` is enabled,
          it concatenates `padding_mask` to `x_B_C_T_H_W` without `type_as(x)`.
        - If `padding_mask` is float32 while `x` is bf16, `torch.cat` promotes the result to float32, and the
          subsequent PatchEmbed Linear crashes with mat1=float vs mat2=bf16.

        This fix is scoped to GRPO rollout (new code path) to avoid touching the base model/network.
        """

        # padding_mask is used by the backbone when concat_padding_mask=True
        if "padding_mask" in data_batch and torch.is_tensor(data_batch["padding_mask"]):
            data_batch["padding_mask"] = data_batch["padding_mask"].to(
                device=self.tensor_kwargs["device"], dtype=self.tensor_kwargs["dtype"]
            )
        # fps participates in embedding; keep it on-device and in fp32/bf16 consistently
        if "fps" in data_batch and torch.is_tensor(data_batch["fps"]):
            data_batch["fps"] = data_batch["fps"].to(device=self.tensor_kwargs["device"], dtype=torch.float32)

    # ----------------------------- GRPO public APIs -----------------------------
    @torch.no_grad()
    def collect_rollout_and_rewards(
        self, data_batch: Dict[str, torch.Tensor], rollout_seed_offset: int = 0
    ) -> GrpoRolloutSamples:
        """
        Collect rollout trajectory + old_log_probs + reward + advantage.

        This is intended to be called by `trainer_grpo` once per outer iteration, and then reused for multiple updates.
        """

        hp = self._get_grpo_params()
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)

        # 模仿官方的 `Text2WorldModelRectifiedFlow.training_step` (359-362):
        # If compute_online is enabled, overwrite t5 embeddings and set mask to all ones.
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            if self.text_encoder is None:
                raise RuntimeError("text_encoder is None but text_encoder_config.compute_online=True")
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings.to(**self.tensor_kwargs)
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device=self.tensor_kwargs["device"])

        data_batch = self._maybe_repeat_batch_for_group(data_batch, hp.num_generations)
        # IMPORTANT: avoid dtype mismatch inside action embedder (Linear) during rollout
        self._ensure_action_dtype_inplace(data_batch)
        # IMPORTANT: avoid padding_mask upcasting latents to float32 inside backbone
        self._ensure_rollout_input_dtypes_inplace(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        b = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = (
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        )

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        # NOTE: When collecting multiple rollout batches per outer iteration, `rollout_seed_offset` must be different
        # to avoid sampling identical noise trajectories across batches.
        # Also offset by DP rank so different data-parallel replicas do not sample identical trajectories.
        dp_rank = _get_data_parallel_rank_for_seed()
        generator.manual_seed(int(hp.seed) + int(rollout_seed_offset) + 1 * int(dp_rank))
        # -------------------- init noise sampling --------------------
        # Dance-GRPO 风格：同一个 prompt 的 group（num_generations 个样本）共享同一个 init_noise，
        # 以提高训练稳定性（参考 action/tmp/train_grpo_flux.py 的 `--init_same_noise`）。
        if bool(hp.init_same_noise) and int(hp.num_generations) > 1:
            assert b % int(hp.num_generations) == 0, "Batch size must be divisible by num_generations."
            n_groups = b // int(hp.num_generations)
            base_noise = torch.randn(
                (n_groups,) + state_shape,
                device=self.tensor_kwargs["device"],
                dtype=self.tensor_kwargs["dtype"],
                generator=generator,
            )
            # shape: [G, ...] -> [G*K, ...]，使每组 K 个样本共享同一份 noise
            init_noise = base_noise.repeat_interleave(int(hp.num_generations), dim=0)
        else:
            init_noise = torch.randn(
                (b,) + state_shape,
                device=self.tensor_kwargs["device"],
                dtype=self.tensor_kwargs["dtype"],
                generator=generator,
            )

        # IMPORTANT: fixed conditioning across updates
        velocity_fn = self.get_velocity_fn_from_batch(
            data_batch, guidance=float(hp.guidance), is_negative_prompt=False
        )

        # UniPC schedule (single source of truth)
        self.sample_scheduler.set_timesteps(
            hp.num_steps,
            device=self.tensor_kwargs["device"],
            shift=float(hp.shift),
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )
        sigmas = self.sample_scheduler.sigmas.to(device=self.tensor_kwargs["device"], dtype=torch.float32)  # [S+1]
        # NOTE: keep timestep tokens as int64 (same as the existing sampling code path)
        timesteps = self.sample_scheduler.timesteps.to(device=self.tensor_kwargs["device"])  # [S]
        # print(f"sigmas: {sigmas}, timesteps: {timesteps}")
        # 在 FlowUniPCMultistepScheduler 中可以看到：
        # timesteps = sigmas * self.config.num_train_timesteps
        # sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        latents = init_noise#.to(self.tensor_kwargs["dtype"])
        init_noise_local = init_noise
        if self.net.is_context_parallel_enabled:
            cp_group = self.get_context_parallel_group()
            # IMPORTANT: keep noise and latents aligned in shape under context parallel.
            init_noise_local = broadcast_split_tensor(init_noise_local, seq_dim=2, process_group=cp_group)
            latents = broadcast_split_tensor(latents, seq_dim=2, process_group=cp_group)

        all_latents = []
        all_next_latents = []
        all_old_log_probs = []

        # 注意这里 all_latents 包含从 init_noise 到 final_latents 的前一步的所有 latents
        # 而 all_next_latents 包含从 init_noise 的下一步到 final_latents 的所有 latents
        for i, t_tok in enumerate(timesteps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            t_B_1 = torch.stack([t_tok]).unsqueeze(0)  # [1,1]
            t_B_1 = t_B_1.repeat(b, 1)
            # print(f"dtype of init_noise: {init_noise.dtype}, dtype of latents: {latents.dtype}, dtype of t_B_1: {t_B_1.dtype}")
            v_pred = velocity_fn(init_noise_local, latents, t_B_1)
            # print("pass 1 time")

            eps = torch.randn(latents.shape, dtype=torch.float32, device=latents.device, generator=generator)
            step_out = grpo_sde_step(
                latents=latents,
                velocity=v_pred,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=float(hp.eta),
                noise=eps,
                fixed_next_latents=None,
            )

            # IMPORTANT: keep next_latents/latents in model dtype for the next denoise call.
            next_latents = step_out.next_latents
            # next_latents= self.sample_scheduler.step(v_pred, t_tok, latents, return_dict=False, generator=generator)[0]
            all_latents.append(latents)
            all_next_latents.append(next_latents)
            all_old_log_probs.append(step_out.log_prob.to(torch.float32))  # [B]
            
            latents = next_latents

        latents_s = torch.stack(all_latents, dim=1)  # [B,S,...]
        next_latents_s = torch.stack(all_next_latents, dim=1)
        old_log_probs_s = torch.stack(all_old_log_probs, dim=1)  # [B,S]

        # Reward uses final latents; gather CP if enabled
        final_latents = latents
        if self.net.is_context_parallel_enabled:
            final_latents = cat_outputs_cp(final_latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        # include gt_video in reward input
        gt_video = data_batch[input_key]
        if torch.is_tensor(gt_video) and gt_video.device != final_latents.device:
            gt_video = gt_video.to(device=final_latents.device)

        # Decode final latents to pixels for reward models that operate in pixel space.
        # NOTE: `decode()` is inherited from Text2WorldModelRectifiedFlow and is torch.no_grad safe.
        pred_video_pixels = self.decode(final_latents.to(torch.float32))

        rewards = self._reward_model(
            RewardInput(
                # Prefer pixels for reward (SSIM uses pixels; dummy doesn't care).
                video=pred_video_pixels.detach(),
                text=None,
                action=data_batch.get("action", None),
                metadata={
                    "is_image_batch": is_image_batch,
                    "gt_video": gt_video,
                },
            )
        ).to(device=self.tensor_kwargs["device"], dtype=torch.float32)
        reward_metrics = self._extract_reward_metrics(batch_size=int(rewards.shape[0]))

        advantages, advantage_metrics = self._compute_rollout_advantages(rewards, reward_metrics, hp)

        # Match DanceGRPO's training sample construction: rollout still runs all S transitions
        # to produce the final sample/reward, but the last transition sigma[S-1] -> sigma[S]
        # is excluded from the policy loss.
        train_latents_s = latents_s[:, :-1]  # [B, S-1, ...]
        train_next_latents_s = next_latents_s[:, :-1]
        train_old_log_probs_s = old_log_probs_s[:, :-1]
        train_timesteps = timesteps[:-1]

        return GrpoRolloutSamples(
            latents=train_latents_s.detach(),
            next_latents=train_next_latents_s.detach(),
            old_log_probs=train_old_log_probs_s.detach(),
            timestep_tokens=train_timesteps.detach(),
            sigmas=sigmas.detach(),
            # IMPORTANT: store the local (possibly context-parallel split) noise for later updates.
            init_noise=init_noise_local.detach(),
            rewards=rewards.detach(),
            reward_metrics={k: v.detach() for k, v in reward_metrics.items()},
            advantages=advantages.detach(),
            velocity_fn=velocity_fn,
            is_image_batch=is_image_batch,
            advantage_metrics={k: v.detach() for k, v in advantage_metrics.items()},
        )

    def compute_grpo_loss(
        self, 
        samples: GrpoRolloutSamples,  # shape: [B, S, ...], S is the number of timesteps
        update_seed: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute clipped GRPO objective on cached rollout samples.

        Annotation:
        - `samples.old_log_probs` is fixed (from rollout).
        - `new_log_probs` is recomputed with current parameters, so multiple calls correspond to multiple updates.
        """

        hp = self._get_grpo_params()
        S = int(samples.timestep_tokens.shape[0])
        train_T = max(1, int(S * hp.timestep_fraction))

        # -------------------- Per-sample timestep permutation (align with train_grpo_flux.py 556-574) --------------------
        # For each sample in the batch, generate its own random permutation over S timesteps.
        # This is different from a global permutation shared by all samples.
        device = samples.latents.device
        B = int(samples.latents.shape[0])

        gen = torch.Generator(device=device)
        gen.manual_seed(int(hp.seed) + int(update_seed))
        # 每条样本进行独立的时间维度打乱
        perms = torch.stack([torch.randperm(S, device=device, generator=gen) for _ in range(B)], dim=0)  # [B, S]
        idx = perms[:, :train_T]  # [B, train_T]

        # 逐样本的时间步索引
        arange_b = torch.arange(B, device=device)
        # IMPORTANT: keep latents in model dtype when feeding to the net (avoid mat1=float vs mat2=bf16).
        latents_i = samples.latents[arange_b[:, None], idx]  # [B,train_T,...]
        next_latents_i = samples.next_latents[arange_b[:, None], idx]
        old_log_probs_i = samples.old_log_probs[arange_b[:, None], idx].to(torch.float32)  # [B,train_T]

        new_log_probs_list = []
        for j in range(train_T):  # 对所有样本逐步遍历各个训练时间步
            # Per-sample timestep token / sigma for this update step
            idx_b = idx[:, j]  # [B]
            t_tok_b = samples.timestep_tokens.index_select(dim=0, index=idx_b)  # [B]
            # Scheduler sigmas has length S+1; idx_b+1 is valid since idx_b in [0,S-1]
            sigma_b = samples.sigmas.index_select(dim=0, index=idx_b)  # [B]
            sigma_next_b = samples.sigmas.index_select(dim=0, index=(idx_b + 1))  # [B]
            # Model expects timesteps_B_T, so shape [B, 1] is preferred (supports per-sample tokens)
            t_B_1 = t_tok_b.view(B, 1)  # [B,1]

            v_pred = samples.velocity_fn(samples.init_noise, latents_i[:, j], t_B_1)
            step_out = grpo_sde_step(
                latents=latents_i[:, j],
                velocity=v_pred,
                sigma=sigma_b,
                sigma_next=sigma_next_b,
                eta=float(hp.eta),
                noise=torch.zeros_like(latents_i[:, j]),
                fixed_next_latents=next_latents_i[:, j],
            )
            new_log_probs_list.append(step_out.log_prob.to(torch.float32))  # 每次 append 一列 [B, 1]

        new_log_probs = torch.stack(new_log_probs_list, dim=1)  # 按 dim=1 进行 stack，size 为 [B,train_T]
        ratio = torch.exp(new_log_probs - old_log_probs_i)
        adv = samples.advantages.to(torch.float32).unsqueeze(1).expand_as(ratio)

        unclipped = -adv * ratio
        clipped = -adv * torch.clamp(ratio, 1.0 - hp.clip_range, 1.0 + hp.clip_range)
        loss = torch.mean(torch.maximum(unclipped, clipped))

        approx_kl = torch.mean(old_log_probs_i - new_log_probs).detach()
        clip_frac = torch.mean(((ratio - 1.0).abs() > hp.clip_range).to(torch.float32)).detach()
        # print(f"old_log_probs_i: {old_log_probs_i}")
        # print(f"new_log_probs: {new_log_probs}")
        # print(f"adv: {adv}")
        # print(f"ratio: {ratio}")
        # print(f"clip_frac: {clip_frac}")
        # print(f"GRPO loss: {loss}")

        output_batch: Dict[str, torch.Tensor] = {
            "grpo_loss": loss.detach(),
            "reward_mean": samples.rewards.mean().detach(),
            "reward_std": samples.rewards.std().detach(),
            "adv_mean": samples.advantages.mean().detach(),
            "adv_std": samples.advantages.std().detach(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
        for name, metric in samples.reward_metrics.items():
            output_batch[self._reward_metric_log_key(name, "mean")] = metric.mean().detach()
            output_batch[self._reward_metric_log_key(name, "std")] = metric.std().detach()
        for name, metric in samples.advantage_metrics.items():
            output_batch[self._advantage_metric_log_key(name, "mean")] = metric.mean().detach()
            output_batch[self._advantage_metric_log_key(name, "std")] = metric.std().detach()
        # 为了保证使用原本 WandbCallback 不出错，需要加一个 edm_loss 字段，这里直接把 grpo_loss 的值赋给它
        output_batch["edm_loss"] = output_batch["grpo_loss"]
        return output_batch, loss

    # ----------------------------- GRPO training -----------------------------
    def training_step(  # 这玩意放这里不是真正用来用的，而是兼容原本的 trainer 类用的
        self, 
        data_batch: Dict[str, torch.Tensor], 
        iteration: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # NOTE: This path performs a *single update* on a fresh rollout, kept for compatibility.
        log.warning(
            "ActionVideo2WorldModelRectifiedFlowGRPO.training_step does a single-update rollout. "
            "For correct GRPO (multi-updates per rollout), please use `trainer_grpo`."
        )
        samples = self.collect_rollout_and_rewards(data_batch)
        return self.compute_grpo_loss(samples, update_seed=int(iteration))
