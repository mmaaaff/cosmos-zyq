# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb

from cosmos_predict2._src.imaginaire.callbacks.every_n import EveryN
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.trainer import ImaginaireTrainer
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm, rotm2euler, rotm2quat
from cosmos_predict2._src.predict2.rl.reward import RewardInput

try:
    from megatron.core import parallel_state
except ImportError:  # pragma: no cover
    parallel_state = None


@dataclass
class RolloutValidationResult:
    """单个 episode rollout 后的结果容器。"""

    key: str
    pred_video: torch.Tensor
    gt_video: torch.Tensor
    action: torch.Tensor
    side_by_side: np.ndarray


def _dist_is_initialized() -> bool:
    """检查 torch.distributed 是否已经可用并初始化。"""

    return dist.is_available() and dist.is_initialized()


def _get_data_parallel_rank_world() -> tuple[int, int]:
    """获取数据并行 rank/world size，优先使用 Megatron 的数据并行组。"""

    if parallel_state is not None and parallel_state.is_initialized():
        try:
            return int(parallel_state.get_data_parallel_rank()), int(parallel_state.get_data_parallel_world_size())
        except Exception:
            pass
    if _dist_is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """在分布式环境中对张量做全局求和；单卡时原样返回。"""

    if _dist_is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _safe_std_from_sums(total: torch.Tensor, total_sq: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """根据 sum/sum of squares/count 计算标准差，并避免空样本除零。"""

    count_clamped = count.clamp_min(1.0)
    mean = total / count_clamped
    var = total_sq / count_clamped - mean * mean
    return torch.sqrt(var.clamp_min(0.0))


def _sanitize_metric_name(name: str) -> str:
    """把 reward component 名称转换成 wandb/日志中安全的 metric key。"""

    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    return safe or "unnamed"


class ActionRolloutRewardValidation(EveryN):
    """
    周期性执行 action-conditioned 长 rollout 验证，并记录 reward 相关指标。
    """

    def __init__(
        self,
        every_n: Optional[int] = None,  # 每隔多少个训练 step 触发一次；None 时沿用 EveryN 默认行为。
        step_size: int = 1,  # EveryN 计数步长，通常保持为 1。
        max_eval_episodes: int = 8,  # 每次验证最多抽取多少个 episode。
        max_chunks_per_episode: int = 4,  # 每个 episode 最多连续 rollout 多少个 action chunk。
        save_video_count: int = 4,  # 最多保存多少个 GT/预测并排视频。
        num_steps: Optional[int] = None,  # 采样步数；None 时读取 model.config.grpo["num_steps"]。
        guidance: Optional[float] = None,  # classifier-free guidance 强度；None 时读取 grpo["guidance"]。
        shift: Optional[float] = None,  # diffusion/sampler 的 shift 参数；None 时读取 grpo["shift"]。
        seed: int = 0,  # 验证采样随机种子基数，会结合 iteration/rank/episode index 派生实际 seed。
        save_fps: int = 4,  # 保存验证视频及 batch 中 fps 字段使用的帧率。
        use_ema: bool = False,  # 是否在验证时切换到 EMA 权重。
        sampler_type: str = "unipc",  # 采样器类型；当前只支持 "unipc"。
        chunk_size: Optional[int] = None,  # 每次生成消耗的 action 数；None 时从模型配置读取。
        resolution: Optional[str] = None,  # 强制 resize 分辨率，格式 "H,W"；None/"none" 时从模型或原视频读取。
        input_root: Optional[str] = None,  # 数据根目录，可用于推导标注目录和视频根目录。
        input_json_sub_folder: Optional[str] = None,  # input_root 下保存 json 标注的子目录。
        annotation_root: Optional[str] = None,  # 显式指定标注 json 目录，优先级高于 input_root/dataset。
        video_root: Optional[str] = None,  # 显式指定视频根目录，优先级高于 input_root/dataset。
        camera_id: Any = None,  # 从标注 videos 字段中选择的相机 id；None 时优先使用 dataset.cam_ids[0]。
        state_key: Optional[str] = None,  # 标注中机械臂状态字段名；None 时使用 dataset._state_key 或 "state"。
        gripper_key: Optional[str] = None,  # 标注中夹爪状态字段名；None 时使用 dataset._gripper_key 或默认键。
        action_scaler: Optional[float] = None,  # arm action 缩放系数；None 时优先使用 dataset.c_act_scaler，否则默认 20.0。
        gripper_scale: Optional[float] = None,  # gripper action 缩放系数；action_scaler 走默认路径时默认 1.0。
        fps_downsample_ratio: Optional[int] = None,  # 读取视频和状态时的降采样比例；None 时使用 dataset 配置。
        start_frame_idx: int = 0,  # 从 episode 的第几帧开始 rollout。
        num_latent_conditional_frames: int = 1,  # 告诉模型 batch 中有多少 latent conditional frames。
        run_at_start: bool = False,  # 是否在训练开始时立即运行一次验证。
    ) -> None:
        super().__init__(every_n=every_n, step_size=step_size, barrier_after_run=True, run_at_start=run_at_start)
        # 保存配置项；路径参数统一转成 Path，后续解析标注/视频时更直接。
        self.max_eval_episodes = int(max_eval_episodes)
        self.max_chunks_per_episode = int(max_chunks_per_episode)
        self.save_video_count = int(save_video_count)
        self.num_steps = num_steps
        self.guidance = guidance
        self.shift = shift
        self.seed = int(seed)
        self.save_fps = int(save_fps)
        self.use_ema = bool(use_ema)
        self.sampler_type = str(sampler_type).lower()
        self.chunk_size = chunk_size
        self.resolution = resolution
        self.input_root = Path(input_root) if input_root is not None else None
        self.input_json_sub_folder = input_json_sub_folder
        self.annotation_root = Path(annotation_root) if annotation_root is not None else None
        self.video_root = Path(video_root) if video_root is not None else None
        self.camera_id = camera_id
        self.state_key = state_key
        self.gripper_key = gripper_key
        self.action_scaler = action_scaler
        self.gripper_scale = gripper_scale
        self.fps_downsample_ratio = fps_downsample_ratio
        self.start_frame_idx = int(start_frame_idx)
        self.num_latent_conditional_frames = int(num_latent_conditional_frames)
        self.name = self.__class__.__name__

        if self.sampler_type != "unipc":
            raise ValueError("ActionRolloutRewardValidation currently only supports sampler_type='unipc'.")

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """训练开始时创建本 callback 的本地输出目录。"""

        del model, iteration
        self.local_dir = Path(self.config.job.path_local) / self.name
        if distributed.is_rank0():
            self.local_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        """EveryN 触发的验证主流程：选择 episode、执行 rollout、聚合 reward、保存日志/视频。"""

        del data_batch, output_batch, loss
        dataloader_val = getattr(trainer, "dataloader_val", None)
        dataset = getattr(dataloader_val, "dataset", None)

        ann_files = self._resolve_annotation_files(dataset)
        if not ann_files:
            log.warning("ActionRolloutRewardValidation skipped: no validation annotation files found.")
            return

        selected = self._select_eval_annotations(ann_files)
        rank, world_size = _get_data_parallel_rank_world()
        # 不同数据并行 rank 处理不同 episode，最后只 all-reduce 数值指标。
        local_ann_files = selected[rank::world_size]
        log.info(
            f"{self.name}: validating {len(local_ann_files)} local / {len(selected)} total episodes at iteration {iteration}",
            rank0_only=False,
        )

        context = self._validation_context(model)
        was_training = model.training
        rewards: list[torch.Tensor] = []
        component_values: dict[str, list[torch.Tensor]] = {}
        saved_payloads: list[tuple[str, np.ndarray]] = []

        try:
            model.eval()
            with context():
                for local_idx, ann_path in enumerate(local_ann_files):
                    try:
                        # 每个 episode 使用确定性 seed；iteration/rank/local_idx 防止不同样本完全相同。
                        result = self._run_annotation_rollout(
                            model=model,
                            annotation_path=ann_path,
                            dataset=dataset,
                            episode_seed=self.seed + int(iteration) * 10_000 + rank * 1_000 + local_idx,
                        )
                        reward = model._reward_model(
                            RewardInput(
                                video=result.pred_video,
                                text=None,
                                action=result.action,
                                metadata={"gt_video": result.gt_video},
                            )
                        ).detach().to(device=result.pred_video.device, dtype=torch.float32)
                        rewards.append(reward.flatten())

                        # reward model 可选地暴露分项指标，例如 SSIM、L1 等，统一随主 reward 聚合。
                        if hasattr(model._reward_model, "get_last_metrics"):
                            for name, value in model._reward_model.get_last_metrics().items():
                                if torch.is_tensor(value):
                                    component_values.setdefault(name, []).append(value.detach().flatten().to(reward))

                        if len(saved_payloads) < self.save_video_count:  # 保存少量视频
                            saved_payloads.append((result.key, result.side_by_side))
                    except Exception as exc:
                        log.warning(f"{self.name}: failed validation rollout for {ann_path}: {exc}", rank0_only=False)
        finally:
            if hasattr(model, "unload_reward_model"):
                model.unload_reward_model()
            torch.cuda.empty_cache()
            # callback 结束后恢复训练模式，避免影响后续 train step。
            if was_training:
                model.train()

        metrics = self._aggregate_reward_metrics(rewards, component_values)
        self._log_metrics(metrics, iteration)
        self._save_videos(saved_payloads, iteration)

    def _validation_context(self, model: ImaginaireModel):
        """根据 use_ema 决定验证时是否进入模型 EMA 权重上下文。"""

        if not self.use_ema:
            return nullcontext
        if not getattr(getattr(model, "config", None), "ema", None) or not model.config.ema.enabled:
            log.warning(f"{self.name}: use_ema=True but model EMA is disabled; using regular weights.")
            return nullcontext
        return partial(model.ema_scope, self.name)

    def _resolve_annotation_files(self, dataset: Any) -> list[Path]:
        """按显式参数、input_root、dataset 的优先级解析待验证标注文件。"""

        if self.annotation_root is not None:
            return sorted(self.annotation_root.glob("*.json"))
        if self.input_root is not None and self.input_json_sub_folder is not None:
            return sorted((self.input_root / self.input_json_sub_folder).glob("*.json"))
        if dataset is not None and hasattr(dataset, "ann_files"):
            return sorted(Path(p) for p in dataset.ann_files)
        return []

    def _select_eval_annotations(self, ann_files: list[Path]) -> list[Path]:
        """从全部标注中均匀抽取最多 max_eval_episodes 个 episode。不是随机采样, 验证集固定"""

        if self.max_eval_episodes <= 0:
            return []
        if len(ann_files) <= self.max_eval_episodes:
            return ann_files
        if self.max_eval_episodes == 1:
            return [ann_files[0]]
        indices = np.linspace(0, len(ann_files) - 1, num=self.max_eval_episodes, dtype=int)
        return [ann_files[int(i)] for i in indices]

    def _run_annotation_rollout(
        self,
        model: ImaginaireModel,
        annotation_path: Path,
        dataset: Any,
        episode_seed: int,
    ) -> RolloutValidationResult:
        """读取单个标注文件，解析 GT 视频和 action 序列，然后执行 rollout。"""

        with open(annotation_path, "r") as f:
            label = json.load(f)

        video_path = self._resolve_video_path(label, dataset)
        gt_video = self._read_and_resize_video(video_path, model)
        fps_downsample_ratio = self._resolve_fps_downsample_ratio(dataset)
        if fps_downsample_ratio > 1:
            # 视频帧和状态/action 必须使用同一个降采样比例，保证时间步对齐。
            gt_video = gt_video[::fps_downsample_ratio]
        actions = self._get_action_sequence_from_states(label, dataset)
        key = self._get_episode_key(label, annotation_path)
        return self._run_rollout_from_arrays(model, gt_video, actions, key=key, episode_seed=episode_seed)

    def _resolve_video_path(self, label: dict[str, Any], dataset: Any) -> Path:
        """根据标注中的 videos 字段和根目录配置推导实际视频路径。"""

        video_root = self.video_root
        if video_root is None and self.input_root is not None:
            video_root = self.input_root
        if video_root is None and dataset is not None and hasattr(dataset, "video_path"):
            video_root = Path(dataset.video_path)
        if video_root is None:
            video_root = Path(".")

        camera_id = self.camera_id
        if camera_id is None and dataset is not None and getattr(dataset, "cam_ids", None):
            camera_id = dataset.cam_ids[0]
        if camera_id is None:
            camera_id = 0

        videos = label["videos"]
        # videos 既可能是 dict(camera_id -> path)，也可能是按 camera id 排列的 list。
        if isinstance(videos, dict):
            item = videos.get(camera_id, videos.get(str(camera_id)))
        else:
            item = videos[int(camera_id)]
        if item is None:
            raise KeyError(f"Could not find camera_id={camera_id} in annotation videos.")

        rel_path = item["video_path"] if isinstance(item, dict) else item
        video_path = Path(rel_path)
        if video_path.is_absolute():
            return video_path
        return video_root / video_path

    def _read_and_resize_video(self, video_path: Path, model: ImaginaireModel) -> np.ndarray:
        """读取视频为 uint8[T,H,W,C]，并按模型/参数要求 resize。"""

        import mediapy

        video = mediapy.read_video(str(video_path))
        video = np.asarray(video)
        if video.dtype != np.uint8:
            # mediapy/其他 reader 可能返回 [0,1] 浮点；统一转为 [0,255] uint8。
            if np.nanmax(video) <= 1.5:
                video = video * 255.0
            video = np.clip(video, 0, 255).astype(np.uint8)

        target_h, target_w = self._resolve_resolution(model, video.shape[1], video.shape[2])
        if (video.shape[1], video.shape[2]) == (target_h, target_w):
            return video
        return np.stack([mediapy.resize_image(frame, (target_h, target_w)) for frame in video], axis=0).astype(np.uint8)

    def _resolve_resolution(self, model: ImaginaireModel, fallback_h: int, fallback_w: int) -> tuple[int, int]:
        """解析目标分辨率：显式参数优先，其次模型配置，最后保留原尺寸。"""

        if self.resolution and self.resolution != "none":
            h, w = self.resolution.split(",")
            return int(h), int(w)
        if hasattr(model, "get_video_height_width"):
            h, w = model.get_video_height_width()
            return int(h), int(w)
        return int(fallback_h), int(fallback_w)

    def _get_action_sequence_from_states(self, label: dict[str, Any], dataset: Any) -> np.ndarray:
        """从绝对 arm/gripper 状态构造模型需要的相对 action 序列。"""

        state_key = self.state_key or getattr(dataset, "_state_key", "state")
        gripper_key = self.gripper_key or getattr(dataset, "_gripper_key", "continuous_gripper_state")
        fps_downsample_ratio = self._resolve_fps_downsample_ratio(dataset)
        arm_states = np.asarray(label[state_key])[::fps_downsample_ratio, :6]
        gripper_states = np.asarray(label[gripper_key])[::fps_downsample_ratio]
        actions = self._relative_actions(arm_states, gripper_states, use_quat=False)

        # 复用训练 dataset 的 action scaler，缺省时与当前 action 训练配置保持一致。
        if dataset is not None and hasattr(dataset, "c_act_scaler") and self.action_scaler is None:
            scaler = np.asarray(dataset.c_act_scaler, dtype=np.float32)
        else:
            action_scaler = 20.0 if self.action_scaler is None else float(self.action_scaler)
            gripper_scale = 1.0 if self.gripper_scale is None else float(self.gripper_scale)
            scaler = np.asarray([action_scaler] * 6 + [gripper_scale], dtype=np.float32)
        return (actions * scaler).astype(np.float32)

    def _resolve_fps_downsample_ratio(self, dataset: Any) -> int:
        """解析 fps 降采样比例，并保证最小为 1。"""

        if self.fps_downsample_ratio is not None:
            return max(int(self.fps_downsample_ratio), 1)
        return max(int(getattr(dataset, "fps_downsample_ratio", 1)), 1)

    @staticmethod
    def _relative_actions(arm_states: np.ndarray, gripper_states: np.ndarray, use_quat: bool = False) -> np.ndarray:
        """把连续绝对位姿转换成上一帧坐标系下的相对位移/相对旋转/action。"""

        action_dim = 8 if use_quat else 7
        actions = np.zeros((arm_states.shape[0] - 1, action_dim), dtype=np.float32)
        for k in range(1, arm_states.shape[0]):
            prev_xyz = arm_states[k - 1, 0:3]
            prev_rpy = arm_states[k - 1, 3:6]
            prev_rotm = euler2rotm(prev_rpy)
            curr_xyz = arm_states[k, 0:3]
            curr_rpy = arm_states[k, 3:6]
            curr_rotm = euler2rotm(curr_rpy)
            # 平移和旋转都变换到上一时刻末端执行器坐标系，和训练 action 定义一致。
            rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
            rel_rotm = prev_rotm.T @ curr_rotm
            actions[k - 1, 0:3] = rel_xyz
            if use_quat:
                actions[k - 1, 3:7] = rotm2quat(rel_rotm)
                actions[k - 1, 7] = gripper_states[k]
            else:
                actions[k - 1, 3:6] = rotm2euler(rel_rotm)
                actions[k - 1, 6] = gripper_states[k]
        return actions

    def _run_rollout_from_arrays(
        self,
        model: ImaginaireModel,
        gt_video: np.ndarray,
        actions: np.ndarray,
        key: str,
        episode_seed: int,
    ) -> RolloutValidationResult:
        """
        用 GT 初始帧和 action chunks 自回归生成长视频，并整理 reward 输入。
        batch size 为 1
        """

        chunk_size = self._resolve_chunk_size(model)
        start = max(self.start_frame_idx, 0)
        max_available_actions = min(actions.shape[0] - start, gt_video.shape[0] - start - 1)
        num_chunks = min(self.max_chunks_per_episode, max_available_actions // chunk_size)
        if num_chunks <= 0:
            raise ValueError(
                f"Not enough frames/actions for rollout: frames={gt_video.shape[0]}, actions={actions.shape[0]}, "
                f"start_frame_idx={start}, chunk_size={chunk_size}"
            )

        device = self._model_device(model)
        dtype = self._model_dtype(model)
        current_frame = gt_video[start]
        pred_chunks: list[torch.Tensor] = []
        action_chunks = []

        for chunk_idx in range(num_chunks):
            action_start = start + chunk_idx * chunk_size
            action_chunk_np = actions[action_start : action_start + chunk_size].astype(np.float32)
            action_chunks.append(action_chunk_np)
            action_chunk = torch.from_numpy(action_chunk_np).unsqueeze(0).to(device=device, dtype=dtype)
            data_batch = self._build_chunk_batch(model, current_frame, action_chunk, chunk_size)
            # 每个 chunk 以上一个 chunk 的最后一帧作为条件帧，形成自回归长 rollout。
            latents = model.generate_samples_from_batch(
                data_batch,
                n_sample=1,
                guidance=self._resolve_guidance(model),
                seed=episode_seed + chunk_idx,
                is_negative_prompt=False,
                num_steps=self._resolve_num_steps(model),
                shift=self._resolve_shift(model),
            )
            decoded = model.decode(latents)
            decoded_uint8 = self._video_tensor_to_uint8_bcthw(decoded)
            if chunk_idx > 0:
                # 后续 chunk 的第 0 帧是上一段末帧条件，拼接时去掉，避免重复帧。
                decoded_uint8 = decoded_uint8[:, :, 1:]
            pred_chunks.append(decoded_uint8)
            current_frame = decoded_uint8[0, :, -1].permute(1, 2, 0).cpu().numpy()

        pred_video = self._concat_chunk_videos(pred_chunks).to(device=device)
        # GT clip 与预测视频裁到同样时长，reward model 可以直接逐帧比较。
        target_len = pred_video.shape[2]
        gt_clip = gt_video[start : start + target_len]
        gt_tensor = self._numpy_video_to_bcthw(gt_clip, device=device)
        action_tensor = torch.from_numpy(np.concatenate(action_chunks, axis=0)).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        side_by_side = self._make_side_by_side(gt_tensor, pred_video)
        return RolloutValidationResult(
            key=key,
            pred_video=pred_video,
            gt_video=gt_tensor,
            action=action_tensor,
            side_by_side=side_by_side,
        )

    def _build_chunk_batch(
        self,
        model: ImaginaireModel,
        initial_frame: np.ndarray,
        action: torch.Tensor,
        chunk_size: int,
    ) -> dict[str, Any]:
        """构造单个 chunk 采样所需的最小 data_batch。"""

        device = self._model_device(model)
        dtype = self._model_dtype(model)
        frame = torch.from_numpy(np.asarray(initial_frame, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        _, h, w = frame.shape
        video = torch.zeros((1, 3, chunk_size + 1, h, w), dtype=torch.uint8, device=device)
        # 只填第 0 帧作为条件，后续帧由模型根据 action 生成。
        video[0, :, 0] = frame.to(device=device)
        data_batch: dict[str, Any] = {
            "dataset_name": "action_rollout_validation",
            "video": video,
            "action": action,
            "fps": torch.tensor([float(self.save_fps)], device=device, dtype=torch.float32),
            "padding_mask": torch.zeros((1, 1, h, w), device=device, dtype=dtype),
            "t5_text_embeddings": torch.zeros((1, 512, 1024), device=device, dtype=dtype),
            "t5_text_mask": torch.ones((1, 512), device=device, dtype=torch.int64),
            "ai_caption": [""],
            "num_conditional_frames": self.num_latent_conditional_frames,
        }
        self._maybe_compute_online_text_embeddings(model, data_batch)
        return data_batch

    def _maybe_compute_online_text_embeddings(self, model: ImaginaireModel, data_batch: dict[str, Any]) -> None:
        """若模型配置要求在线文本编码，则用空 caption 生成真实 text embedding。"""

        text_encoder_config = getattr(getattr(model, "config", None), "text_encoder_config", None)
        if not getattr(text_encoder_config, "compute_online", False):
            return
        if getattr(model, "text_encoder", None) is None:
            raise RuntimeError("text_encoder_config.compute_online=True but model.text_encoder is None.")
        text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings.to(device=self._model_device(model), dtype=self._model_dtype(model))
        data_batch["t5_text_mask"] = torch.ones(
            text_embeddings.shape[0], text_embeddings.shape[1], device=self._model_device(model), dtype=torch.int64
        )

    def _resolve_chunk_size(self, model: ImaginaireModel) -> int:
        """解析每个 rollout chunk 包含的 action 数。"""

        if self.chunk_size is not None:
            return int(self.chunk_size)
        net_config = getattr(getattr(model, "config", None), "net", None)
        if net_config is not None and hasattr(net_config, "num_action_per_chunk"):
            return int(net_config.num_action_per_chunk)
        grpo = getattr(getattr(model, "config", None), "grpo", {}) or {}
        return int(grpo.get("chunk_size", 12))

    def _resolve_num_steps(self, model: ImaginaireModel) -> int:
        """解析推理采样步数。"""

        if self.num_steps is not None:
            return int(self.num_steps)
        grpo = getattr(getattr(model, "config", None), "grpo", {}) or {}
        return int(grpo.get("num_steps", 35))

    def _resolve_guidance(self, model: ImaginaireModel) -> float:
        """解析推理 guidance 强度。"""

        if self.guidance is not None:
            return float(self.guidance)
        grpo = getattr(getattr(model, "config", None), "grpo", {}) or {}
        return float(grpo.get("guidance", 0.0))

    def _resolve_shift(self, model: ImaginaireModel) -> float:
        """解析推理 sampler shift 参数。"""

        if self.shift is not None:
            return float(self.shift)
        grpo = getattr(getattr(model, "config", None), "grpo", {}) or {}
        return float(grpo.get("shift", 5.0))

    @staticmethod
    def _model_device(model: ImaginaireModel) -> torch.device:
        """从模型 tensor_kwargs 中读取推理设备，缺省时优先使用 cuda。"""

        tensor_kwargs = getattr(model, "tensor_kwargs", {})
        return torch.device(tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    @staticmethod
    def _model_dtype(model: ImaginaireModel) -> torch.dtype:
        """从模型 tensor_kwargs 中读取推理 dtype。"""

        tensor_kwargs = getattr(model, "tensor_kwargs", {})
        return tensor_kwargs.get("dtype", torch.float32)

    @staticmethod
    def _concat_chunk_videos(chunks: list[torch.Tensor]) -> torch.Tensor:
        """沿时间维拼接多个 chunk 的预测视频。"""

        if not chunks:
            raise ValueError("Expected at least one chunk video.")
        return torch.cat(chunks, dim=2)

    @staticmethod
    def _video_tensor_to_uint8_bcthw(video: torch.Tensor) -> torch.Tensor:
        """把模型输出统一转换成 uint8[B,C,T,H,W] 视频张量。"""

        video = video.detach()
        if video.dtype == torch.uint8:
            return video.contiguous()
        video = video.to(torch.float32)
        if torch.isfinite(video).all() and video.min() < -1e-3:
            # decode 输出可能是 [-1,1]，保存/可视化前转成 [0,1]。
            video = (video + 1.0) * 0.5
        return (video.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous()

    @staticmethod
    def _numpy_video_to_bcthw(video: np.ndarray, device: torch.device) -> torch.Tensor:
        """把 numpy 视频 [T,H,W,C] 转为 torch 视频 [B,C,T,H,W]。"""

        tensor = torch.from_numpy(np.asarray(video, dtype=np.uint8)).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        return tensor.to(device=device)

    @staticmethod
    def _bcthw_to_t_h_w_c(video: torch.Tensor) -> np.ndarray:
        """把 torch 视频 [B,C,T,H,W] 转回 numpy [T,H,W,C]，用于 mediapy/wandb。"""

        video = ActionRolloutRewardValidation._video_tensor_to_uint8_bcthw(video).detach().cpu()
        return video[0].permute(1, 2, 3, 0).numpy()

    @staticmethod
    def _make_side_by_side(gt_video: torch.Tensor, pred_video: torch.Tensor) -> np.ndarray:
        """生成 GT 和预测左右拼接的视频，便于人工查看 rollout 质量。"""

        gt_np = ActionRolloutRewardValidation._bcthw_to_t_h_w_c(gt_video)
        pred_np = ActionRolloutRewardValidation._bcthw_to_t_h_w_c(pred_video)
        min_t = min(gt_np.shape[0], pred_np.shape[0])
        return np.concatenate([gt_np[:min_t], pred_np[:min_t]], axis=2)

    def _aggregate_reward_metrics(
        self,
        rewards: list[torch.Tensor],
        component_values: dict[str, list[torch.Tensor]],
    ) -> dict[str, float]:
        """聚合各 rank 的 reward 均值/标准差和 reward component 指标。"""

        device = rewards[0].device if rewards else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if rewards:
            reward_flat = torch.cat([r.to(device=device, dtype=torch.float32).flatten() for r in rewards])
        else:
            reward_flat = torch.zeros((0,), device=device, dtype=torch.float32)

        count = torch.tensor(float(reward_flat.numel()), device=device)
        total = reward_flat.sum()
        total_sq = (reward_flat * reward_flat).sum()
        _all_reduce_sum(count)  # 跨 rank 求和
        _all_reduce_sum(total)
        _all_reduce_sum(total_sq)

        # 用 count/total/total_sq 做 all-reduce，比跨 rank 收集全部 reward 更省显存。
        metrics = {
            "val_rollout/reward_count": float(count.item()),
            "val_rollout/reward_mean": float((total / count.clamp_min(1.0)).item()),
            "val_rollout/reward_std": float(_safe_std_from_sums(total, total_sq, count).item()),
        }

        for name, values in component_values.items():
            if values:
                flat = torch.cat([v.to(device=device, dtype=torch.float32).flatten() for v in values])
            else:
                flat = torch.zeros((0,), device=device, dtype=torch.float32)
            comp_count = torch.tensor(float(flat.numel()), device=device)
            comp_total = flat.sum()
            comp_total_sq = (flat * flat).sum()
            _all_reduce_sum(comp_count)
            _all_reduce_sum(comp_total)
            _all_reduce_sum(comp_total_sq)
            safe_name = _sanitize_metric_name(name)
            metrics[f"val_rollout/reward_component_{safe_name}_mean"] = float(
                (comp_total / comp_count.clamp_min(1.0)).item()
            )
            metrics[f"val_rollout/reward_component_{safe_name}_std"] = float(
                _safe_std_from_sums(comp_total, comp_total_sq, comp_count).item()
            )
        return metrics

    def _log_metrics(self, metrics: dict[str, float], iteration: int) -> None:
        """只在 rank0 写日志和 wandb reward 指标。"""

        if not distributed.is_rank0():
            return
        log.info(f"{self.name} iteration {iteration}: {metrics}")
        if wandb.run:
            wandb.log(metrics, step=iteration)

    def _save_videos(self, payloads: list[tuple[str, np.ndarray]], iteration: int) -> None:
        """只在 rank0 保存 GT/预测并排视频，并按需上传 wandb。"""

        if not distributed.is_rank0() or not payloads:
            return
        import mediapy

        save_dir = self.local_dir / f"iter_{iteration:09d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        wandb_payload = {}
        for idx, (key, video) in enumerate(payloads[: self.save_video_count]):
            filename = f"{idx:03d}_{self._safe_file_stem(key)}.mp4"
            path = save_dir / filename
            mediapy.write_video(str(path), video, fps=self.save_fps)
            if wandb.run:
                wandb_payload[f"val_rollout/video_{idx:03d}"] = wandb.Video(str(path), fps=self.save_fps, format="mp4")
        if wandb_payload:
            wandb.log(wandb_payload, step=iteration)

    @staticmethod
    def _safe_file_stem(value: str) -> str:
        """把 episode key 转成适合作为文件名的短字符串。"""

        safe = re.sub(r"[^0-9a-zA-Z_.-]+", "_", str(value)).strip("_")
        return safe[:120] or "episode"

    @staticmethod
    def _get_episode_key(label: dict[str, Any], annotation_path: Path) -> str:
        """从标注中提取可读的 episode 标识，缺省时使用文件名。"""

        for key in ("episode_id", "original_path"):
            if key in label:
                return str(label[key])
        metadata = label.get("episode_metadata", {})
        for key in ("episode_id", "segment_id"):
            if key in metadata:
                return str(metadata[key])
        return annotation_path.stem
