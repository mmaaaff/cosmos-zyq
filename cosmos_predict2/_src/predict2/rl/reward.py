"""
Reward model interface for GRPO-style RL training.

Annotation:
- This module is intentionally lightweight and framework-agnostic.
- The reward model can be replaced later (e.g., a real video reward network).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import torch
import torch.nn.functional as F


import numpy as np
import os
import torchvision


@dataclass
class RewardInput:
    """
    Standardized reward input container.

    Notes:
    - `video` can be either decoded pixels (recommended for reward models) or latents (for placeholder rewards).
    - Layout is not enforced here; downstream reward implementations should document expected layout.
    """
    video: Optional[torch.Tensor]
    text: Optional[list[str]]
    action: Optional[torch.Tensor]
    metadata: Dict[str, Any]


class BaseRewardModel(torch.nn.Module):
    """
    Reward model base class.

    Contract:
    - forward() returns rewards of shape [B] on the same device as inputs (or a specified device).
    """

    def forward(self, inp: RewardInput) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class DummyRewardModel(BaseRewardModel):
    """
    Returns zero reward for each sample.
    """

    def forward(self, inp: RewardInput) -> torch.Tensor:
        # Prefer deriving batch size from `video` if present; otherwise fall back to action/text.
        if inp.video is not None:
            batch_size = inp.video.shape[0]
            device = inp.video.device
        elif inp.action is not None:
            batch_size = inp.action.shape[0]
            device = inp.action.device
        elif inp.text is not None:
            batch_size = len(inp.text)
            device = torch.device("cpu")
        else:
            raise ValueError("DummyRewardModel requires at least one of: video/action/text to infer batch size.")

        return torch.zeros((batch_size,), device=device, dtype=torch.float32)


class SSIM_Reward(BaseRewardModel):
    """
    返回每个样本的 SSIM reward（越大越好，范围近似 [0, 1]）。

    设计约定（清晰注释，便于你在 GRPO 里接线）：
    - `inp.video`：模型生成的视频/图像（推荐用 decode 后像素域；latents 也可以但 SSIM 意义弱）。
      支持形状：
        - [B, C, T, H, W]（推荐）
        - [B, T, C, H, W]
        - [B, C, H, W]（图像，等价于 T=1）
    - 参考/GT 视频需要通过 `inp.metadata` 提供（否则 SSIM 无从计算）。
      支持的 key（按顺序优先匹配）：
        - "gt_video", "target_video", "reference_video", "video_gt", "video_target", "video_ref"

    输出：
    - shape: [B]，dtype=float32，device 与 `inp.video` 一致。
    """

    def forward(self, inp: RewardInput) -> torch.Tensor:
        if inp.video is None:
            raise ValueError("SSIM_Reward requires inp.video (predicted video/image tensor).")

        # -------- 1) 取出参考视频（GT/target）--------
        ref = None
        if "gt_video" in inp.metadata and inp.metadata["gt_video"] is not None:
                ref = inp.metadata["gt_video"]
        else:
            raise ValueError("SSIM_Reward requires a reference video in inp.metadata. Expected key: gt_video")
        if not isinstance(ref, torch.Tensor):
            raise TypeError(f"SSIM_Reward expects reference video to be torch.Tensor, got {type(ref)}")

        pred = inp.video

        # -------- 2) 统一 layout -> [B, C, T, H, W] --------
        # def _to_bcthw(x: torch.Tensor, *, name: str) -> torch.Tensor:
        #     if x.ndim == 4:
        #         # [B, C, H, W] -> [B, C, 1, H, W]
        #         return x.unsqueeze(2).contiguous()
        #     raise ValueError(f"SSIM_Reward expects {name} to be 4D/5D tensor, got ndim={x.ndim} shape={tuple(x.shape)}")

        # pred = _to_bcthw(pred, name="inp.video")
        # ref = _to_bcthw(ref, name="inp.metadata[ref]")

        if pred.shape != ref.shape:
            raise ValueError(f"SSIM_Reward requires pred/ref to have same shape. pred={tuple(pred.shape)} ref={tuple(ref.shape)}")

        # -------- 3) 数值范围归一化到 [0, 1]（SSIM 常用假设）--------
        def _to_01(x: torch.Tensor) -> torch.Tensor:
            # uint8: [0,255] -> [0,1]
            if x.dtype == torch.uint8:
                x = x.to(torch.float32) / 255.0
                return x
            # float: 尽量兼容 [-1,1] 与 [0,1]
            x = x.to(torch.float32)
            # 以全局最小/最大做启发式判断：如果存在负值，按 [-1,1] 处理
            if torch.isfinite(x).all() and x.min() < -1e-3:
                x = (x + 1.0) * 0.5
            return x.clamp(0.0, 1.0)

        pred_01 = _to_01(pred)
        ref_01 = _to_01(ref)
        
        # ----------------------------
        # self.save_dir = "/inspire/qb-ilm/project/robot3d/czxs25210241/cosmos-zyq/output/tmp"
        # self.save_video_tensor(pred_01, "pred")
        # self.save_video_tensor(ref_01, "ref")
        # ----------------------------

        # -------- 4) 计算 SSIM（逐帧），再对 T 平均，输出 [B] --------
        # 实现说明（清晰注释）：
        # - 用高斯窗口计算局部均值/方差/协方差（conv2d，按 channel 分组）。
        # - 对每一帧返回一个标量 SSIM，然后对时间维平均得到每个样本的 reward。

        B, C, T, H, W = pred_01.shape
        device = pred_01.device

        # 组装高斯核（11x11, sigma=1.5），与常见 SSIM 实现一致。
        win_size = 11
        sigma = 1.5
        coords = torch.arange(win_size, device=device, dtype=torch.float32) - (win_size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        window_2d = (g[:, None] * g[None, :]).to(torch.float32)  # [K, K]
        window = window_2d.expand(C, 1, win_size, win_size).contiguous()  # [C,1,K,K] for groups=C

        # [B,C,T,H,W] -> [B*T, C, H, W]
        x = pred_01.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        y = ref_01.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # padding="same" 的近似：用 reflect padding 减少边界效应
        pad = win_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        y_pad = F.pad(y, (pad, pad, pad, pad), mode="reflect")

        mu_x = F.conv2d(x_pad, window, groups=C)
        mu_y = F.conv2d(y_pad, window, groups=C)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        # Var[X] = E[X^2] - E[X]^2
        sigma_x2 = F.conv2d(x_pad * x_pad, window, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y_pad * y_pad, window, groups=C) - mu_y2
        sigma_xy = F.conv2d(x_pad * y_pad, window, groups=C) - mu_xy

        # 常用常数（以 data_range=1.0）
        c1 = (0.01**2)
        c2 = (0.03**2)

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)  # shape: [B*T, C, H, W]

        # 聚合：先对 (C,H,W) 取均值 -> 每帧一个标量；再 reshape 回 [B,T]，对 T 平均
        ssim_per_frame = ssim_map.mean(dim=(1, 2, 3))  # [B*T]
        ssim_bt = ssim_per_frame.view(B, T)
        reward_b = ssim_bt.mean(dim=1)  # [B]

        return reward_b.to(dtype=torch.float32, device=pred.device)
    
    # ---------------------- 用来看看 sde 采样的质量行不行 ---------------------------
    def save_video_tensor(self, video: torch.Tensor, prefix: str = "video"):
        """保存视频张量到文件"""
        import uuid
        import time
            
        # 确保是 [B, C, T, H, W] 格式
        if video.ndim == 5:
            B, C, T, H, W = video.shape
            for b in range(min(B, 2)):  # 最多保存前2个batch
                # 创建文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"{prefix}_{timestamp}_{unique_id}_b{b}.mp4"
                filepath = os.path.join(self.save_dir, filename)
                
                # 取出单个样本 [C, T, H, W]
                video_sample = video[b]
                
                # 保存为视频文件
                self._save_video_file(video_sample, filepath)
                
                # 同时保存第一帧为图片（便于快速查看）
                if T > 0:
                    img_filename = f"{prefix}_{timestamp}_{unique_id}_b{b}_frame0.png"
                    img_path = os.path.join(self.save_dir, img_filename)
                    self._save_image(video_sample[:, 0], img_path)
    
    def _save_video_file(self, video: torch.Tensor, filepath: str):
        """保存单个视频到文件"""
        # video: [C, T, H, W]
        C, T, H, W = video.shape
        
        # 转换为 [T, C, H, W] -> [T, H, W, C]
        video_np = video.permute(1, 2, 3, 0).cpu().numpy()
        
        # 确保数值范围在 [0, 1]
        video_np = np.clip(video_np, 0, 1)
        
        if C == 1:
            # 灰度视频，需要复制为RGB
            video_np = np.repeat(video_np, 3, axis=-1)
        elif C == 3:
            # RGB视频，保持原样
            pass
        else:
            # 其他通道数，取前3个通道
            video_np = video_np[..., :3]
        
        # 转换为 uint8
        video_np = (video_np * 255).astype(np.uint8)
        
        # 使用 OpenCV 或 imageio 保存
        try:
            import imageio
            imageio.mimwrite(filepath, video_np, fps=10, quality=8)
            print(f"Saved video to {filepath}")
        except ImportError:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, 10, (W, H))
            for frame in video_np:
                out.write(frame)
            out.release()
            print(f"Saved video to {filepath} (using OpenCV)")
            
    def _save_image(self, image: torch.Tensor, filepath: str):
        """保存单张图片"""
        import matplotlib.pyplot as plt
        
        # image: [C, H, W]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        
        plt.imsave(filepath, image_np)


class OpticalFlowReward(BaseRewardModel):
    """
    使用 OpenCV Farneback 稠密光流比较生成视频与 GT 视频的运动一致性。

    设计约定：
    - `inp.video` 为预测视频，`inp.metadata["gt_video"]` 为 GT 视频。
    - 输入必须为 `[B, C, T, H, W]`。
    - 第一版只支持 `estimator="farneback"`。
    - 输出 reward shape 为 `[B]`，值越大越好；默认返回负误差。
    """

    def __init__(
        self,
        estimator: str = "farneback",
        score_mode: str = "robust",
        resize_hw: tuple[int, int] | list[int] | None = (128, 160),
        frame_stride: int = 1,
        max_frames: int = 0,
        eps: float = 1e-3,
        # Farneback 参数：
        # - pyr_scale: 金字塔层间缩放比例，越小层数利用越细。
        # - levels: 金字塔层数，越大越能覆盖大位移，但更慢。
        # - winsize: 局部窗口大小，越大越平滑稳健，但会损失细节。
        # - iterations: 每层迭代次数，越大收敛更充分但更慢。
        # - poly_n: 多项式展开邻域大小，越大越平滑。
        # - poly_sigma: 多项式展开高斯平滑强度，通常与 poly_n 配套调。
        # - flags: OpenCV Farneback 标志位，0 表示默认行为。
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0,
    ):
        super().__init__()
        estimator = str(estimator).lower()
        if estimator != "farneback":
            raise ValueError(f"OpticalFlowReward only supports estimator='farneback', got {estimator}")
        score_mode = str(score_mode).lower()
        if score_mode not in {"robust", "mse"}:
            raise ValueError(f"OpticalFlowReward score_mode must be 'robust' or 'mse', got {score_mode}")

        self.estimator = estimator
        self.score_mode = score_mode
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None
        self.frame_stride = max(int(frame_stride), 1)
        self.max_frames = max(int(max_frames), 0)
        self.eps = float(eps)
        self.pyr_scale = float(pyr_scale)
        self.levels = int(levels)
        self.winsize = int(winsize)
        self.iterations = int(iterations)
        self.poly_n = int(poly_n)
        self.poly_sigma = float(poly_sigma)
        self.flags = int(flags)

    def _to_bcthw(self, x: torch.Tensor, *, name: str) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"OpticalFlowReward expects {name} to be [B,C,T,H,W], got shape={tuple(x.shape)}")
        return x.contiguous()

    def _to_01(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            return x.to(torch.float32) / 255.0
        x = x.to(torch.float32)
        if torch.isfinite(x).all() and x.min() < -1e-3:
            x = (x + 1.0) * 0.5
        return x.clamp(0.0, 1.0)

    def _preprocess_video(self, x: torch.Tensor) -> np.ndarray:
        x = self._to_01(x)
        if self.frame_stride > 1:
            x = x[:, :, :: self.frame_stride]
        if self.max_frames > 0:
            x = x[:, :, : self.max_frames]
        if self.resize_hw is not None:
            target_h, target_w = self.resize_hw
            b, c, t, _, _ = x.shape
            x_bt = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, x.shape[-2], x.shape[-1])
            x_bt = F.interpolate(x_bt, size=(target_h, target_w), mode="bilinear", align_corners=False)
            x = x_bt.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()

        if x.shape[1] == 3:
            r = x[:, 0]
            g = x[:, 1]
            b = x[:, 2]
            x = (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(1)
        elif x.shape[1] != 1:
            raise ValueError(f"OpticalFlowReward expects channel count 1 or 3, got {x.shape[1]}")

        x = (x * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
        return x.squeeze(1).cpu().numpy()

    def _compute_farnebck_flow(self, prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            prev=prev_frame,
            next=next_frame,
            flow=None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags,
        )

    def _score_flow_pair(self, pred_flow: np.ndarray, gt_flow: np.ndarray) -> float:
        diff = pred_flow.astype(np.float32) - gt_flow.astype(np.float32)
        if self.score_mode == "mse":
            return float(np.mean(diff ** 2))
        flow_err = np.sqrt(np.sum(diff ** 2, axis=-1) + self.eps ** 2)
        return float(np.mean(flow_err))

    def forward(self, inp: RewardInput) -> torch.Tensor:
        if inp.video is None:
            raise ValueError("OpticalFlowReward requires inp.video")
        gt_video = inp.metadata.get("gt_video")
        if gt_video is None:
            raise ValueError("OpticalFlowReward requires inp.metadata['gt_video']")
        if not isinstance(gt_video, torch.Tensor):
            raise TypeError(f"OpticalFlowReward expects gt_video to be torch.Tensor, got {type(gt_video)}")

        pred = self._to_bcthw(inp.video, name="inp.video")
        gt = self._to_bcthw(gt_video, name="inp.metadata['gt_video']")
        if pred.shape != gt.shape:
            raise ValueError(f"OpticalFlowReward requires pred/ref to have same shape. pred={tuple(pred.shape)} ref={tuple(gt.shape)}")
        if pred.shape[2] < 2:
            raise ValueError(f"OpticalFlowReward requires at least 2 frames, got T={pred.shape[2]}")

        pred_np = self._preprocess_video(pred.detach())
        gt_np = self._preprocess_video(gt.detach())

        rewards = []
        for pred_video, gt_video_np in zip(pred_np, gt_np):
            frame_scores = []
            for t in range(pred_video.shape[0] - 1):
                pred_flow = self._compute_farnebck_flow(pred_video[t], pred_video[t + 1])
                gt_flow = self._compute_farnebck_flow(gt_video_np[t], gt_video_np[t + 1])
                frame_scores.append(self._score_flow_pair(pred_flow, gt_flow))
            rewards.append(-float(np.mean(frame_scores)))

        return torch.tensor(rewards, device=pred.device, dtype=torch.float32)


class VJEPA2Reward(BaseRewardModel):
    """
    使用普通版 V-JEPA2 encoder 的 sliding-window 特征相似度作为 reward。

    设计约定：
    - 输入：`inp.video` 为生成视频，`inp.metadata["gt_video"]` 为 GT 视频。
    - 只使用 encoder：与官方 demo 对齐，调用 `AutoModel.get_vision_features(...)`。
    - reward 计算方式：
      1) 沿时间维用长度 `num_frames`、步长 `stride` 的窗口滑过视频；
      2) 每个窗口用 V-JEPA2 encoder 提取 patch-wise features；
      3) 对每个窗口的 feature 做 token 平均池化，得到窗口级 embedding；
      4) 对对应窗口计算 cosine similarity；
      5) 对所有窗口的 cosine similarity 取平均，输出 shape `[B]`。
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitg-fpc64-384",
        num_frames: int = 64,
        image_size: int = 384,
        stride: int = 1,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.stride = int(stride)
        if self.num_frames <= 0:
            raise ValueError(f"VJEPA2Reward expects num_frames > 0, got {self.num_frames}")
        if self.stride <= 0:
            raise ValueError(f"VJEPA2Reward expects stride > 0, got {self.stride}")

        # - V-JEPA2 按需加载（第一次算 reward 时加载），避免在 model update 阶段常驻占用显存。
        # - 调用 `unload()` 后会释放模型对象；下次 forward 会自动重新加载。
        self._model: Optional[torch.nn.Module] = None
        self._processor = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        from transformers import AutoModel, AutoVideoProcessor

        model = AutoModel.from_pretrained(self.model_name)
        processor = AutoVideoProcessor.from_pretrained(self.model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # 优先以 processor 的 crop size 为准，避免手工配置与 checkpoint 不一致。
        crop_size = getattr(processor, "crop_size", None)
        if isinstance(crop_size, dict) and "height" in crop_size:
            self.image_size = int(crop_size["height"])

        self._model = model
        self._processor = processor

    def unload(self, clear_cuda_cache: bool = True) -> None:
        """
        卸载 V-JEPA2 模型
        """
        # if self._model is not None:
        #     # 显式迁回 CPU，避免模型被其他对象引用导致无法释放
        #     self._model.to(device="cpu")
        self._model = None
        self._processor = None
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _to_bcthw(self, x: torch.Tensor, *, name: str) -> torch.Tensor:
        """
        将输入统一为 [B, C, T, H, W]。
        """
        if x.ndim == 4:
            # [B, C, H, W] -> [B, C, 1, H, W]
            return x.unsqueeze(2).contiguous()
        if x.ndim != 5:
            raise ValueError(
                f"VJEPA2Reward expects {name} to be 4D/5D tensor, "
                f"got shape={tuple(x.shape)}"
            )
        # 兼容 [B, T, C, H, W]
        if x.shape[1] not in (1, 3) and x.shape[2] in (1, 3):
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def _to_01(self, x: torch.Tensor) -> torch.Tensor:
        """
        数值映射到 [0, 1]，兼容 uint8 / [-1,1] / [0,1]。
        """
        if x.dtype == torch.uint8:
            return x.to(torch.float32) / 255.0
        x = x.to(torch.float32)
        if torch.isfinite(x).all() and x.min() < -1e-3:
            x = (x + 1.0) * 0.5
        return x.clamp(0.0, 1.0)

    def _ensure_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        统一到 3 通道，便于送入 V-JEPA2 processor。
        """
        _, c, _, _, _ = x.shape
        if c == 3:
            return x
        if c == 1:
            return x.repeat(1, 3, 1, 1, 1)
        if c > 3:
            return x[:, :3].contiguous()
        raise ValueError(f"VJEPA2Reward expects channel count in {{1,3}} or >=3, got {c}")

    def _sample_frames(self, x: torch.Tensor, target_t: Optional[int] = None) -> torch.Tensor:
        """
        时间维重采样到固定帧数。短视频时用它补成单窗口。
        """
        target_t = self.num_frames if target_t is None else int(target_t)
        _, _, t, _, _ = x.shape
        if t == target_t:
            return x
        # 线性等间隔采样，允许重复索引（当 t < target_t）。
        idx = torch.linspace(0, t - 1, steps=target_t, device=x.device)
        idx = idx.round().long().clamp(0, t - 1)
        return x.index_select(dim=2, index=idx)

    def _get_model_device(self) -> torch.device:
        self._ensure_model_loaded()
        assert self._model is not None
        return next(self._model.parameters()).device

    def _ensure_model_device(self, target_device: torch.device) -> None:
        """
        按需把 V-JEPA2 移到目标设备，避免在 CPU 上做推理。
        """
        self._ensure_model_loaded()
        assert self._model is not None
        current = self._get_model_device()
        if current != target_device:
            self._model.to(device=target_device)

    def _build_sliding_windows(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """
        构造 sliding windows，输出形状 `[B, N, C, F, H, W]`。N 为窗口数，F 为窗口长度。
        """
        _, _, t, _, _ = video_bcthw.shape
        if t < self.num_frames:
            # 短视频退化为单窗口，并重采样到模型要求的帧数。
            return self._sample_frames(video_bcthw).unsqueeze(1)

        windows = video_bcthw.unfold(dimension=2, size=self.num_frames, step=self.stride)
        # unfold 后形状为 [B, C, N, H, W, F]，这里调整为 [B, N, C, F, H, W]。
        windows = windows.permute(0, 2, 1, 5, 3, 4).contiguous()
        return windows

    def _pool_encoder_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        根据 `vjepa2/src/models/vision_transformer.py`，encoder `forward()` 最终返回 shape `[B, N, D]` 的 token 序列。
        """
        if feats.ndim != 3:
            raise RuntimeError(
                "VJEPA2Reward expects get_vision_features() to return patch-wise features "
                f"with shape [B, N, D], got {tuple(feats.shape)}"
            )
        return feats.mean(dim=1)

    def _prepare_processor_batch(self, windows_bncfhw: torch.Tensor) -> list[torch.Tensor]:
        """
        将窗口张量转换为 processor 期望的 `T x C x H x W` clip 列表。
        """
        flat = windows_bncfhw.reshape(-1, *windows_bncfhw.shape[2:])  # [B*N, C, F, H, W]
        flat = self._ensure_rgb(flat)
        flat = (self._to_01(flat) * 255.0).round().to(torch.uint8)
        # 逐窗口转为 `T x C x H x W`，与本地 demo 的调用形式保持一致。
        return [clip.permute(1, 0, 2, 3).cpu() for clip in flat]

    def _encode_video_windows(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """
        编码所有窗口，输出形状 `[B, N, D]`。
        """
        windows = self._build_sliding_windows(video_bcthw)  # [B, N, C, F, H, W]
        b, n = windows.shape[:2]
        processor_inputs = self._prepare_processor_batch(windows)
        self._ensure_model_loaded()
        assert self._model is not None
        assert self._processor is not None
        model_device = self._get_model_device()
        model_inputs = self._processor(processor_inputs, return_tensors="pt")
        if "pixel_values_videos" not in model_inputs:
            raise RuntimeError("VJEPA2Reward expects AutoVideoProcessor to return 'pixel_values_videos'.")
        pixel_values = model_inputs["pixel_values_videos"].to(device=model_device, dtype=torch.float32)

        with torch.no_grad():
            if not hasattr(self._model, "get_vision_features"):
                raise RuntimeError("VJEPA2Reward requires a V-JEPA2 AutoModel with get_vision_features().")
            feats = self._model.get_vision_features(pixel_values)

        pooled = self._pool_encoder_features(feats)  # [B*N, D]
        return pooled.view(b, n, -1)

    def forward(self, inp: RewardInput) -> torch.Tensor:
        if inp.video is None:
            raise ValueError("VJEPA2Reward requires inp.video (pred video tensor).")
        if "gt_video" not in inp.metadata or inp.metadata["gt_video"] is None:
            raise ValueError("VJEPA2Reward requires inp.metadata['gt_video'].")

        pred = inp.video
        ref = inp.metadata["gt_video"]

        pred = self._to_bcthw(pred, name="inp.video")
        ref = self._to_bcthw(ref, name="inp.metadata['gt_video']")
        if pred.shape[0] != ref.shape[0]:
            raise ValueError(
                f"VJEPA2Reward expects pred/ref to have the same batch size, got {pred.shape[0]} vs {ref.shape[0]}"
            )
        if pred.shape[2] != ref.shape[2]:
            raise ValueError(
                f"VJEPA2Reward expects pred/ref to have the same number of frames for aligned windows, "
                f"got {pred.shape[2]} vs {ref.shape[2]}"
            )
        self._ensure_model_device(pred.device)

        z_pred = self._encode_video_windows(pred)  # [B, N, D]
        z_ref = self._encode_video_windows(ref)    # [B, N, D]
        if z_pred.shape != z_ref.shape:
            raise RuntimeError(
                f"VJEPA2Reward expects pred/ref window embeddings to align, got {tuple(z_pred.shape)} vs {tuple(z_ref.shape)}"
            )

        window_reward = F.cosine_similarity(z_pred, z_ref, dim=-1)  # [B, N]
        reward = window_reward.mean(dim=1)  # [B]
        return reward.to(device=pred.device, dtype=torch.float32)