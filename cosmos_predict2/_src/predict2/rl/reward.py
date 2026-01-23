"""
Reward model interface for GRPO-style RL training.

Annotation:
- This module is intentionally lightweight and framework-agnostic.
- The reward model can be replaced later (e.g., a real video reward network).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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