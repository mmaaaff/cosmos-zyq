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


