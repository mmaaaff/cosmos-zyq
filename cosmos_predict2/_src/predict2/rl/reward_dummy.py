"""
Dummy reward for GRPO integration bring-up.

Annotation:
- This is a placeholder reward model to validate the end-to-end GRPO plumbing.
- It returns zeros (or a simple heuristic if desired later).
"""

from __future__ import annotations

import torch

from cosmos_predict2._src.predict2.rl.reward_base import BaseRewardModel, RewardInput


class DummyRewardModel(BaseRewardModel):
    """
    Returns zero reward for each sample.
    """

    def forward(self, inp: RewardInput) -> torch.Tensor:
        print(RewardInput)
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


