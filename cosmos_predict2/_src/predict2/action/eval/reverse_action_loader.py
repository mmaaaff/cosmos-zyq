"""Action loader for forward-then-reverse action-conditioned evaluation."""

from __future__ import annotations

import numpy as np
from loguru import logger

from cosmos_predict2.action_conditioned import load_default_action_fn


def reverse_action_sequence(forward_actions: np.ndarray) -> np.ndarray:
    """Build the reverse action sequence for one contiguous forward action sequence.

    The first six action dimensions are relative end-effector deltas, so reversing the
    trajectory uses the opposite deltas in reverse temporal order. The gripper dimension
    stores the next-frame gripper state; the initial gripper state is unavailable in the
    action-only sequence, so the final reverse gripper value is filled with the first
    forward action's gripper value.
    """
    if forward_actions.ndim != 2 or forward_actions.shape[1] != 7:
        raise ValueError(f"Expected forward_actions with shape [T, 7], got {forward_actions.shape}")
    if forward_actions.shape[0] == 0:
        raise ValueError("Expected at least one forward action.")

    reverse_actions = forward_actions[::-1].copy()
    reverse_actions[:, :6] *= -1

    if forward_actions.shape[0] > 1:
        reverse_actions[:-1, 6] = forward_actions[:-1, 6][::-1]
    reverse_actions[-1, 6] = forward_actions[0, 6]
    return reverse_actions


def load_forward_reverse_action_fn():
    """Construct an action loader that keeps the first N chunks and appends their reverse."""

    default_load_fn = load_default_action_fn()

    def load_fn(json_data: dict, video_path: str, args) -> dict:
        action_data = default_load_fn(json_data, video_path, args)
        actions = action_data["actions"]
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        num_chunks = int(getattr(args, "eval_reverse_action_num_chunks", 0))
        if num_chunks <= 0:
            raise ValueError(
                "eval_reverse_action_num_chunks must be > 0 when using "
                "load_forward_reverse_action_fn."
            )

        chunk_size = int(args.chunk_size)
        forward_len = num_chunks * chunk_size
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected actions with shape [T, 7], got {actions.shape}")
        if actions.shape[0] < forward_len:
            raise ValueError(
                f"Not enough actions for {num_chunks} chunks of size {chunk_size}: "
                f"need {forward_len}, got {actions.shape[0]}."
            )

        forward_actions = actions[:forward_len].copy()
        reverse_actions = reverse_action_sequence(forward_actions)
        action_data["actions"] = np.concatenate([forward_actions, reverse_actions], axis=0)

        logger.info(
            "Built forward-reverse eval actions: "
            f"num_chunks={num_chunks}, chunk_size={chunk_size}, "
            f"shape={action_data['actions'].shape}"
        )
        return action_data

    return load_fn
