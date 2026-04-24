"""
GRPO SDE sampler utilities (log-probability aware).

Why this exists:
- COSMOS inference sampling uses UniPC (deterministic multi-step solver) and therefore does not provide log-probabilities.
- GRPO/PPO-style objectives require log_prob under the policy for state transitions.

Design choices (critical):
- We reuse UniPC's `sigmas` schedule as the single source of truth for sigma/timestep scheduling.
- The stochasticity is injected via an explicit Gaussian transition:
    x_{i+1} ~ Normal(mean_i, std_i)
  and we compute log_prob(x_{i+1} | x_i).
- The transition mean uses the DanceGRPO/Flux SDE correction term rather than a plain Euler ODE mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class GrpoStepOutput:
    next_latents: torch.Tensor
    pred_x0: torch.Tensor
    log_prob: torch.Tensor  # shape [B]


def _normal_log_prob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Compute log N(x | mean, std) aggregated over non-batch dims.

    Annotation:
    - We aggregate by mean over non-batch dims, matching the style used in many diffusion-RL references.
    - `std` can be scalar or broadcastable to x.
    """

    # Ensure broadcastable std
    var = std * std
    # log N = -0.5 * ((x-mean)^2 / var + log(2*pi*var))
    logp = -0.5 * ((x - mean) ** 2) / var - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    # Aggregate over all non-batch dimensions
    reduce_dims = tuple(range(1, logp.ndim))
    return logp.mean(dim=reduce_dims)


def grpo_sde_step(
    *,
    latents: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float,
    noise: torch.Tensor,
    fixed_next_latents: torch.Tensor | None = None,
) -> GrpoStepOutput:
    """
    One stochastic step with log_prob, using UniPC-style sigma parameterization.

    Args:
        latents: current x_i, shape [B, ...]
        velocity: shape [B, ...]
        sigma: sigma_i (broadcastable), scalar tensor or shape [B] / [B,1,...]
        sigma_next: sigma_{i+1}
        eta: noise strength; eta=0 -> deterministic transition
        noise: standard normal eps with same shape as latents
        fixed_next_latents: if provided, do NOT sample; instead compute log_prob of this next state.

    Returns:
        GrpoStepOutput with `next_latents`, `pred_x0`, `log_prob` (shape [B]).

    Notes:
    - This follows the `sde_solver=True` path in DanceGRPO's `train_grpo_flux.py`:
        pred_x0 = x - sigma * v
        mean   = x + (sigma_next - sigma) * v
        score  = -(x - (1 - sigma) * pred_x0) / sigma**2
        mean   = mean + (-0.5 * eta**2 * score) * (sigma_next - sigma)
        std    = eta * sqrt(sigma - sigma_next)
    """

    # Ensure float32 for stability in log-prob calculations
    latents_f = latents.to(torch.float32)
    velocity_f = velocity.to(torch.float32)

    # Broadcast sigma to latents shape
    while sigma.ndim < latents_f.ndim:
        sigma = sigma.unsqueeze(-1)
    while sigma_next.ndim < latents_f.ndim:
        sigma_next = sigma_next.unsqueeze(-1)

    dsigma = sigma_next - sigma  # negative when sigma decreases
    mean = latents_f + dsigma * velocity_f

    # delta > 0 when sigma decreases
    delta = (sigma - sigma_next)
    std = (eta * torch.sqrt(delta))

    pred_x0 = latents_f - sigma * velocity_f

    sigma_safe = sigma.clamp_min(1e-6)
    score_estimate = -(latents_f - pred_x0 * (1.0 - sigma)) / (sigma_safe * sigma_safe)
    mean = mean + (-0.5 * eta * eta * score_estimate) * dsigma

    if fixed_next_latents is None:
        next_latents = mean + std * noise.to(torch.float32)
    else:
        next_latents = fixed_next_latents.to(torch.float32)

    log_prob = _normal_log_prob(next_latents, mean, std)
    # print(f"mean: {mean[0, 0, 0, :10, 10]}")
    # print(f"std: {std}")
    # print(f"log_probs: {log_prob}")
    return GrpoStepOutput(next_latents=next_latents.to(latents.dtype), pred_x0=pred_x0.to(latents.dtype), log_prob=log_prob)
