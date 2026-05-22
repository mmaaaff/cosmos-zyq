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
    transition_mean: torch.Tensor
    transition_std: torch.Tensor


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
    sigma_dependent_eta: bool = False,
    sigma_max: torch.Tensor | None = None,
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
        sigma_dependent_eta: if True, use Flow-OPD's eta_t = eta * sqrt(sigma / (1 - sigma)).
        sigma_max: substitute denominator value when sigma == 1 for sigma-dependent eta.

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

    latents_f = latents.to(torch.float32)
    velocity_f = velocity.to(torch.float32)
    sigma = sigma.to(device=latents_f.device, dtype=torch.float32)
    sigma_next = sigma_next.to(device=latents_f.device, dtype=torch.float32)

    while sigma.ndim < latents_f.ndim:
        sigma = sigma.unsqueeze(-1)
    while sigma_next.ndim < latents_f.ndim:
        sigma_next = sigma_next.unsqueeze(-1)

    if sigma_dependent_eta:
        assert sigma_max is not None, "sigma_max is required when sigma_dependent_eta=True."
        sigma_max = sigma_max.to(device=latents_f.device, dtype=torch.float32)
        while sigma_max.ndim < latents_f.ndim:
            sigma_max = sigma_max.unsqueeze(-1)
        sigma_denom = torch.where(sigma == 1, sigma_max, sigma)
        eta_t = eta * torch.sqrt(sigma / (1.0 - sigma_denom))
    else:
        eta_t = torch.as_tensor(eta, device=latents_f.device, dtype=torch.float32)

    dsigma = sigma_next - sigma  # negative when sigma decreases
    prev_sample_mean = latents_f + dsigma * velocity_f
    pred_x0 = latents_f - sigma * velocity_f

    # delta > 0 when sigma decreases
    delta = (sigma - sigma_next)
    std = eta_t * torch.sqrt(delta)

    score_estimate = -(latents_f - pred_x0 * (1.0 - sigma)) / sigma**2
    log_term = (-0.5 * eta_t**2 * score_estimate)
    prev_sample_mean = prev_sample_mean + log_term * dsigma

    if fixed_next_latents is None:
        next_latents = prev_sample_mean + std * noise.to(torch.float32)
    else:
        next_latents = fixed_next_latents.to(torch.float32)

    next_latents = next_latents.to(latents.dtype)
    # Compute log_prob on the exact value that callers cache/reuse, so rollout old_log_prob
    # and training-time new_log_prob evaluate the same transition.
    log_prob = _normal_log_prob(next_latents.to(torch.float32), prev_sample_mean, std)
    # print(f"mean: {mean[0, 0, 0, :10, 10]}")
    # print(f"std: {std}")
    # print(f"log_probs: {log_prob}")
    return GrpoStepOutput(
        next_latents=next_latents,
        pred_x0=pred_x0.to(latents.dtype),
        log_prob=log_prob,
        transition_mean=prev_sample_mean,
        transition_std=std,
    )
