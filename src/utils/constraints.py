"""Constraint functions for neural network regularization."""

import torch


def compute_firing_rate_constraint(
    firing_rate: torch.Tensor,
    target_firing_rate: float = 1.0,
    lagrange_multiplier: torch.Tensor = None,
    rho: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute augmented Lagrangian loss for firing rate constraint.

    Implements the constraint: mean(firing_rate) = target_firing_rate
    Using ALM: loss = λᵀ·h + (ρ/2)·||h||²
    where h = firing_rate - target

    Args:
        firing_rate (torch.Tensor): Firing rates, shape (..., n_channels)
            Will be averaged across all dimensions except the last (channel dimension)
        target_firing_rate (float): Target firing rate per channel (default 1.0)
        lagrange_multiplier (torch.Tensor): Lagrange multipliers, shape (n_channels,)
            If None, only quadratic penalty is applied
        rho (float): Penalty parameter for quadratic term (default 1.0)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - alm_loss: Augmented Lagrangian loss (scalar)
            - constraint_violation: h = mean(firing_rate) - target, shape (n_channels,)
    """
    # Compute constraint violation: h_i = mean(r_i) - target
    constraint_violation = firing_rate.mean(dim=tuple(range(firing_rate.ndim - 1))) - target_firing_rate

    # Quadratic penalty term: (ρ/2) · ||h||²
    quadratic_penalty = rho / 2.0 * (constraint_violation ** 2).sum()

    # Linear penalty term: λᵀ · h (if Lagrange multiplier provided)
    if lagrange_multiplier is not None:
        linear_penalty = (lagrange_multiplier * constraint_violation).sum()
    else:
        linear_penalty = 0.0

    alm_loss = linear_penalty + quadratic_penalty

    return alm_loss, constraint_violation


def update_lagrange_multiplier(
    lagrange_multiplier: torch.Tensor,
    constraint_violation: torch.Tensor,
    rho: float = 1.0,
) -> None:
    """
    Update Lagrange multipliers using dual-ascent step.

    Implements: λ_new = λ + ρ · h

    Args:
        lagrange_multiplier (torch.Tensor): Lagrange multipliers, shape (n_channels,)
        constraint_violation (torch.Tensor): Constraint violation h, shape (n_channels,)
        rho (float): Penalty parameter (default 1.0)
    """
    with torch.no_grad():
        lagrange_multiplier.add_(rho * constraint_violation)
