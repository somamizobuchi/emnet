"""Constraint functions for neural network regularization."""

import torch
import numpy as np

from typing import Optional


def make_raised_cosine_basis(n_taps: int, n_basis: int, log_offset: float = 1.0) -> torch.Tensor:
    """
    Build a raised cosine temporal basis matrix (Pillow et al., 2008).

    Basis functions are raised cosine bumps whose centres are uniformly spaced
    on a log-compressed time axis.  This gives finer resolution near t=0 and
    coarser resolution for older lags — matching the known temporal precision
    of neural responses.

    Each column integrates to roughly the same energy, so a unit-norm
    coefficient vector produces a unit-norm kernel regardless of which
    basis functions are activated.

    Args:
        n_taps   : Number of time samples in the kernel (T_train).
        n_basis  : Number of basis vectors (< n_taps; typically 5-10).
        log_offset: Shifts the log-time axis to avoid log(0); larger values
                    give more uniform (linear) spacing, smaller values give
                    stronger log-compression (default 1.0).

    Returns:
        basis : Tensor of shape (n_taps, n_basis), columns have unit L2 norm.
    """
    assert n_basis <= n_taps, "n_basis must be <= n_taps"

    t = np.arange(n_taps, dtype=np.float64)
    t_log = np.log(t + log_offset)

    t_log_min, t_log_max = t_log[0], t_log[-1]

    # Width chosen so adjacent bumps overlap at half-peak
    width = (t_log_max - t_log_min) / max(n_basis - 1, 1) * 1.5

    # Inset the last centre by one full width so its support doesn't reach
    # t_log_max (lag 0), guaranteeing basis[-1, :] == 0.
    centres = np.linspace(t_log_min, t_log_max - width / 2, n_basis)

    # Each column: raised cosine centred at centres[j]
    # cos²(π/2 * (t_log - c_j) / half_period) on its support, else 0
    basis = np.zeros((n_taps, n_basis), dtype=np.float64)
    for j, c in enumerate(centres):
        phase = (t_log - c) / width * np.pi
        mask = np.abs(phase) < np.pi / 2
        basis[mask, j] = np.cos(phase[mask]) ** 2

    # Normalize each column to unit L2 norm
    col_norms = np.linalg.norm(basis, axis=0, keepdims=True)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    basis /= col_norms

    return torch.tensor(basis, dtype=torch.float32)


def compute_firing_rate_constraint(
    firing_rate: torch.Tensor,
    target_firing_rate: float = 1.0,
    lagrange_multiplier: Optional[torch.Tensor] = None,
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
    constraint_violation = (
        firing_rate.mean(dim=tuple(range(firing_rate.ndim - 1))) - target_firing_rate
    )

    # Quadratic penalty term: (ρ/2) · ||h||²
    quadratic_penalty = rho / 2.0 * (constraint_violation**2).sum()

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
