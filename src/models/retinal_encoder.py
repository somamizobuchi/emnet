import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinalEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_spatial: int,
        n_temporal: int,
        target_firing_rate: float = 1.0,
        rho: float = 1.0,
    ) -> None:
        super().__init__()
        self.N = n_channels
        self.T = n_temporal
        self.X = n_spatial
        self.target_firing_rate = target_firing_rate
        self.rho = rho

        self.spatial_kernels = nn.Parameter(
            torch.zeros(self.N, self.X, self.X, dtype=torch.float32)
        )
        self.temporal_kernels = nn.Parameter(torch.zeros(self.N, self.T))
        self.nonlinear = nn.Softplus()

        # Lagrange multiplier for firing rate constraint
        self.register_parameter(
            "Lambda",
            nn.Parameter(torch.zeros(self.N)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (b, t, x, x)
        """
        r = torch.matmul(
                x.view([x.shape[0], -1, self.X * self.X]),
                self.spatial_kernels.view([-1, self.X * self.X]).T) # (b, t, N)
        # Valid channel-wise temporal convolution
        r = F.conv1d(r.transpose_(1, 2), self.temporal_kernels.unsqueeze(1), groups=self.N)

        return self.nonlinear(r) # (b, N, t - T + 1)

    def compute_firing_rate_constraint(
        self,
        firing_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute augmented Lagrangian loss for firing rate constraint.

        Args:
            firing_rate (torch.Tensor): Firing rates from forward pass, shape (batch, n_channels, time)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - alm_loss: Augmented Lagrangian loss (scalar)
                - constraint_violation: h = mean(firing_rate) - target, shape (n_channels,)
        """
        # Compute per-channel mean firing rate across batch and time: h_i = mean(r_i) - target
        constraint_violation = firing_rate.mean(dim=(0, 2)) - self.target_firing_rate

        # Quadratic penalty term: (ρ/2) · ||h||²
        quadratic_penalty = self.rho / 2.0 * (constraint_violation ** 2).sum()

        # Linear penalty term: λᵀ · h
        linear_penalty = (self.Lambda * constraint_violation).sum()

        alm_loss = linear_penalty + quadratic_penalty

        return alm_loss, constraint_violation

    def update_lagrange_multiplier(self, constraint_violation: torch.Tensor) -> None:
        """
        Update Lagrange multipliers using dual-ascent step.

        Implements: λ_new = λ + ρ · h

        Args:
            constraint_violation (torch.Tensor): Constraint violation h, shape (n_channels,)
        """
        with torch.no_grad():
            self.Lambda.add_(self.rho * constraint_violation)

    def normalize_kenels(self):
        with torch.no_grad():
            kernels_flat = self.spatial_kernels.view([-1, self.X * self.X])
            self.spatial_kernels.copy_((kernels_flat / kernels_flat.norm(dim=1, keepdim=True)).reshape([-1, self.X, self.X]))
            self.temporal_kernels.copy_(self.temporal_kernels / self.temporal_kernels.norm(dim=1, keepdim=True))
