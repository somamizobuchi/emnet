import torch
import torch.nn as nn


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

        # Use nn.Linear for spatial projection
        self.spatial_projection = nn.Linear(self.X * self.X, self.N, bias=False)

        # Use nn.Conv1d for temporal filtering (depthwise convolution)
        self.temporal_conv = nn.Conv1d(
            self.N, self.N, kernel_size=self.T, groups=self.N, bias=False
        )

        # Initialize with small random values and normalize
        nn.init.normal_(self.spatial_projection.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.temporal_conv.weight, mean=0.0, std=0.01)

        # Normalize spatial and temporal weights to unit norm
        with torch.no_grad():
            # Spatial: weight is (N, X*X), normalize along dim=1
            self.spatial_projection.weight.copy_(
                self.spatial_projection.weight / self.spatial_projection.weight.norm(dim=1, keepdim=True)
            )
            # Temporal: weight is (N, 1, T), squeeze, normalize, unsqueeze
            temp_weight = self.temporal_conv.weight.squeeze(1)  # (N, T)
            temp_weight = temp_weight / temp_weight.norm(dim=1, keepdim=True)
            self.temporal_conv.weight.copy_(temp_weight.unsqueeze(1))

        self.nonlinear = nn.Softplus()

        # Lagrange multiplier for firing rate constraint (initialized to zero)
        self.register_parameter(
            "Lambda",
            nn.Parameter(torch.zeros(self.N)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (b, t, x, x)
        """
        # Reshape and apply spatial projection
        # x: (batch, time, X, X) -> (batch, time, X*X)
        x_flat = x.view([x.shape[0], -1, self.X * self.X])
        r = self.spatial_projection(x_flat)  # (batch, time, N)

        # Transpose for temporal convolution: (batch, time, N) -> (batch, N, time)
        r = r.transpose(1, 2)

        # Apply depthwise temporal convolution
        r = self.temporal_conv(r)  # (batch, N, time - T + 1)

        return self.nonlinear(r)  # (batch, N, time - T + 1)

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
        """Normalize spatial and temporal kernels to unit L2 norm per channel."""
        with torch.no_grad():
            # Normalize spatial projection weights: (N, X*X) -> normalize along dim=1
            self.spatial_projection.weight.copy_(
                self.spatial_projection.weight / self.spatial_projection.weight.norm(dim=1, keepdim=True)
            )

            # Normalize temporal convolution weights: (N, 1, T) -> squeeze, normalize, unsqueeze
            temp_weight = self.temporal_conv.weight.squeeze(1)  # (N, T)
            temp_weight = temp_weight / temp_weight.norm(dim=1, keepdim=True)
            self.temporal_conv.weight.copy_(temp_weight.unsqueeze(1))
