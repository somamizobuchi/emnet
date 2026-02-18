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
            spatial_norm = self.spatial_projection.weight.norm(dim=1, keepdim=True)
            self.spatial_projection.weight.copy_(
                self.spatial_projection.weight
                / (spatial_norm + 1e-8)
            )
            # Temporal: weight is (N, 1, T), squeeze, normalize, unsqueeze
            temp_weight = self.temporal_conv.weight.squeeze(1)  # (N, T)
            temporal_norm = temp_weight.norm(dim=1, keepdim=True)
            temp_weight = temp_weight / (temporal_norm + 1e-8)
            self.temporal_conv.weight.copy_(temp_weight.unsqueeze(1))

        self.nonlinear = nn.Softplus(beta=2.5)

        # Gain and bias for each channel (biologically: metabolic tuning)
        self.log_gain = nn.Parameter(torch.zeros(self.N))
        self.log_bias = nn.Parameter(torch.ones(self.N) * -1.0)

        # Lagrange multiplier for firing rate constraint (initialized to zero)
        # Registered as buffer so it's not updated by the primal optimizer
        self.register_buffer(
            "lagrange_multiplier",
            torch.zeros(self.N),
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

        # Apply gain and bias before nonlinearity
        # r: (batch, N, time_reduced), log_gain/log_bias: (N)
        # Clamp to prevent exponential overflow (exp(88) ≈ 1e38, safe for float32)
        gain = torch.clamp(self.log_gain, min=-10, max=10).exp().view(1, -1, 1)
        bias = torch.clamp(self.log_bias, min=-10, max=10).exp().view(1, -1, 1)

        return gain * self.nonlinear(r + bias)  # (batch, N, time - T + 1)

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
        quadratic_penalty = self.rho / 2.0 * (constraint_violation**2).sum()

        # Linear penalty term: λᵀ · h
        lm = self.lagrange_multiplier
        assert isinstance(lm, torch.Tensor)
        linear_penalty = (lm * constraint_violation).sum()

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
            lagrange_multiplier = self.lagrange_multiplier
            lagrange_multiplier.add_(self.rho * constraint_violation)

    @property
    def spatial_weights(self) -> torch.Tensor:
        """Get spatial weights reshaped to (N, X, X)."""
        return self.spatial_projection.weight.view(self.N, self.X, self.X)

    @property
    def temporal_weights(self) -> torch.Tensor:
        """Get temporal weights squeezed to (N, T)."""
        return self.temporal_conv.weight.squeeze(1)

    @property
    def Lambda(self) -> torch.Tensor:
        """Access Lagrange multipliers."""
        lm = self.lagrange_multiplier
        assert isinstance(lm, torch.Tensor)
        return lm

    def normalize_kernels(self):
        """Normalize spatial and temporal kernels to unit L2 norm per channel."""
        with torch.no_grad():
            # Normalize spatial projection weights: (N, X*X) -> normalize along dim=1
            # Add epsilon to prevent division by zero
            spatial_norm = self.spatial_projection.weight.norm(dim=1, keepdim=True)
            self.spatial_projection.weight.copy_(
                self.spatial_projection.weight
                / (spatial_norm + 1e-8)
            )

            # Normalize temporal convolution weights: (N, 1, T) -> squeeze, normalize, unsqueeze
            temp_weight = self.temporal_conv.weight.squeeze(1)  # (N, T)
            temporal_norm = temp_weight.norm(dim=1, keepdim=True)
            temp_weight = temp_weight / (temporal_norm + 1e-8)
            self.temporal_conv.weight.copy_(temp_weight.unsqueeze(1))

    def check_for_nans(self) -> dict[str, bool]:
        """
        Check all parameters and buffers for NaN values.

        Returns:
            dict: Dictionary mapping parameter names to boolean (True if contains NaN)
        """
        nan_status = {}
        for name, param in self.named_parameters():
            nan_status[name] = torch.isnan(param).any().item()
        for name, buf in self.named_buffers():
            nan_status[name] = torch.isnan(buf).any().item()
        return nan_status

    def compute_spatial_variance(self) -> torch.Tensor:
        """
        Compute the spatial variance (localization) of the kernels.
        Higher variance means the kernel is more spread out.

        Returns:
            torch.Tensor: Mean spatial variance across all channels (scalar)
        """
        # W shape: (N, X*X)
        W = self.spatial_projection.weight
        # Ensure we use the normalized version for variance calculation
        W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        # We look at the "mass" of the squared weights: (N, X, X)
        W_sq = W_norm.view(self.N, self.X, self.X).pow(2)

        # Pixel coordinates
        coords = torch.arange(self.X, device=W.device, dtype=W.dtype)

        # Marginal distributions (N, X)
        Wx = W_sq.sum(dim=1)  # Marginalize over Y to get X distribution
        Wy = W_sq.sum(dim=2)  # Marginalize over X to get Y distribution

        # Normalize marginals to sum to 1 (per channel) to treat as probability distribution
        Wx_sum = Wx.sum(dim=1, keepdim=True) + 1e-8
        Wy_sum = Wy.sum(dim=1, keepdim=True) + 1e-8
        Wx_p = Wx / Wx_sum
        Wy_p = Wy / Wy_sum

        # Center of mass (N,)
        mean_x = (Wx_p * coords.view(1, -1)).sum(dim=1)
        mean_y = (Wy_p * coords.view(1, -1)).sum(dim=1)

        # Variance (moment of inertia) around the center of mass
        var_x = (Wx_p * (coords.view(1, -1) - mean_x.view(-1, 1)) ** 2).sum(dim=1)
        var_y = (Wy_p * (coords.view(1, -1) - mean_y.view(-1, 1)) ** 2).sum(dim=1)

        return (var_x + var_y).mean()
