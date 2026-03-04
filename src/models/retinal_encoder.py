import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinalEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_spatial: int,
        n_temporal: int,
        delay: int = 0,
        target_firing_rate: float = 1.0,
        rho: float = 1.0,
    ) -> None:
        """
        Args:
            n_channels         : Number of RGC channels.
            n_spatial          : Spatial ROI side length (pixels).
            n_temporal         : Total kernel length including delay (samples).
            delay              : Trailing zero samples enforcing a causal delay.
            target_firing_rate : Target mean firing rate for ALM constraint.
            rho                : ALM penalty parameter.
        """
        super().__init__()
        self.N = n_channels
        self.T = n_temporal
        self.D = delay
        self.T_train = n_temporal - delay
        assert self.T_train >= 1, "n_temporal must be greater than delay"
        self.X = n_spatial
        self.target_firing_rate = target_firing_rate
        self.rho = rho

        # ------------------------------------------------------------------
        # Temporal parameterisation (raw taps)
        # ------------------------------------------------------------------
        self.temporal_weight = nn.Parameter(          # (N, 1, T_train), learned
            torch.empty(self.N, 1, self.T_train)
        )
        nn.init.normal_(self.temporal_weight, mean=0.0, std=0.01)
        with torch.no_grad():
            temp = self.temporal_weight.squeeze(1)
            self.temporal_weight.copy_(
                (temp / (temp.norm(dim=1, keepdim=True) + 1e-8)).unsqueeze(1)
            )

        # ------------------------------------------------------------------
        # Spatial projection
        # ------------------------------------------------------------------
        self.spatial_projection = nn.Linear(self.X * self.X, self.N, bias=False)
        nn.init.normal_(self.spatial_projection.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            spatial_norm = self.spatial_projection.weight.norm(dim=1, keepdim=True)
            self.spatial_projection.weight.copy_(
                self.spatial_projection.weight / (spatial_norm + 1e-8)
            )

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

    def _trainable_taps(self) -> torch.Tensor:
        """Return the (N, T_train) trainable portion of the kernel."""
        return self.temporal_weight.squeeze(1)  # (N, T_train)

    def _full_kernel(self) -> torch.Tensor:
        """Return the full (N, 1, T) kernel with trailing zeros for the delay.

        F.conv1d uses cross-correlation (no kernel flip). The kernel index 0
        corresponds to the oldest input sample in the receptive field. Appending
        D zeros at the end therefore forces the output to ignore the D most
        recent samples, implementing a causal delay of D steps.
        """
        taps = self.temporal_weight  # (N, 1, T_train)
        if self.D == 0:
            return taps
        zeros = torch.zeros(self.N, 1, self.D, device=taps.device, dtype=taps.dtype)
        return torch.cat([taps, zeros], dim=2)  # (N, 1, T)

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

        # Depthwise conv with the full (delay-prepended) kernel
        kernel = self._full_kernel()  # (N, 1, T)
        r = F.conv1d(r, kernel, groups=self.N)  # (batch, N, time - T + 1)

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
        """Get full temporal weights (N, T) including leading delay zeros."""
        return self._full_kernel().squeeze(1).detach()

    @property
    def Lambda(self) -> torch.Tensor:
        """Access Lagrange multipliers."""
        lm = self.lagrange_multiplier
        assert isinstance(lm, torch.Tensor)
        return lm

    def compute_temporal_smoothness_loss(self) -> torch.Tensor:
        """Second-derivative penalty on raw taps."""
        taps = self._trainable_taps()  # (N, T_train)
        d2 = taps[:, :-2] - 2 * taps[:, 1:-1] + taps[:, 2:]
        return (d2**2).mean()

    def normalize_kernels(self):
        """Normalize spatial and temporal kernels to unit L2 norm per channel."""
        with torch.no_grad():
            # Spatial
            spatial_norm = self.spatial_projection.weight.norm(dim=1, keepdim=True)
            self.spatial_projection.weight.copy_(
                self.spatial_projection.weight / (spatial_norm + 1e-8)
            )

            # Temporal
            taps = self._trainable_taps()          # (N, T_train)
            tap_norm = taps.norm(dim=1, keepdim=True)  # (N, 1)
            self.temporal_weight.copy_(
                (taps / (tap_norm + 1e-8)).unsqueeze(1)
            )

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
        W_sq = W.view(self.N, self.X, self.X).pow(2)

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
