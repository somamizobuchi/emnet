import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.constraints import make_raised_cosine_basis


class V1Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_temporal: int,
        delay: int = 0,
        n_basis: int = 0,
        log_offset: float = 1.0,
    ) -> None:
        """
        Args:
            in_channels  : Number of input (RGC) channels.
            out_channels : Number of V1 channels.
            n_temporal   : Total kernel length including delay (samples).
            delay        : Trailing zero samples enforcing a causal delay.
            n_basis      : Raised-cosine basis functions.
                           0 (default) = raw tap parameterisation.
                           >0          = basis parameterisation; the kernel
                           is always B @ coeffs, guaranteeing smooth kernels.
            log_offset   : Log-compression for the raised cosine basis
                           (default 1.0; larger → more linear spacing).
        """
        super().__init__()
        self.NI = in_channels
        self.NO = out_channels
        self.T = n_temporal
        self.D = delay
        self.T_train = n_temporal - delay
        assert self.T_train >= 1, "n_temporal must be greater than delay"

        # ------------------------------------------------------------------
        # Temporal parameterisation
        # ------------------------------------------------------------------
        if n_basis > 0:
            assert n_basis <= self.T_train, "n_basis must be <= n_temporal - delay"
            self.n_basis = n_basis
            basis = make_raised_cosine_basis(self.T_train, n_basis, log_offset)
            self.register_buffer("basis", basis)          # (T_train, n_basis), fixed
            self.temporal_coeffs = nn.Parameter(          # (NO, n_basis), learned
                torch.empty(self.NO, n_basis)
            )
            nn.init.normal_(self.temporal_coeffs, mean=0.0, std=0.01)
        else:
            self.n_basis = 0
            self.temporal_weight = nn.Parameter(          # (NO, 1, T_train), learned
                torch.empty(self.NO, 1, self.T_train)
            )
            nn.init.normal_(self.temporal_weight, mean=0.0, std=0.01)

        # ------------------------------------------------------------------
        # Spatial projection
        # ------------------------------------------------------------------
        self.spatial_projection = nn.Linear(self.NI, self.NO, bias=False)
        nn.init.normal_(self.spatial_projection.weight, mean=0.0, std=0.01)

    def _trainable_taps(self) -> torch.Tensor:
        """Return the (NO, T_train) trainable portion of the kernel."""
        if self.n_basis > 0:
            return self.temporal_coeffs @ self.basis.T
        else:
            return self.temporal_weight.squeeze(1)

    def _full_kernel(self) -> torch.Tensor:
        """Return the full (NO, 1, T) kernel with trailing zeros for the delay.

        F.conv1d uses cross-correlation (no kernel flip). The kernel index 0
        corresponds to the oldest input sample in the receptive field. Appending
        D zeros at the end therefore forces the output to ignore the D most
        recent samples, implementing a causal delay of D steps.
        """
        taps = self._trainable_taps().unsqueeze(1)  # (NO, 1, T_train)
        if self.D == 0:
            return taps
        zeros = torch.zeros(self.NO, 1, self.D, device=taps.device, dtype=taps.dtype)
        return torch.cat([taps, zeros], dim=2)  # (NO, 1, T)

    @property
    def temporal_weights(self) -> torch.Tensor:
        """Get full temporal weights (NO, T) including trailing delay zeros."""
        return self._full_kernel().squeeze(1).detach()

    def normalize_kernels(self):
        """Normalize the projected temporal kernels to unit L2 norm per channel."""
        with torch.no_grad():
            taps = self._trainable_taps()              # (NO, T_train)
            tap_norm = taps.norm(dim=1, keepdim=True)  # (NO, 1)
            if self.n_basis > 0:
                self.temporal_coeffs.copy_(
                    self.temporal_coeffs / (tap_norm + 1e-8)
                )
            else:
                self.temporal_weight.copy_(
                    (taps / (tap_norm + 1e-8)).unsqueeze(1)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, NI, time)
        r = self.spatial_projection(x.transpose(1, 2))  # (batch, time, NO)
        r = r.transpose(1, 2)                           # (batch, NO, time)
        kernel = self._full_kernel()                    # (NO, 1, T)
        return F.conv1d(r, kernel, groups=self.NO)      # (batch, NO, time - T + 1)
