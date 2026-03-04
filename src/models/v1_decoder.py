import torch
import torch.nn as nn
import torch.nn.functional as F


class V1Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_temporal: int,
        delay: int = 0,
    ) -> None:
        """
        Args:
            in_channels  : Number of input (RGC) channels.
            out_channels : Number of V1 channels.
            n_temporal   : Total kernel length including delay (samples).
            delay        : Trailing zero samples enforcing a causal delay.
        """
        super().__init__()
        self.NI = in_channels
        self.NO = out_channels
        self.T = n_temporal
        self.D = delay
        self.T_train = n_temporal - delay
        assert self.T_train >= 1, "n_temporal must be greater than delay"

        # ------------------------------------------------------------------
        # Temporal parameterisation (raw taps)
        # ------------------------------------------------------------------
        self.temporal_weight = nn.Parameter(  # (NO, 1, T_train), learned
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
        return self.temporal_weight.squeeze(1)

    def _full_kernel(self) -> torch.Tensor:
        """Return the full (NO, 1, T) kernel with trailing zeros for the delay.

        F.conv1d uses cross-correlation (no kernel flip). The kernel index 0
        corresponds to the oldest input sample in the receptive field. Appending
        D zeros at the end therefore forces the output to ignore the D most
        recent samples, implementing a causal delay of D steps.
        """
        taps = self.temporal_weight  # (NO, 1, T_train)
        if self.D == 0:
            return taps
        zeros = torch.zeros(self.NO, 1, self.D, device=taps.device, dtype=taps.dtype)
        return torch.cat([taps, zeros], dim=2)  # (NO, 1, T)

    @property
    def temporal_weights(self) -> torch.Tensor:
        """Get full temporal weights (NO, T) including trailing delay zeros."""
        return self._full_kernel().squeeze(1).detach()

    def compute_temporal_smoothness_loss(self) -> torch.Tensor:
        """Second-derivative penalty on raw taps."""
        taps = self._trainable_taps()  # (NO, T_train)
        d2 = taps[:, :-2] - 2 * taps[:, 1:-1] + taps[:, 2:]
        return (d2**2).mean()

    def normalize_kernels(self):
        """Normalize the projected temporal kernels to unit L2 norm per channel."""
        with torch.no_grad():
            taps = self._trainable_taps()  # (NO, T_train)
            tap_norm = taps.norm(dim=1, keepdim=True)  # (NO, 1)
            self.temporal_weight.copy_((taps / (tap_norm + 1e-8)).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, NI, time)
        # einsum avoids two transposes: (batch, NI, time) -> (batch, NO, time)
        r = torch.einsum('bct,oc->bot', x, self.spatial_projection.weight)
        kernel = self._full_kernel()  # (NO, 1, T)
        return F.softplus(
            F.conv1d(r, kernel, groups=self.NO)
        )  # (batch, NO, time - T + 1)
