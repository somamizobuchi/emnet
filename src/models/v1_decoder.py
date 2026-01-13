import torch
import torch.nn as nn

class V1Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_temporal: int) -> None:
        super().__init__()
        self.NI = in_channels
        self.NO = out_channels
        self.T = n_temporal

        # Use nn.Linear for spatial projection
        self.spatial_projection = nn.Linear(self.NI, self.NO, bias=False)

        # Use nn.Conv1d for temporal filtering (depthwise convolution)
        self.temporal_conv = nn.Conv1d(
            self.NO, self.NO, kernel_size=self.T, groups=self.NO, bias=False
        )

        # Initialize with small random values
        nn.init.normal_(self.spatial_projection.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.temporal_conv.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, NI, time)
        # Spatial projection: transpose to (batch, time, NI), project, transpose back
        r = self.spatial_projection(x.transpose(1, 2))  # (batch, time, NO)
        r = r.transpose(1, 2)  # (batch, NO, time)

        # Temporal convolution (depthwise, each channel filtered independently)
        r = self.temporal_conv(r)  # (batch, NO, time - T + 1)
        return r
