import torch
import torch.nn as nn

class FrameDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.NI = in_channels
        self.NO = out_channels

        # Use nn.Linear instead of manual parameter + matmul
        self.decoder = nn.Linear(self.NI, self.NO, bias=False)

        # Initialize with small random values
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, NI, time) -> (batch, time, NO)
        return torch.einsum('bct,oc->bto', x, self.decoder.weight)
