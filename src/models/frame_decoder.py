import torch
import torch.nn as nn

class FrameDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.NI = in_channels
        self.NO = out_channels

        self.decoder = nn.Parameter(torch.zeros([self.NI, self.NO]).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.matmul(x.transpose(1,2), self.decoder)
        return r
