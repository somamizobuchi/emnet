import torch
import torch.nn as nn
import torch.nn.functional as F

class V1Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_temporal: int) -> None:
        super().__init__()
        self.NI = in_channels
        self.NO = out_channels
        self.T = n_temporal

        self.spatial_kernels = nn.Parameter(torch.zeros([self.NI, self.NO]).float())
        self.temporal_kernels = nn.Parameter(torch.zeros([self.NO, self.T]).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.matmul(x.transpose_(1, 2), self.spatial_kernels)
        r = F.conv1d(r.transpose_(1,2), self.temporal_kernels.unsqueeze(1), groups=self.NO)
        return r
