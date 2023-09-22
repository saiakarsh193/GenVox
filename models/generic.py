import torch
from torch import nn
from typing import Optional

class LinearNorm(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            w_init_gain: str = "linear"
        ):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: Optional[int] = None,
            dilation: int = 1,
            bias: bool = True,
            w_init_gain: str = "linear"
        ):
        super().__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        nn.init.xavier_uniform_(
            self.conv.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.conv(x)
