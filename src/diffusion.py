import torch
from torch import nn
import os
from torch.nn import functional as F
from utils import *
from tqdm import tqdm

class DiffusionUNet(nn.Module):
    def __init__(self, channels=[32, 32, 64, 128, 128], depth=2, in_channels=4, n_time=32, bottleneck_blocks=2):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.encoders.append(nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1))
        out_channels = []
        out_channels.append(channels[0])
        for i in range(len(channels)-1):
            self.encoders.append(ResidualBlockTime(channels[i], channels[i+1], n_time * 4))
            out_channels.append(channels[i+1])
            for _ in range(depth-1):
                self.encoders.append(ResidualBlockTime(channels[i+1], channels[i+1], n_time * 4))
                out_channels.append(channels[i+1])
            if i != len(channels) - 2:
                self.encoders.append(nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, stride=2, padding=1))
                out_channels.append(channels[i+1])

        for _ in range(bottleneck_blocks):
            self.bottleneck.append(ResidualBlockTime(channels[-1], channels[-1], n_time * 4))
        prev_channels = channels[-1]

        for i in range(len(channels)-1, 0, -1):
            current_channels = channels[i]
            for j in range(depth+1):
                self.decoders.append(ResidualBlockTime(prev_channels+out_channels.pop(), current_channels, n_time * 4))
                prev_channels = current_channels
                if j == 2 and i != 1:
                    self.decoders.append(UpsampleBlock(current_channels))

        self.time_proj = nn.Sequential(
            nn.Linear(n_time, n_time * 4),
            nn.SiLU(),
            nn.Linear(n_time * 4, n_time * 4)
        )

        self.final_layer = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)
        )
    def forward(self, x, t):
        t = self.time_proj(t)
        skip_connections = []
        for layer in self.encoders:
            if isinstance(layer, ResidualBlockTime):
                x = layer(x, t)
            else:
                x = layer(x)
            skip_connections.append(x)
        for layer in self.bottleneck:
            x = layer(x, t)
        for layer in self.decoders:
            if isinstance(layer, ResidualBlockTime):
                x = torch.cat([x, skip_connections.pop()], dim=1)
                x = layer(x, t)
            else:
                x = layer(x)
        return self.final_layer(x)

if __name__ == '__main__':
    model = DiffusionUNet()
    x = torch.randn(1, 4, 48, 48)
    t = torch.randn(1, 32)
    y = model(x, t)
    print(y.shape)
    print(sum(p.numel() for p in model.parameters()))   