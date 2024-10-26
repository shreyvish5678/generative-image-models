import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x, causal_mask=False):
        input_shape = x.shape 
        
        batch_size, sequence_length, d_embed = input_shape 
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        
        weight /= math.sqrt(self.d_head) 
        weight = F.softmax(weight, dim=-1) 

        output = weight @ v
        output = output.transpose(1, 2).reshape(input_shape)  
        output = self.out_proj(output) 
        
        return output
    
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    def forward(self, x):
        residue = x 
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        
        x = x.view((n, c, h * w)).transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        
        x = x.view((n, c, h, w))
        x += residue
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        
        x = self.conv_2(x)
        return x + self.residual_layer(residue)
    
class ResidualBlockTime(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=128):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, t):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        t = F.silu(t)
        t = self.linear_time(t)
        
        merged = x + t.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_2(merged)
        merged = F.silu(merged)
        merged = self.conv_2(merged)

        return merged + self.residual_layer(residue)
    
class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)