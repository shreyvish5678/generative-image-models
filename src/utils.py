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


def add_noise(x_0, t, alphas_cumprod, device):
    a_bar = alphas_cumprod[t].reshape(-1, 1, 1, 1)
    noise = torch.randn_like(x_0).to(device)
    x_t = x_0 * a_bar + (1 - a_bar).sqrt() * noise
    return x_t, noise

def time_embedding(t, n_time=32):
    freqs = torch.pow(10000, -torch.arange(start=0, end=n_time / 2) / (n_time / 2)) 
    x = torch.tensor([t])[:, None] * freqs[None]
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    return x

def time_embedding_2d(t, n_time=32):
    freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=n_time // 2, dtype=torch.float32) / (n_time // 2))
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if n_time % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()  
        self.norm_1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(num_heads, hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * 4)),
            nn.GELU(),
            nn.Linear(int(hidden_size * 4), hidden_size)
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(t).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x