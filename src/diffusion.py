import torch
from torch import nn
import os
from torch.nn import functional as F
from utils import *
from tqdm import tqdm

class DiffusionUNet(nn.Module):
    def __init__(self, channels=[32, 32, 64, 128, 128], depth=2, in_channels=4, n_time=32, bottleneck_blocks=2, steps=1000, betas=[0.0001, 0.04]):
        super().__init__()
        self.steps = steps
        self.betas = betas  
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
    
    
    def train_diffusion(self, config):
        device = config["device"] if "device" in config else "cuda" if torch.cuda.is_available() else "cpu"
        model = self.to(device)
        dataloader = config["dataloader"]
        epochs = config["epochs"]
        optimizer = config["optimizer"]
        save_path = config["save_path"]
        save_interval = config["save_interval"]
        self.betas = torch.linspace(self.betas[0], self.betas[1], self.steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        os.makedirs(save_path, exist_ok=True)   
        model.train() 
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = batch.to(device)
                t = torch.randint(0, self.steps, (batch.shape[0],), dtype=torch.long).to(device)
                x_t, noise = add_noise(batch, t, self.alphas_cumprod, device)
                time_embedding = time_embedding(t, n_time=32).to(device)
                optimizer.zero_grad()
                noise_pred = model(x_t, time_embedding)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                checkpoint_path = f"{save_path}/diff_epoch_{epoch+1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved at {checkpoint_path}")

class DiffusionTransformer(nn.Module):
    def __init__(self, in_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, heads=16, mlp_ratio=4, class_dropout=0.1, num_classes=1000, learn_sigma=True):
        pass

if __name__ == '__main__':
    model = DiffusionUNet(n_time=48)
    x = torch.randn(1, 4, 48, 48)
    t = torch.randn(1, 48)
    y = model(x, t)
    print(y.shape)
    print(sum(p.numel() for p in model.parameters()))   