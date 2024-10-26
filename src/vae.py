import torch
from torch import nn
import os
from torch.nn import functional as F
from src.utils import *
from tqdm import tqdm
    
class Decoder(nn.Module):
    def __init__(self, channels=[128, 128, 64, 32], depth=3, in_channels=4, out_channels=3):
        super().__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0))
        self.model.append(nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1))
        self.model.append(ResidualBlock(channels[0], channels[0]))  
        self.model.append(AttentionBlock(channels[0]))
        for i in range(len(channels)-1):
            self.model.append(ResidualBlock(channels[i], channels[i+1]))
            for _ in range(depth-1):
                self.model.append(ResidualBlock(channels[i+1], channels[i+1]))
            if i == len(channels) - 2:
                self.model.append(nn.Upsample(scale_factor=2))
                self.model.append(AttentionBlock(channels[i+1]))
                self.model.append(ResidualBlock(channels[i+1], channels[i+1]))
                self.model.append(nn.GroupNorm(32, channels[i+1]))
                self.model.append(nn.SiLU())
                self.model.append(nn.Conv2d(channels[i+1], out_channels, kernel_size=3, padding=1))
            else:
                self.model.append(nn.Upsample(scale_factor=2))
                self.model.append(nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1))
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, channels=[32, 64, 128, 128], depth=2, in_channels=3, out_channels=4, clamp_bound=10):
        super().__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1))
        for i in range(len(channels)-1):
            self.model.append(ResidualBlock(channels[i], channels[i+1]))
            for _ in range(depth-1):
                self.model.append(ResidualBlock(channels[i+1], channels[i+1]))
            if i == len(channels) - 2:
                self.model.append(nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, stride=2, padding=0))
                self.model.append(AttentionBlock(channels[i+1]))
                self.model.append(ResidualBlock(channels[i+1], channels[i+1]))
                self.model.append(nn.GroupNorm(32, channels[i+1]))
                self.model.append(nn.SiLU())
                self.model.append(nn.Conv2d(channels[i+1], out_channels*2, kernel_size=3, padding=1))
            else:
                self.model.append(nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, stride=2, padding=0))
        self.model.append(nn.Conv2d(out_channels*2, out_channels*2, kernel_size=1, padding=0))
        self.clamp_bound = clamp_bound
    def forward(self, x):
        for layer in self.model:
            if getattr(layer, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = layer(x)
        mean, logvar = x.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -self.clamp_bound, self.clamp_bound)
        stdev = (0.5 * logvar).exp()
        noise = torch.randn_like(mean)
        z = mean + stdev * noise
        return z, mean, logvar
    
class VAE(nn.Module):
    def __init__(self, encoder_channels=[32, 64, 128, 128], encoder_depth=2, decoder_channels=[128, 128, 64, 32], decoder_depth=3, in_channels=3, out_channels=3, clamp_bound=10, in_size=192):
        super().__init__()
        self.encoder = Encoder(encoder_channels, encoder_depth, in_channels, out_channels, clamp_bound)
        self.decoder = Decoder(decoder_channels, decoder_depth, out_channels, in_channels)
        self.latent_size = (1, out_channels, in_size // 2**(len(encoder_channels) - 1), in_size // 2**(len(encoder_channels) - 1))

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x, x_recon, mean, logvar
    
    def sample(self, n, latent=None):
        if latent is None:
            latent = torch.randn(n, self.latent_size[1], self.latent_size[2], self.latent_size[3])
        return self.decoder(latent)
    
    def vae_loss(self, x, x_recon, mean, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_loss
    
    def train_vae(self, config):
        device = config["device"] if "device" in config else "cuda" if torch.cuda.is_available() else "cpu"
        model = self.to(device)
        dataloader = config["dataloader"]
        epochs = config["epochs"]
        optimizer = config["optimizer"]
        save_path = config["save_path"]
        save_interval = config["save_interval"]
        os.makedirs(save_path, exist_ok=True)   
        model.train()  
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                x = batch.to(device)
                optimizer.zero_grad()
                
                x, x_recon, mean, logvar = model(x)
                
                loss = self.vae_loss(x, x_recon, mean, logvar)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                checkpoint_path = f"{save_path}/vae_epoch_{epoch+1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved at {checkpoint_path}")