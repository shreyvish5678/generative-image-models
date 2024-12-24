import os
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt

class BaseDiffusionModel(nn.Module):
    def __init__(self, config, unet=True):
        super().__init__()
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.steps = config["steps"]
        self.betas = torch.linspace(config["betas"][0], config["betas"][1], self.steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.model = DiffusionUNet(config) if unet else None
        self.model = self.model.to(self.device) 
        
    def save_checkpoint(self, epoch, save_path, model_name="diffusion_model"):
        checkpoint_path = os.path.join(save_path, "weights", f"{model_name}_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

    def save_samples(self, epoch, save_path, n_samples=4):
        samples = self.sample(n_samples, steps=50)
        samples = (samples + 1) / 2
        samples = samples.permute(0, 2, 3, 1).cpu().detach().numpy()
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 4, 4))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i])
            ax.axis("off")
        plt.savefig(os.path.join(save_path, "images", f"diffusion_samples_epoch_{epoch}.png"))
        plt.close()
    @torch.no_grad()
    def sample(self, n, steps=50):
        self.eval()
        outputs = torch.randn((n, self.model.in_channels, self.model.in_size, self.model.in_size)).to(self.device)
        ddim_steps = torch.linspace(0, self.steps - 1, steps, dtype=torch.long).to(self.device)
        alphas_cumprod_t = self.alphas_cumprod[ddim_steps]
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod_t[:-1]])
        for i in tqdm(reversed(range(steps)), desc="Sampling", total=steps):
            t = ddim_steps[i].repeat(n)
            alpha_t = alphas_cumprod_t[i]
            alpha_prev = alphas_cumprod_prev[i]
            beta_t = 1 - alpha_t
            noise = torch.randn_like(outputs) if i > 0 else 0
            time_embedding = get_time_embedding(t, n_time=self.model.n_time).to(self.device)
            pred_noise = self.model(outputs, time_embedding)
            x0_pred = (outputs - beta_t.sqrt() * pred_noise) / alpha_t.sqrt()
            outputs = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise
            outputs = torch.clamp(outputs, -1, 1)
        return outputs

    def train_model(self, config):
        dataloader = config["dataloader"]
        epochs = config["epochs"]
        optimizer = config["optimizer"]
        save_path = config["save_path"]
        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "weights"), exist_ok=True)
        save_interval = config["save_interval"]
        scheduler = config.get("scheduler", None)
        use_amp = config.get("use_amp", False)
        scaler = torch.GradScaler(self.device, enabled=use_amp)
        os.makedirs(save_path, exist_ok=True)
        self.model.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False) as pbar:
                for batch in pbar:
                    batch = batch.to(self.device)
                    t = torch.randint(0, self.steps, (batch.shape[0],), dtype=torch.long).to(self.device)
                    x_t, noise = add_noise(batch, t, self.alphas_cumprod, self.device)
                    time_embedding = get_time_embedding(t, n_time=self.model.n_time).to(self.device)
                    optimizer.zero_grad()
                    with torch.autocast(self.device, enabled=use_amp):
                        noise_pred = self.model(x_t, time_embedding)
                        loss = F.mse_loss(noise_pred, noise)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                self.save_checkpoint(epoch + 1, save_path)
                self.save_samples(epoch + 1, save_path)


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.in_size = config["input_size"]
        self.n_time = config["n_time"]
        n_time = config["n_time"]   
        self.in_channels = config["in_channels"]
        in_channels = config["in_channels"]
        channels = config["channels"]
        depth = config["depth"]
        blocks = config["blocks"]
        self.encoders = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # Encoder
        self.encoders.append(nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1))
        skip_connections = []
        for i in range(len(channels) - 1):
            self.encoders.append(ResidualBlock(channels[i], channels[i+1], n_time * 4))
            skip_connections.append(channels[i+1])
            for _ in range(depth - 1):
                self.encoders.append(ResidualBlock(channels[i+1], channels[i+1], n_time * 4))
                skip_connections.append(channels[i+1])
            if i != len(channels) - 2:
                self.encoders.append(nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, stride=2, padding=1))
                skip_connections.append(channels[i+1])

        # Bottleneck
        for _ in range(blocks):
            self.bottleneck.append(ResidualBlock(channels[-1], channels[-1], n_time * 4))

        # Decoder
        skip_idx = len(skip_connections) - 1
        prev_channels = channels[-1]
        for i in range(len(channels) - 1, 0, -1):
            current_channels = channels[i - 1]
            for j in range(depth):
                self.decoders.append(ResidualBlock(prev_channels + skip_connections[skip_idx], current_channels, n_time * 4))
                prev_channels = current_channels
                skip_idx -= 1
            if i != 1:  # Add upsampling except for the last layer
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
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            skip_connections.append(x)
        for layer in self.bottleneck:
            x = layer(x, t)
        for layer in self.decoders:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skip_connections.pop()], dim=1)
                x = layer(x, t)
            else:
                x = layer(x)
        return self.final_layer(x)

if __name__ == "__main__":
    config_unet = {
        "device": "cuda",
        "input_size": 64,
        "n_time": 64,
        "in_channels": 3,
        "steps": 1000,
        "betas": [0.0001, 0.04],
        "channels": [32, 32, 64, 128, 128],
        "depth": 2,
        "blocks": 2
    }
    model_unet = DiffusionUNet(config_unet)
    print(sum(p.numel() for p in model_unet.parameters())) 
    output = model_unet.sample(1, 50)
    print(output.shape)