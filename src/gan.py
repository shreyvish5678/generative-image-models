import torch
from torch import nn
import os
from torch.nn import functional as F
from utils import *
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, channels=[256, 128, 64, 32], depth=2, channel=1, noise_vector=128, in_size=192):
        super().__init__()
        self.model = nn.ModuleList()
        self.input_size = in_size // 2 ** (len(channels) - 1)
        self.in_channels = channels[0] // 16
        self.proj = nn.Sequential(
            nn.Linear(noise_vector, self.in_channels * self.input_size ** 2, bias=False),
            nn.LayerNorm(self.in_channels * self.input_size ** 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model.append(nn.Conv2d(self.in_channels, channels[0], kernel_size=3, padding=1))  
        self.model.append(AttentionBlock(channels[0]))
        for i in range(len(channels)-1):
            self.model.append(nn.Upsample(scale_factor=2))
            self.model.append(ResidualBlock(channels[i], channels[i+1]))
            for _ in range(depth-1):
                self.model.append(ResidualBlock(channels[i+1], channels[i+1]))
            self.model.append(nn.BatchNorm2d(channels[i+1]))
            self.model.append(nn.LeakyReLU(0.2, inplace=True))
        self.model.append(nn.Conv2d(channels[-1], channel, kernel_size=3, padding=1))
    def forward(self, x):
        x = self.proj(x).view(-1, self.in_channels, self.input_size, self.input_size)
        for layer in self.model:
            x = layer(x)
        return x
        
class Discriminator(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], depth=1, channel=1, in_size=192, dropout=0.2, sigmoid=True):
        super().__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Conv2d(channel, channels[0], kernel_size=3, padding=1))
        for i in range(len(channels)-1):
            self.model.append(nn.MaxPool2d(2, 2))
            self.model.append(ResidualBlock(channels[i], channels[i+1]))
            for _ in range(depth-1):
                self.model.append(ResidualBlock(channels[i+1], channels[i+1]))
            self.model.append(nn.LeakyReLU(0.2, inplace=True))
            self.model.append(nn.Dropout(dropout))
        self.proj = nn.Linear(channels[-1] * (in_size // 2 ** (len(channels) - 1)) ** 2, 1)
        self.sigmoid = sigmoid
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        if self.sigmoid:
            x = F.sigmoid(x)
        return x
    

class GAN(nn.Module):
    def __init__(self, generator_channels=[256, 128, 64, 32], discriminator_channels=[32, 64, 128, 256], generator_depth=2, discriminator_depth=1, channel=1, noise_vector=128, in_size=192, dropout=0.2):
        super().__init__()
        self.generator = Generator(generator_channels, generator_depth, channel, noise_vector, in_size)
        self.discriminator = Discriminator(discriminator_channels, discriminator_depth, channel, in_size, dropout)
        self.noise_size = noise_vector

    def disc_loss(self, real, fake):
        real_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
        fake_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        return real_loss + fake_loss
    
    def gen_loss(self, fake):
        return F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
    
    def sample(self, n):
        return self.generator(torch.randn(n, self.noise_size))
    
    def train_gan(self, config):
        device = config["device"] if "device" in config else "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        dataloader = config["dataloader"]
        epochs = config["epochs"]
        gen_optimizer = config["gen_optimizer"]
        disc_optimizer = config["disc_optimizer"]
        save_path = config["save_path"]
        save_interval = config["save_interval"]
        os.makedirs(save_path, exist_ok=True)   
        self.generator.train()
        self.discriminator.train()
        for epoch in range(epochs):
            gen_loss = 0.0
            disc_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                real = batch.to(device)
                disc_optimizer.zero_grad()

                noise = torch.randn(len(real), self.noise_size).to(device)
                fake = self.generator(noise)
                real_pred = self.discriminator(real)
                fake_pred = self.discriminator(fake.detach())

                disc_loss = self.disc_loss(real_pred, fake_pred)
                disc_loss.backward()
                disc_optimizer.step()

                gen_optimizer.zero_grad()
                with torch.no_grad():
                    fake_pred = self.discriminator(fake)
                gen_loss = self.gen_loss(fake_pred)
                gen_loss.backward()
                gen_optimizer.step()

                gen_loss += gen_loss.item()
                disc_loss += disc_loss.item()

            avg_gen_loss = gen_loss / len(dataloader)
            avg_disc_loss = disc_loss / len(dataloader)

            print(f"Epoch {epoch+1}/{epochs} Generator Loss: {avg_gen_loss} Discriminator Loss: {avg_disc_loss}")
            
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                gen_checkpoint_path = f"{save_path}/gen_epoch_{epoch+1}.pt"
                disc_checkpoint_path = f"{save_path}/disc_epoch_{epoch+1}.pt"
                torch.save(self.generator.state_dict(), gen_checkpoint_path)
                torch.save(self.discriminator.state_dict(), disc_checkpoint_path)
                print(f"Model saved at {gen_checkpoint_path}")
                print(f"Model saved at {disc_checkpoint_path}")

class WGAN:
    def __init__(self, generator_channels=[256, 128, 64, 32], discriminator_channels=[32, 64, 128, 256], generator_depth=2, discriminator_depth=1, channel=1, noise_vector=128, in_size=192, dropout=0.2, gp_weight=10, every_n_critic=5):
        super().__init__()
        self.generator = Generator(generator_channels, generator_depth, channel, noise_vector, in_size)
        self.discriminator = Discriminator(discriminator_channels, discriminator_depth, channel, in_size, dropout, sigmoid=False)
        self.noise_size = noise_vector
        self.gp_weight = gp_weight
        self.every_n_critic = every_n_critic

    def disc_loss(self, real, fake):
        return fake.mean() - real.mean()
    
    def gen_loss(self, fake):
        return -fake.mean()
    
    def sample(self, n):
        return self.generator(torch.randn(n, self.noise_size))
    
    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1).to(real.device)

        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)
        interpolated_pred = self.discriminator(interpolated)

        grad = torch.autograd.grad(outputs=interpolated_pred, inputs=interpolated, grad_outputs=torch.ones_like(interpolated_pred), create_graph=True, retain_graph=True)[0]
        grad = grad.view(batch_size, -1)
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()
    
    def train_wgan(self, config):
        device = config["device"] if "device" in config else "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        dataloader = config["dataloader"]
        epochs = config["epochs"]
        gen_optimizer = config["gen_optimizer"]
        disc_optimizer = config["disc_optimizer"]
        save_path = config["save_path"]
        save_interval = config["save_interval"]
        os.makedirs(save_path, exist_ok=True)   
        self.generator.train()
        self.discriminator.train()
        for epoch in range(epochs):
            gen_loss = 0.0
            disc_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                real = batch.to(device)
                for _ in range(self.every_n_critic):
                    noise = torch.randn(len(real), self.noise_size).to(device)
                    with torch.no_grad():
                        fake = self.generator(noise)
                    disc_optimizer.zero_grad()

                    real_pred = self.discriminator(real)
                    fake_pred = self.discriminator(fake)
                    gp = self.gradient_penalty(real, fake)

                    disc_loss = self.disc_loss(real_pred, fake_pred) + gp * self.gp_weight
                    disc_loss.backward()
                    disc_optimizer.step()

                    disc_loss += disc_loss.item() / self.every_n_critic

                gen_optimizer.zero_grad()
                noise = torch.randn(len(real), self.noise_size).to(device)
                fake = self.generator(noise)

                with torch.no_grad():
                    fake_pred = self.discriminator(fake)
                gen_loss = self.gen_loss(fake_pred)

                gen_loss.backward()
                gen_optimizer.step()
                gen_loss += gen_loss.item()

            avg_gen_loss = gen_loss / len(dataloader)
            avg_disc_loss = disc_loss / len(dataloader)

            print(f"Epoch {epoch+1}/{epochs} Generator Loss: {avg_gen_loss} Discriminator Loss: {avg_disc_loss}")
            
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                gen_checkpoint_path = f"{save_path}/gen_epoch_{epoch+1}.pt"
                disc_checkpoint_path = f"{save_path}/disc_epoch_{epoch+1}.pt"
                torch.save(self.generator.state_dict(), gen_checkpoint_path)
                torch.save(self.discriminator.state_dict(), disc_checkpoint_path)
                print(f"Model saved at {gen_checkpoint_path}")
                print(f"Model saved at {disc_checkpoint_path}")

if __name__ == '__main__':
    model = GAN()
    x = torch.randn(1, 128)
    output = model.generator(x)
    print(output.shape)
    pred = model.discriminator(output)
    print(pred.shape)
    print(sum(p.numel() for p in model.generator.parameters()))
    print(sum(p.numel() for p in model.discriminator.parameters()))