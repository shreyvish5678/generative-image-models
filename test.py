from diffusion import *
import matplotlib.pyplot as plt
config_unet = {
    "device": "cuda",
    "input_size": 128,
    "n_time": 256,
    "in_channels": 3,
    "steps": 1000,
    "betas": [0.0001, 0.04],
    "channels": [32, 32, 64, 128, 128],
    "depth": 2,
    "blocks": 4
}
model_unet = BaseDiffusionModel(config_unet)
PATH = "diffusion_model_epoch_3.pt"
model_unet.model.load_state_dict(torch.load(PATH))
print("Total Parameters:", sum(p.numel() for p in model_unet.parameters())) 
sample_output = model_unet.sample(1, 1000)[0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
print(sample_output.shape)
print(sample_output.min(), sample_output.max())
plt.imshow(sample_output)
plt.axis("off")
plt.savefig("output.png")