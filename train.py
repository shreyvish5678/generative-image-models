from diffusion import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
class ImageOnlyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))
        ]
        #remove this line to use all images
        #self.image_paths = self.image_paths[:100]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageOnlyDataset("thumbnails128x128", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
config_unet = {
    "device": "cuda",
    "input_size": 128,
    "n_time": 256,
    "in_channels": 3,
    "steps": 1000,
    "betas": [0.0001, 0.04],
    "channels": [32, 32, 64, 128, 128],
    "depth": 2,
    "blocks": 2
}
model_unet = BaseDiffusionModel(config_unet)
print(model_unet.model)
print("Total Parameters:", sum(p.numel() for p in model_unet.model.parameters())) 
optimizer = torch.optim.Adam(model_unet.model.parameters(), lr=1e-4)
config = {
    "device": "cuda",
    "use_amp": True,
    "dataloader": dataloader,
    "epochs": 50,
    "optimizer": optimizer,
    "save_path": "diffusion_unet",
    "save_interval": 1
}
model_unet.train_model(config)