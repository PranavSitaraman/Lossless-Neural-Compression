import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from PIL import Image
from convae import ConvAutoencoder
from config import config
from tqdm import tqdm
import zlib
from io import BytesIO
import struct

# Ensure experiment directories exist
os.makedirs(config.experiment_root, exist_ok=True)
os.makedirs(config.results_dir, exist_ok=True)
os.makedirs(config.logs_dir, exist_ok=True)

# Define transformations for ImageNet images
transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor()
])

# Load a subset of ImageNet (Tiny ImageNet or a sampled subset)
imagenet_dataset = torchvision.datasets.ImageFolder(root=config.data_root, transform=transform)
subset_indices = list(range(5000, 10000))  # Select a smaller subset for evaluating
test_dataset = Subset(imagenet_dataset, subset_indices)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

# Initialize the autoencoder
model = ConvAutoencoder(image_size=config.image_size)
model_path = config.model_path
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print("Loaded pre-trained model.")
else:
    print("No pre-trained model found.")

def compress_png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def encode_images(images, model):
    model.eval()

    with torch.no_grad():
        latents = torch.zeros((len(images), 32768)) # TODO
        reconstructions = torch.zeros(images.shape) # TODO
    
    differences = (images.numpy() * 255).clip(0, 255).astype(np.int16) \
                - (reconstructions.numpy() * 255).clip(0, 255).astype(np.int16)
    signs = (differences < 0)
    differences = np.abs(differences).astype(np.uint8)
    compressed = []
    for i in range(len(images)):
        compressed_latent = zlib.compress(latents[i].numpy().tobytes(), level=config.compression_quality)
        diff_image = Image.fromarray(differences[i].transpose(1, 2, 0))
        compressed_difference = compress_png(diff_image)
        compressed_sign = zlib.compress(signs[i].tobytes(), level=config.compression_quality)
        compressed.append((len(compressed_latent), len(compressed_difference), len(compressed_sign),
                           compressed_latent + compressed_difference + compressed_sign))
    return compressed

def save_compressed(test_loader, model, num_images):
    model.eval()
    with torch.no_grad():
        count = 0
        for _, (image_batch, _) in tqdm(enumerate(test_loader)):
            if count >= num_images:
                break
            compressed = encode_images(image_batch, model)
            for i, (latent_len, difference_len, sign_len, bytes) in enumerate(compressed):
                original = transforms.ToPILImage()(image_batch[i])
                original.save(os.path.join(config.results_dir, f"original-{i}.png"))
                header = struct.pack("3I", latent_len, difference_len, sign_len)
                with open(os.path.join(config.results_dir, f"compressed-{i}.bin"), 'wb') as f:
                    f.write(header)
                    f.write(bytes)
            count += len(image_batch)

save_compressed(test_loader, model, 32)