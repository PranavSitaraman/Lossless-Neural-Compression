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

# Initialize the autoencoder
model = ConvAutoencoder(image_size=config.image_size)
model_path = config.model_path
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print("Loaded pre-trained model.")
else:
    print("No pre-trained model found.")

def decompress_png(png_bytes):
    buffer = BytesIO(png_bytes)
    image = Image.open(buffer)
    return image

def save_decompressed(model, num_images):
    for i in range(num_images):
        with open(os.path.join(config.results_dir, f"compressed-{i}.bin"), 'rb') as f:
            header = f.read(12)  # 3 * 4 bytes
            latent_len, difference_len, sign_len = struct.unpack('3I', header)
            compressed_latent = f.read(latent_len)
            compressed_difference = f.read(difference_len)
            compressed_sign = f.read(sign_len)

            latent_raw = zlib.decompress(compressed_latent)
            latent = np.frombuffer(latent_raw, dtype=np.float32).reshape((32768,))

            diff_image = decompress_png(compressed_difference)
            difference = np.array(diff_image).reshape((1024,1024,3)).astype(np.int16)

            sign_raw = zlib.decompress(compressed_sign)
            sign = np.frombuffer(sign_raw, dtype=np.bool).reshape((3,1024,1024)).transpose(1, 2, 0)

            difference[sign] *= -1
            with torch.no_grad():
                reconstruction = torch.zeros((1024,1024,3))
            image = (reconstruction.numpy().astype(np.int16) + difference).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(os.path.join(config.results_dir, f"decompressed-{i}.png"))
            
save_decompressed(model, 32)