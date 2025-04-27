import sys
import os

# Move up one directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
os.chdir("..")

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from config import config
import zlib
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

experiment_dir = str(config.results_dir)
fig, axes = plt.subplots(3, 5, figsize=(20, 10))

for i in range(10, 15):
    original_path = os.path.join(experiment_dir, f"original_{i}.png")
    original = Image.open(original_path)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    image = transform(original)
    num_colors = 7

    quantized = transforms.GaussianBlur(kernel_size=9, sigma=1.7)(image).permute(1, 2, 0)
    reshaped = quantized.view(-1, 3)
    indices = torch.randperm(reshaped.shape[0])[:num_colors]
    centers = reshaped[indices]
    
    for _ in range(10):
        distances = torch.cdist(reshaped, centers)
        labels = torch.argmin(distances, dim=1)
        mask = F.one_hot(labels, num_classes=num_colors).float().T
        old_centers = centers.clone()
        sums = mask.sum(dim=1, keepdim=True)
        new_centers = (mask @ reshaped) / (sums + 1e-6)
        centers = torch.where(sums > 0, new_centers, old_centers)

    distances = torch.cdist(reshaped, centers)
    labels = torch.argmin(distances, dim=1)
    quantized = centers[labels].view(quantized.shape).permute(2, 0, 1)
    diff = image.float() - quantized.float()

    # Transform for visualization
    target = diff / 2 + 0.5
    target = torch.clamp(target, 0, 1)
    
    # Convert tensor back to PIL image
    image = transforms.ToPILImage()(image)
    quantized = transforms.ToPILImage()(quantized)
    target = transforms.ToPILImage()(target)
    
    # Display original
    axes[0, i - 10].imshow(image)
    axes[0, i - 10].set_title(f"Image {i} Original")
    axes[0, i - 10].axis('on')

    # Display quantized
    axes[1, i - 10].imshow(quantized)
    axes[1, i - 10].set_title(f"Image {i} Quantized")
    axes[1, i - 10].axis('on')
    
    # Display target
    axes[2, i - 10].imshow(target)
    axes[2, i - 10].set_title(f"Image {i} Target")
    axes[2, i - 10].axis('on')

# Adjust layout and save
plt.tight_layout()
plt.savefig('tests/efficient-quantization.png')
plt.show()