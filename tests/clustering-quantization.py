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

# Directory containing the files
experiment_dir = str(config.results_dir)

# Iterate through each ID and load corresponding files
fig, axes = plt.subplots(3, 5, figsize=(20, 10))

for i in range(10, 15):
    # Construct the original image path
    original_path = os.path.join(experiment_dir, f"original_{i}.png")
    
    # Load image
    original = Image.open(original_path)
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    
    # Apply transform
    image = transform(original)
    num_colors = 7
    
    # Apply Gaussian blur
    quantized = transforms.GaussianBlur(kernel_size=9, sigma=1.7)(image).to(torch.float32).permute(1, 2, 0) * 255
    reshaped = quantized.view(-1, 3)
    
    # For each pixel, find the closest color from our 10 defined colors
    # This is done by finding the index of the closest value in color_values
    # and then retrieving that valuereshaped = blurred.view(-1, 3)
    indices = torch.randperm(reshaped.shape[0])[:num_colors]
    centers = reshaped[indices]

    for _ in range(10):
        distances = torch.cdist(reshaped, centers)
        labels = torch.argmin(distances, dim=1)
        for j in range(num_colors):
            if (labels == j).sum() > 0:
                centers[j] = reshaped[labels == j].mean(dim=0)

    quantized = centers[labels].view(quantized.shape).to(torch.uint8)
    quantized_np = quantized.numpy()
    image_np = (image.permute(1, 2, 0) * 255).numpy()
    target_np = np.zeros_like(quantized_np)
    
    for h in range(quantized_np.shape[0]):
        for w in range(quantized_np.shape[1]):
            for c in range(quantized_np.shape[2]):
                target_np[h, w, c] = image_np[h, w, c] - quantized_np[h, w, c]
    
    # Convert back to torch tensor in the correct format
    quantized = torch.from_numpy(quantized_np).permute(2, 0, 1) / 255.0
    target = torch.from_numpy(target_np).permute(2, 0, 1) / (2 * 255.0) + 0.5
    target = torch.clamp(target, 0, 1)
    
    image = torch.from_numpy(image_np).permute(2, 0, 1) / 255.0
    
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
plt.savefig('tests/clustering-quantization.png')
plt.show()