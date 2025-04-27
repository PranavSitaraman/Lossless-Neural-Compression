import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from config import config
import zlib
from io import BytesIO

experiment_dir = config.results_dir
start_img = 20
num_images = 5

fig, axes = plt.subplots(num_images, 4, figsize=(15, 20))
for i in range(start_img, start_img + num_images):
    original_path = os.path.join(experiment_dir, f"original_{i}.png")
    output_path = os.path.join(experiment_dir, f"output_{i}.png")
    latent_img_path = os.path.join(experiment_dir, f"latent_img_{i}.png")
    diff_img_path = os.path.join(experiment_dir, f"diff_img_{i}.png")
    latent_path = os.path.join(experiment_dir, f"latent_{i}.zz")
    extra_diff_path = os.path.join(experiment_dir, f"extra_diff_{i}.zz")

    original = Image.open(original_path) if os.path.exists(original_path) else None
    output = Image.open(output_path) if os.path.exists(output_path) else None
    latent_img = Image.open(latent_img_path) if os.path.exists(latent_img_path) else None
    diff_img = Image.open(diff_img_path) if os.path.exists(diff_img_path) else None

    latent = zlib.decompress(open(latent_path, "rb").read()) if os.path.exists(latent_path) else None
    extra_diff = zlib.decompress(open(extra_diff_path, "rb").read()) if os.path.exists(extra_diff_path) else None
    
    axes[i - start_img, 0].imshow(original)
    axes[i - start_img, 0].set_title("Original")
    axes[i - start_img, 0].axis("off")
    
    axes[i - start_img, 1].imshow(latent_img)
    axes[i - start_img, 1].set_title("Latent")
    axes[i - start_img, 1].axis("off")

    axes[i - start_img, 2].imshow(diff_img)
    axes[i - start_img, 2].set_title("Difference")
    axes[i - start_img, 2].axis("off")
    
    axes[i - start_img, 3].imshow(output)
    axes[i - start_img, 3].set_title("Reconstruction")
    axes[i - start_img, 3].axis("off")

    original_size = os.path.getsize(original_path) if os.path.exists(original_path) else None
    latent_size = os.path.getsize(latent_path) if os.path.exists(latent_path) else None
    diff_size = os.path.getsize(diff_img_path) if os.path.exists(diff_img_path) else None
    extra_diff_size = os.path.getsize(extra_diff_path) if os.path.exists(extra_diff_path) else None

    print(f"Image ID: {i}")
    print(f"Total compressed original: {original_size};")
    print(f"Total compressed reconstruction: {latent_size} (latent) + {diff_size + extra_diff_size} (difference) = {latent_size + diff_size + extra_diff_size}.\n")

plt.tight_layout()
plt.savefig(config.analysis_path)