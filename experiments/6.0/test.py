import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset, Dataset
import torch.distributed as dist
import torchvision
from tqdm import tqdm
import os
import socket
from config import config
from matplotlib import pyplot as plt
import zlib
from PIL import Image
from model import ConvAutoencoder, TransformedDataset
from contextlib import redirect_stdout

experiment_dir = config.results_dir

if __name__ == "__main__":
    os.makedirs(config.experiment_root, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    print(
        f"Hello from rank {rank} of {world_size} on {socket.gethostname()} where there are" \
        f" {torch.cuda.device_count()} available GPUs.", flush=True
    )

    dist.init_process_group(backend="gloo")
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    device = local_rank
    torch.cuda.set_device(device)

    print(f"Using GPU{device} on Machine {os.uname().nodename.split('.')[0]} (Rank {rank})", flush=True)

    model = ConvAutoencoder(image_size=config.image_size).to(device)
    model = DistributedDataParallel(model, device_ids=[device])

    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path))
        print("Loaded pre-trained model.", flush=True)
    else:
        print("No pre-trained model found. Training from scratch.", flush=True)
    
    dataset = torchvision.datasets.ImageFolder(root=config.data_root)
    subset_indices = list(range(config.training_size + rank, config.total_size, world_size))
    test_dataset = TransformedDataset(Subset(dataset, subset_indices), config.image_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )

    start = True
    fig, axes = plt.subplots(config.batch_size, 8, figsize=(32, 32))
    num_batch = 0

    with open(os.path.join(config.experiment_root, f"analysis-{rank}.txt"), 'w') as out, redirect_stdout(out):
        for idx, (image_batch, target_batch, quantized_batch) in enumerate(test_loader):
            count = config.batch_size * world_size * num_batch + config.batch_size * idx + rank
            image_batch, target_batch, quantized_batch = image_batch.to(device), target_batch.to(device), quantized_batch.to(device)
            
            model.eval()
            with torch.no_grad():
                latents = model.module.encode(target_batch/2 + 0.5)
                reconstruction = 2 * (model.module.decode(latents, target_batch.size(0)) - 0.5)
                reconstruction = torch.round(reconstruction.clamp(-1, 1) * 255.0).to(torch.int16)
            
            original_img = torch.round(image_batch.clamp(0, 1) * 255.0).to(torch.uint8)
            target = torch.round(target_batch.clamp(-1, 1) * 255.0).to(torch.int16)
            quantized_img = torch.round(quantized_batch.clamp(0, 1) * 255.0).to(torch.uint8)
            target_img = ((target + 255) // 2).clamp(0, 255).to(torch.uint8)
            
            learned_batch = original_img.to(torch.int16) - reconstruction.to(torch.int16)
            residual = original_img.to(torch.int16) - reconstruction.to(torch.int16) - quantized_img.to(torch.int16)
            residual_img = ((residual.clamp(-255, 255) + 255) // 2).clamp(0, 255).to(torch.uint8)
            reconstruct_img = ((reconstruction + 255) // 2).clamp(0, 255).to(torch.uint8)

            learned_quantized_img = learned_batch.clamp(0, 255).to(torch.uint8)
            lq_overflow = learned_batch.to(torch.int16) - learned_quantized_img.to(torch.int16)

            method1 = (reconstruction.to(torch.int16) + quantized_img.to(torch.int16) + residual.to(torch.int16)).clamp(0, 255).to(torch.uint8)
            method2 = (reconstruction.to(torch.int16) + learned_quantized_img.to(torch.int16) + lq_overflow.to(torch.int16)).clamp(0, 255).to(torch.uint8)

            latents = latents.cpu().numpy()
            lq_overflow = lq_overflow.cpu().numpy()
            residual = residual.cpu().numpy()

            base = num_batch * (config.batch_size * world_size)
            for i in range(image_batch.size(0)):
                original_img_i = transforms.ToPILImage()(original_img[i])
                target_img_i = transforms.ToPILImage()(target_img[i])
                quantized_img_i = transforms.ToPILImage()(quantized_img[i])
                residual_img_i = transforms.ToPILImage()(residual_img[i])
                reconstruct_img_i = transforms.ToPILImage()(reconstruct_img[i])
                learned_quantized_img_i = transforms.ToPILImage()(learned_quantized_img[i])
                method1_i = transforms.ToPILImage()(method1[i])
                method2_i = transforms.ToPILImage()(method2[i])

                latents_i = zlib.compress(bytes(latents[i]), level=config.compression_quality)
                lq_overflow_i = zlib.compress(bytes(lq_overflow[i]), level=config.compression_quality)
                residual_i = zlib.compress(bytes(residual[i]), level=config.compression_quality)

                # Image encodings/visualizations

                count = base + i * world_size + rank
                
                with open(os.path.join(experiment_dir, f"original_{count}.png"), "wb") as f:
                    original_img_i.save(f, format="PNG")
                    f.flush()
                    os.fsync(f.fileno())
                
                with open(os.path.join(experiment_dir, f"target_{count}.png"), "wb") as f:
                    target_img_i.save(f, format="PNG")
                    f.flush()
                    os.fsync(f.fileno())

                with open(os.path.join(experiment_dir, f"quantized_{count}.png"), "wb") as f:
                    quantized_img_i.save(f, format="PNG")
                    f.flush()
                    os.fsync(f.fileno())

                with open(os.path.join(experiment_dir, f"residual_{count}.png"), "wb") as f:
                    residual_img_i.save(f, format="PNG")
                    f.flush()
                    os.fsync(f.fileno())

                with open(os.path.join(experiment_dir, f"reconstruct_{count}.png"), "wb") as f:
                    reconstruct_img_i.save(f, format="PNG")
                    f.flush()
                    os.fsync(f.fileno())

                with open(os.path.join(experiment_dir, f"learned_quantized_{count}.png"), "wb") as f:
                    learned_quantized_img_i.save(f, format="PNG")
                    f.flush()
                    os.fsync(f.fileno())

                with open(os.path.join(experiment_dir, f"method1_{count}.png"), "wb") as f:
                    method1_i.save(f, format="PNG")
                    f.flush()
                    os.fsync(f.fileno())

                with open(os.path.join(experiment_dir, f"method2_{count}.png"), "wb") as f:
                    method2_i.save(f, format="PNG")
                    f.flush()
                    os.fsync(f.fileno())

                # Compressed raw data

                with open(os.path.join(experiment_dir, f"latents_{count}.zz"), "wb") as f:
                    f.write(latents_i)
                    f.flush()
                    os.fsync(f.fileno())
                
                with open(os.path.join(experiment_dir, f"lq_overflow_{count}.zz"), "wb") as f:
                    f.write(lq_overflow_i)
                    f.flush()
                    os.fsync(f.fileno())
                
                with open(os.path.join(experiment_dir, f"residual_{count}.zz"), "wb") as f:
                    f.write(residual_i)
                    f.flush()
                    os.fsync(f.fileno())

                # Visualize
            
                if start and rank == 0:
                    axes[i, 0].imshow(original_img_i)
                    axes[i, 0].set_title("Original")
                    axes[i, 0].axis("off")

                    axes[i, 1].imshow(target_img_i)
                    axes[i, 1].set_title("Target")
                    axes[i, 1].axis("off")
                    
                    axes[i, 2].imshow(reconstruct_img_i)
                    axes[i, 2].set_title("Reconstruction")
                    axes[i, 2].axis("off")

                    axes[i, 3].imshow(quantized_img_i)
                    axes[i, 3].set_title("Quantized")
                    axes[i, 3].axis("off")

                    axes[i, 4].imshow(learned_quantized_img_i)
                    axes[i, 4].set_title("Learned Quantization")
                    axes[i, 4].axis("off")

                    axes[i, 5].imshow(residual_img_i)
                    axes[i, 5].set_title("Residual to Truth")
                    axes[i, 5].axis("off")
                    
                    axes[i, 6].imshow(method1_i)
                    axes[i, 6].set_title("Method 1 Output")
                    axes[i, 6].axis("off")

                    axes[i, 7].imshow(method2_i)
                    axes[i, 7].set_title("Method 2 Output")
                    axes[i, 7].axis("off")
                
                original_size = os.path.getsize(os.path.join(experiment_dir, f"original_{count}.png"))
                latent_size = os.path.getsize(os.path.join(experiment_dir, f"latents_{count}.zz"))
                quantized_size = os.path.getsize(os.path.join(experiment_dir, f"quantized_{count}.png"))
                residual_size = os.path.getsize(os.path.join(experiment_dir, f"residual_{count}.zz"))
                learned_quantized_sized = os.path.getsize(os.path.join(experiment_dir, f"learned_quantized_{count}.png"))
                lq_overflow_size = os.path.getsize(os.path.join(experiment_dir, f"lq_overflow_{count}.zz"))

                print(f"Image ID: {count}")
                print(f"Original size: {original_size};")
                print(f"Method 1 size: {latent_size + quantized_size + residual_size} = {latent_size} (latent) + {quantized_size} (quantized) + {residual_size} (residual);")
                print(f"Method 2 size: {latent_size + learned_quantized_sized + lq_overflow_size} = {latent_size} (latent) + {learned_quantized_sized} (learned quantization) + {lq_overflow_size} (overflow).\n")

            start = False
            plt.tight_layout()
            plt.savefig(config.analysis_path)
            num_batch += 1
            dist.barrier()