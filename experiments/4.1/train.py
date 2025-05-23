import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.optim import Adam
import torchvision
from tqdm import tqdm
import os
import socket
from config import config
from convae import ConvAutoencoder, EdgeAwareLoss
from transformed import TransformedDataset
    
if __name__ == "__main__":
    os.makedirs(config.experiment_root, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    rank          = int(os.environ["SLURM_PROCID"])
    world_size    = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    assert (
    gpus_per_node == torch.cuda.device_count()
    ), f'SLURM_GPUS_ON_NODE={gpus_per_node} vs torch.cuda.device_count={torch.cuda.device_count()}'

    print(
    f"Hello from rank {rank} of {world_size} on {socket.gethostname()} where there are" \
    f" {gpus_per_node} allocated GPUs per node." \
    f' | (CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]})', flush=True
    )

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    device = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(device)

    print(f"Using GPU{device} on Machine {os.uname().nodename.split('.')[0]} (Rank {rank})", flush=True)

    model = ConvAutoencoder(image_size=config.image_size).to(device)
    model = DistributedDataParallel(model, device_ids=[device])

    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path))
        print("Loaded pre-trained model.")
    else:
        print("No pre-trained model found. Training from scratch.")
    
    criterion = EdgeAwareLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    imagenet_dataset = torchvision.datasets.ImageFolder(root=config.data_root)
    subset_indices = list(range(config.subset_size))
    train_dataset = TransformedDataset(Subset(imagenet_dataset, subset_indices), config.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank),
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
        shuffle=False
    )
    scaler = torch.amp.GradScaler("cuda")

    for i in range(config.training_epochs):
        print(f"[GPU{rank}] Epoch {i} | Batchsize: {len(next(iter(train_loader))[0])} | Steps: {len(train_loader)}", flush=True)
        train_loader.sampler.set_epoch(i)
        
        if rank == 0:
            print(f"Epoch {i+1}/{config.training_epochs} starting...", flush=True)
        model.train()
        epoch_loss_tensor = torch.tensor(0.0, device=device)

        for image, target, quantized in tqdm(train_loader, desc=f"Rank {rank} Training", disable=(rank != 0)):
            image, target, quantized = image.to(device), target.to(device), quantized.to(device)

            with torch.amp.autocast("cuda"):
                outputs = 2 * (model(target/2 + 0.5) - 0.5)
                loss = criterion(outputs, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss_tensor += loss.to(device, non_blocking=True)
        
        print(f"Rank {rank} | Epoch Loss: {epoch_loss_tensor.item()}", flush=True)
        dist.reduce(epoch_loss_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            print(f"Epoch [{i+1}/{config.training_epochs}], Total Loss: {epoch_loss_tensor.item():.4f}", flush=True)
            torch.save(model.state_dict(), config.model_path)

    dist.destroy_process_group()