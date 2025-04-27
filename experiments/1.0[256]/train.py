import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import socket
from convae import ConvAutoencoder
from config import config

os.makedirs(config.experiment_root, exist_ok=True)
os.makedirs(config.results_dir, exist_ok=True)
os.makedirs(config.logs_dir, exist_ok=True)

class ColorQuantizedDataset(Dataset):
    def __init__(self, dataset, image_size, num_colors):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.num_colors = num_colors
    
    def __len__(self):
        return len(self.dataset)
    
    def reward_fn(self, image):
        color_values = np.linspace(0, 255, 3)
        quantized = transforms.GaussianBlur(kernel_size=9, sigma=1.7)(image)
    
        # Convert to numpy for easier processing
        quantized_np = (quantized.permute(1, 2, 0) * 255).numpy()
        image_np = (image.permute(1, 2, 0) * 255).numpy()

        # Create a target array to store the difference
        target_np = np.zeros_like(quantized_np)
        
        # For each pixel, find the closest of each RGB value
        for h in range(quantized_np.shape[0]):
            for w in range(quantized_np.shape[1]):
                for c in range(quantized_np.shape[2]):
                    closest_idx = (np.abs(color_values - quantized_np[h, w, c])).argmin()
                    target_np[h, w, c] = image_np[h, w, c] - color_values[closest_idx]
        
        # Convert back to torch tensor in the correct format
        target = torch.from_numpy(target_np).permute(2, 0, 1) / 255.0 + 0.5
        target = torch.clamp(target, 0, 1)
        return target
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image = self.transform(image)
        target = self.reward_fn(image)
        return image, target

rank          = int(os.environ["SLURM_PROCID"])
world_size    = int(os.environ["WORLD_SIZE"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
num_colors    = 10

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

model = ConvAutoencoder(input_size=config.image_size).to(device)
model = DDP(model, device_ids=[device])
if os.path.exists(config.model_path):
    model.load_state_dict(torch.load(config.model_path))
    print("Loaded pre-trained model.")
else:
    print("No pre-trained model found. Training from scratch.")
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

imagenet_dataset = torchvision.datasets.ImageFolder(root=config.data_root)
subset_indices = list(range(config.subset_size))
train_dataset = ColorQuantizedDataset(Subset(imagenet_dataset, subset_indices), config.image_size, num_colors=num_colors)

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
    for original, target in tqdm(train_loader, desc=f"Rank {rank} Training", disable=(rank != 0)):
        original, target = original.to(device), target.to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(original)
            loss = loss_fn(outputs, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss_tensor += loss.to(device, non_blocking=True)
    
    print(f"Rank {rank} | Epoch Loss: {epoch_loss_tensor.item()}", flush=True)
    dist.reduce(epoch_loss_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss = epoch_loss_tensor.item() / world_size
        print(f"Epoch [{i+1}/{config.training_epochs}], Avg Loss: {avg_loss:.4f}", flush=True)
        torch.save(model.state_dict(), config.model_path)

dist.destroy_process_group()