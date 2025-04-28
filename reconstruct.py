import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset, Dataset
import torch.distributed as dist
import torchvision
from tqdm import tqdm
import os
from config import config
from convae import ConvAutoencoder
import zlib
from transformed import TransformedDataset

if __name__ == "__main__":
    os.makedirs(config.experiment_root, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor()
    ])

    imagenet_dataset = torchvision.datasets.ImageFolder(root=config.data_root)
    subset_indices = list(range(config.subset_size))
    test_dataset = TransformedDataset(Subset(imagenet_dataset, subset_indices), config.image_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    num_images = 32
    num_colors = 7
    experiment_dir = config.results_dir

    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    model = ConvAutoencoder(image_size=config.image_size)
    model = DistributedDataParallel(model)
    model_path = config.model_path
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model.")
    else:
        print("No pre-trained model found.")
        exit(0)

    os.makedirs(experiment_dir, exist_ok=True)

    count = 0
    for image_batch, target_batch, quantized_batch in tqdm(test_loader):
        if count >= num_images:
            break

        model.eval()
        with torch.no_grad():
            latents = model.module.encode(target_batch)
            reconstructions = torch.round(model.module.decode(latents).clamp(-1, 1) * 255.0).to(torch.int16)
        
        image_batch = torch.round(image_batch.clamp(0, 1) * 255.0).to(torch.uint8)
        target_batch = torch.round(target_batch.clamp(-1, 1) * 255.0).to(torch.int16)
        quantized_batch = torch.round(quantized_batch.clamp(0, 1) * 255.0).to(torch.uint8)

        differences = image_batch.to(torch.int16) - reconstructions.to(torch.int16)
        latent_img = ((reconstructions + 255) // 2).clamp(0, 255).to(torch.uint8)
        target_batch = ((target_batch + 255) // 2).clamp(0, 255).to(torch.uint8)
        clamped_diff = differences.clamp(0, 255)
        diff_img = clamped_diff.to(torch.uint8)
        extra_diff = differences.to(torch.int16) - clamped_diff.to(torch.int16)
        latents = latents.numpy()
        output = (reconstructions.to(torch.int16) + diff_img.to(torch.int16) + extra_diff.to(torch.int16)).clamp(0, 255).to(torch.uint8)
        extra_diff = extra_diff.numpy()

        for i in range(image_batch.size(0)):
            original = transforms.ToPILImage()(image_batch[i])
            target_i = transforms.ToPILImage()(target_batch[i])
            quantized_i = transforms.ToPILImage()(quantized_batch[i])
            latent_img_i = transforms.ToPILImage()(latent_img[i])
            diff_img_i = transforms.ToPILImage()(diff_img[i])
            output_i = transforms.ToPILImage()(output[i])
            latent_i = zlib.compress(latents[i], level=config.compression_quality)
            extra_diff_i = zlib.compress(extra_diff[i], level=config.compression_quality)
            
            original.save(os.path.join(experiment_dir, f"original_{count}.png"))
            output_i.save(os.path.join(experiment_dir, f"output_{count}.png"))
            target_i.save(os.path.join(experiment_dir, f"target_{count}.png"))
            quantized_i.save(os.path.join(experiment_dir, f"quantized_{count}.png"))
            latent_img_i.save(os.path.join(experiment_dir, f"latent_img_{count}.png"))
            diff_img_i.save(os.path.join(experiment_dir, f"diff_img_{count}.png"))
            open(os.path.join(experiment_dir, f"latent_{count}.zz"), "wb").write(latent_i)
            open(os.path.join(experiment_dir, f"extra_diff_{count}.zz"), "wb").write(extra_diff_i)
            count += 1