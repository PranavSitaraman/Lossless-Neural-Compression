import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

class TransformedDataset(Dataset):
    def __init__(self, dataset, image_size, num_colors=7):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.num_colors = num_colors

    def __len__(self):
        return len(self.dataset)

    def reward_fn(self, image):
        quantized = transforms.GaussianBlur(kernel_size=9, sigma=1.7)(image).permute(1, 2, 0)
        reshaped = quantized.view(-1, 3)
        indices = torch.randperm(reshaped.shape[0])[: self.num_colors]
        centers = reshaped[indices]
        
        for _ in range(10):
            distances = torch.cdist(reshaped, centers)
            labels = torch.argmin(distances, dim=1)
            mask = F.one_hot(labels, num_classes=self.num_colors).float().T
            old_centers = centers.clone()
            sums = mask.sum(dim=1, keepdim=True)
            new_centers = (mask @ reshaped) / (sums + 1e-6)
            centers = torch.where(sums > 0, new_centers, old_centers)

        distances = torch.cdist(reshaped, centers)
        labels = torch.argmin(distances, dim=1)
        quantized = centers[labels].view(quantized.shape).permute(2, 0, 1)
        target = image.float() - quantized.float()
        return target, quantized

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image = self.transform(image)
        target, quantized = self.reward_fn(image)
        return image, target, quantized