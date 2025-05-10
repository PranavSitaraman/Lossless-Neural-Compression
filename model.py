import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

def gradients(x: torch.Tensor):
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :-1] = (x[..., 1:] - x[..., :-1]).abs()
    gy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return gx, gy

class EdgeAwareLoss(nn.Module):
    def __init__(self, lambda_grad: float = 0.01):
        super().__init__()
        self.lambda_grad = lambda_grad

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(output, target)
        gx_o, gy_o = gradients(output)
        gx_t, gy_t = gradients(target)
        grad_loss = F.l1_loss(gx_o, gx_t) + F.l1_loss(gy_o, gy_t)
        return l1 + self.lambda_grad * grad_loss

class TopKActivation(nn.Module):
    """
    Keeps only the top-k absolute activations in each sample; zeros out the rest.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        flat = x.view(B, -1)
        abs_flat = flat.abs()
        N = abs_flat.size(1)
        if self.k >= N:
            return x
        topk_vals = torch.topk(abs_flat, self.k, dim=1, largest=True).values
        thresh = topk_vals[:, -1].unsqueeze(1)
        mask = (abs_flat >= thresh).type_as(flat)
        flat = flat * mask
        return flat.view_as(x)

class ConvAutoencoder(nn.Module):
    def __init__(
        self, image_size: int = 1024, patch_size: int = 128,
        stride: int = 64, latent_dim: int = 512, topk: int = 512
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        assert self.patch_size % 8 == 0, "patch_size must be divisible by 8"

        # ── Encoder ──
        channels = [3, 32, 64, 128, 256]
        enc = []
        for c_in, c_out in zip(channels, channels[1:]):
            enc += [nn.Conv2d(c_in, c_out, 4, 2, 1),
                    nn.GroupNorm(8, c_out),
                    nn.ReLU(inplace=True)]
        enc += [nn.Conv2d(256, latent_dim, 1)]
        self.encoder = nn.Sequential(*enc)

        # Top-K activation layer
        self.topk_act = TopKActivation(k=topk)

        # ── Decoder ──
        def up_block(c_in: int, c_out: int):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c_in, c_out, 3, 1, 1),
                nn.GroupNorm(8, c_out),
                nn.ReLU(inplace=True),
            )
        self.decoder = nn.Sequential(
            up_block(latent_dim, 256),
            up_block(256, 128),
            up_block(128, 64),
            up_block(64, 32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
    
    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        ps, s = self.patch_size, self.stride
        patches = F.unfold(x, kernel_size=ps, stride=s)
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(-1, 3, ps, ps)
        return patches

    def combine_patches(self, patches: torch.Tensor, batch_size: int) -> torch.Tensor:
        ps, s = self.patch_size, self.stride
        device = patches.device

        L = patches.size(0) // batch_size
        patches = patches.view(batch_size, L, 3 * ps * ps).contiguous()
        patches = patches.transpose(1, 2)

        H = W = self.image_size
        out = F.fold(patches, (H, W), kernel_size=ps, stride=s)

        ones = torch.ones((batch_size, 3, H, W), device=device)
        divisor = F.fold(F.unfold(ones, kernel_size=ps, stride=s), (H, W), kernel_size=ps, stride=s)
        return (out / divisor).clamp(0, 1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.create_patches(x.clamp(0, 1)))
        return self.topk_act(z)

    def decode(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self.combine_patches(self.decoder(z), batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        z = self.encode(x)
        return self.decode(z, B)

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
        quantized = image.permute(1, 2, 0)
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