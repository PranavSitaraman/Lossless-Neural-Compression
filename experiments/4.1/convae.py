import torch
import torch.nn as nn
import torch.nn.functional as F

def gradients(x: torch.Tensor):
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :-1] = (x[..., 1:] - x[..., :-1]).abs()
    gy[..., :-1, :] = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return gx, gy

class EdgeAwareLoss(nn.Module):
    def __init__(self, lambda_grad: float = 0.1):
        super().__init__()
        self.lambda_grad = lambda_grad

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(output, target)
        gx_o, gy_o = gradients(output)
        gx_t, gy_t = gradients(target)
        grad_loss = F.l1_loss(gx_o, gx_t) + F.l1_loss(gy_o, gy_t)
        return l1 + self.lambda_grad * grad_loss

class ConvAutoencoder(nn.Module):
    def __init__(self, image_size: int = 1024, patch_size: int = 128,
                 stride: int = 64, latent_dim: int = 512):
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
        return self.encoder(self.create_patches(x.clamp(0, 1)))

    def decode(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self.combine_patches(self.decoder(z), batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        z = self.encode(x)
        return self.decode(z, B)