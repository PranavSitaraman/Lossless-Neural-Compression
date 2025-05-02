import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, image_size: int = 1024, patch_size: int = 128, latent_channels: int = 512):
        super(ConvAutoencoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),       # 128 → 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),      # 64 → 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),     # 32 → 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),    # 16 → 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, latent_channels, 4, stride=2, padding=1),  # 8 → 4
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, 3, stride=2, padding=1, output_padding=1),  # 4 → 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),               # 8 → 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),                # 16 → 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),                 # 32 → 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),                  # 64 → 128
            nn.Tanh()
        )

    # ---------- patch utilities ----------
    def split_into_patches(self, x: torch.Tensor) -> torch.Tensor:
        ps = self.patch_size
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)            # (B,C,n_h,n_w,ps,ps)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, ps, ps)
        return x

    def assemble_patches(self, patches: torch.Tensor) -> torch.Tensor:
        ps = self.patch_size
        npr = self.image_size // ps
        B = patches.size(0) // (npr * npr)
        patches = patches.view(B, npr, npr, 3, ps, ps)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(B, 3, self.image_size, self.image_size)

    # ---------- encode / decode ----------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.split_into_patches(x))

    def decode(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        return self.assemble_patches(self.decoder(encoded_patches))

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))