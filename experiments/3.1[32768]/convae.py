import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, image_size=1024, patch_size=128, latent_dim=512):
        super(ConvAutoencoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 128x128 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1), # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 64x64 -> 128x128
            nn.Sigmoid()  # For pixel values between 0 and 1
        )

    def split_into_patches(self, x):
        ps = self.patch_size
        assert self.image_size % ps == 0 and self.image_size % ps == 0, "Image size must be divisible by patch size"
        
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)  # shape: (B, C, num_patches_h, num_patches_w, ps, ps)
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, num_patches_h, num_patches_w, C, ps, ps)
        x = x.reshape(-1, 3, ps, ps)     # (B*num_patches, C, ps, ps)
        return x

    def assemble_patches(self, patches):
        ps = self.patch_size
        num_patches_per_row = self.image_size // ps
        batch_size = patches.shape[0] // (num_patches_per_row * num_patches_per_row)

        patches = patches.view(batch_size, num_patches_per_row, num_patches_per_row, 3, ps, ps)
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, num_patches_h, ps, num_patches_w, ps)
        patches = patches.reshape(batch_size, 3, self.image_size, self.image_size)
        return patches

    def encode(self, x):
        patches = self.split_into_patches(x)
        encoded_patches = self.encoder(patches)
        return encoded_patches

    def decode(self, encoded_patches):
        decoded_patches = self.decoder(encoded_patches)
        reconstructed_images = self.assemble_patches(decoded_patches)
        return reconstructed_images

    def forward(self, x):
        encoded_patches = self.encode(x)
        reconstructed_images = self.decode(encoded_patches)
        return reconstructed_images