import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, input_size=1024):
        super(ConvAutoencoder, self).__init__()
      
        # Encoder (Conv + ReLU)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 1024 → 512
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 512 → 256
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 256 → 128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 → 64
            nn.ReLU()
        )

        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        with torch.no_grad():
            dummy_output = self.encoder_conv(dummy_input)
        self.flattened_size = dummy_output.view(1, -1).shape[1]
        self.feature_shape = dummy_output.shape[1:]  # (C, H, W) for unflattening

        # Fully connected bottleneck
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, latent_dim),
            nn.ReLU()
        )

        # Decoder (Linear + reshape + ConvTranspose)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            nn.ReLU()
        )

        self.decoder_deconv = nn.Sequential(
            nn.Unflatten(1, self.feature_shape),  # Reshape to (C, H, W)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 128 → 256
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # 256 → 512
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 512 → 1024
            nn.Sigmoid()
        )
    def encoder(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_fc(x)
        return x
    def decoder(self, x):
        x = self.decoder_fc(x)
        x = self.decoder_deconv(x)
        return x

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_fc(x)
        x = self.decoder_fc(x)
        x = self.decoder_deconv(x)
        return x