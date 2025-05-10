import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32768, input_size=1024):
      
        super(ConvAutoencoder, self).__init__()
      
        # Encoder (Conv + ReLU)
        self.encoder_conv = nn.Sequential(
            # --- Block 1: from 3 → 16, then 16 → 16 ---
            
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 1024 → 512

            # --- Block 2: 16 → 32, then 32 → 32 ---
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 512 → 256

            # --- Block 3: 32 → 64, then 64 → 64 ---
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 → 128

            # --- Block 4: 64 → 128, then 128 → 128 ---
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 → 64

            # --- Block 5: 128 → 128, then 128 → 128 ---
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32

            # Block 6
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32 -> 16
        )

        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, 3, input_size, input_size)
        with torch.no_grad():
            dummy_output = self.encoder_conv(dummy_input)
        self.flattened_size = dummy_output.view(1, -1).shape[1] 
        self.feature_shape = dummy_output.shape[1:]  # (C, H, W) for unflattening
        print(f"DEBUG| Architecture Flattened size: {self.flattened_size} (before FC)")
        print(f"DEBUG|NOTE: FC layer is not used in this implementation")
        # Fully connected bottleneck

        self.encoder_fc = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(self.flattened_size, latent_dim),
            # nn.ReLU()
        )

        # --------------------
        #      Decoder
        # --------------------
        # Map latent_dim back to flattened_size
        self.decoder_fc = nn.Sequential(# Currently just the identity since we disabled the FC
            # nn.Linear(latent_dim, self.flattened_size),
            # nn.ReLU()
        )
        
        # Transposed conv blocks to go from (256, 32, 32) → (3, 1024, 1024)
        # Here we "undo" each pooling step. We'll do:
        #  (256, 32, 32)   -> Up -> (256, 64, 64)   -> reduce channels -> (128, 64, 64)
        #  (128, 64, 64)   -> Up -> (128, 128, 128) -> reduce channels -> (64, 128, 128)
        #  (64, 128, 128)  -> Up -> (64, 256, 256)  -> reduce channels -> (32, 256, 256)
        #  (32, 256, 256)  -> Up -> (32, 512, 512)  -> reduce channels -> (16, 512, 512)
        #  (16, 512, 512)  -> Up -> (16, 1024,1024) -> reduce channels -> (3, 1024, 1024)

        # You can do this with either `nn.ConvTranspose2d(...)` or upsampling + conv.
        # Below uses upsampling + conv for clarity and symmetry:

        self.decoder_deconv = nn.Sequential(
            # --- UpBlock 1: (32,16,16)->(32,32,32) => conv(32->32, 32->32)
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # --- UpBlock 2: (32,32,32)->(32,64,64) => conv(32->64, 64->128)
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # --- Block 3 ---
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # --- Block 4 ---
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # --- Block 5 ---
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # --- Block 6 ---
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()  # difference is [-1,1]
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