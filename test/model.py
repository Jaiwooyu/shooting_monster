import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BasketballShootingAnalyzer(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        # 3D CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        # Transformer layers
        encoder_layer = TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=6)
        
        # Final classification layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn_layers(x)
        
        # Reshape for transformer
        batch_size, channels, t, h, w = x.size()
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=0)
        
        # Final classification
        x = self.fc_layers(x)
        return x