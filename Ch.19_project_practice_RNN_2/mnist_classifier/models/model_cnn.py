import torch
import torch.nn as nn
import math

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, X):
        y = self.block(X)
        return y

class ConvolutionClassifier(nn.Module):
    def __init__(self, output_size, image_size=28, base_channels=32):
        super().__init__()
        self.output_size = output_size
        self.image_size = image_size
        self.base_channels = base_channels

        blocks = []
        current_channels = 1
        current_size = self.image_size
        
        out_channels = self.base_channels
        while current_size > 1:
            block = ConvolutionBlock(current_channels, out_channels)
            blocks.append(block)
            
            current_channels = out_channels
            out_channels *= 2 # Double the channels for the next block
            
            # Calculate the new size based on the ConvolutionBlock's architecture
            current_size = math.floor((current_size - 1) / 2) + 1
        
        self.blocks = nn.Sequential(*blocks)
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, X):
        assert X.dim() > 2, 'CNN에서 X의 차원은 2 이하이면 안됩니다.'
        if X.dim() == 3:
            X = X.reshape(-1, 1, X.shape[-2], X.shape[-1])
        y = self.layers(self.blocks(X))

        return y