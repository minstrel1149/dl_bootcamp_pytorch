import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            # in_channels가 아니라 out_channels. 왜?
            # out_channels는 곧 kernel의 개수와 동일
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, X):
        y = self.block(X)
        return y
    
class ConvolutionalClassifier(nn.Module):
    def __init__(self, output_size):
        self.output_size = output_size

        super().__init__()

        self.blocks = nn.Sequential(  # (n, 1, 28, 28)
            ConvolutionBlock(1, 32),  # (n, 32, 14, 14)
            ConvolutionBlock(32, 64),  # (n, 64, 7, 7)
            ConvolutionBlock(64, 128),  # (n, 128, 4, 4)
            ConvolutionBlock(128, 256),  # (n, 256, 2, 2)
            ConvolutionBlock(256, 512)  # (n, 512, 1, 1)
        )

        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, X):
        assert X.dim() > 2, 'CNN에서 X의 차원은 2 이하이면 안됩니다.'
        if X.dim() == 3:
            X = X.reshape(-1, 1, X.shape[-2], X.shape[-1])
        # self.layers에 통과시키기 전에 (n, 512, 1, 1)에서 마지막 두 Dimension을 없애주는 형태
        y = self.layers(self.blocks(X).squeeze())
        return y