import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm


class DiscriminatorSNBlock(nn.Module):
    # A generator block to upsample the input by a factor of 2
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorSNBlock, self).__init__()
        self.relu = nn.ReLU()
        self.downsample = nn.AvgPool2d(2)

        self.conv_module = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)),
            nn.ReLU(),
        )

        self.residual_conv = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1))

    def forward(self, x):
        identity = x

        # SN Conv and Downsample
        x = self.conv_module(x)
        x = self.downsample(x)

        # Residual connection
        return x + self.residual_conv(self.downsample(identity))


class SAGANDiscriminator(nn.Module):
    def __init__(self, n_heads=1):
        super(SAGANDiscriminator, self).__init__()
        self.n_heads = n_heads

        self.block0 = DiscriminatorSNBlock(3, 32)  # 64 x 64
        self.block1 = DiscriminatorSNBlock(32, 64)  # 32 x 32
        self.block2 = DiscriminatorSNBlock(64, 128)  # 16 x 16
        self.block3 = DiscriminatorSNBlock(128, 256)  # 8 x 8
        self.block4 = DiscriminatorSNBlock(256, 256)  # 4 x 4

        self.flatten = nn.Flatten()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Linear(256, 1, bias=False)

        self.attn1 = nn.MultiheadAttention(64, num_heads=self.n_heads)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        # Forward
        out = self.block0(x)
        out = self.block1(out)

        #### Attention 1 ####
        identity = out
        B, C, H, W = out.shape
        q_1 = out.view(H * W, B, C)
        k_1 = out.view(H * W, B, C)
        v_1 = out.view(H * W, B, C)
        out, attn_map_1 = self.attn1(q_1, k_1, v_1)
        out = out.view(B, C, H, W)

        # Residual connection
        out = identity + self.alpha * out

        # Forward (contd.)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.avg_pool(out)
        out = self.flatten(out)

        return self.clf(out).squeeze(), attn_map_1
