import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm


class GeneratorSNBlock(nn.Module):
    # A generator block to upsample the input by a factor of 2
    def __init__(self, in_channels, out_channels):
        super(GeneratorSNBlock, self).__init__()
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_module = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.residual_conv = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1))

    def forward(self, x):
        identity = x

        # Upsample and SN Conv
        x = self.upsample(x)
        x = self.conv_module(x)

        # Residual connection
        return x + self.residual_conv(self.upsample(identity))


class SAGANGenerator(nn.Module):
    def __init__(self, z_dim, in_channels, n_heads=1):
        super(SAGANGenerator, self).__init__()
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.n_heads = n_heads

        self.z_linear = spectral_norm(
            nn.Linear(self.z_dim, 4 * 4 * self.in_channels, bias=False)
        )

        self.block1 = GeneratorSNBlock(self.in_channels, 256)  # 8 x 8
        self.block2 = GeneratorSNBlock(256, 128)  # 16 x 16
        self.block3 = GeneratorSNBlock(128, 64)  # 32 x 32
        self.block4 = GeneratorSNBlock(64, 32)  # 64 x 64
        self.block5 = GeneratorSNBlock(32, 16)  # 128 x 128

        self.attn1 = nn.MultiheadAttention(64, num_heads=self.n_heads)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.last = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, padding=1),
        )

    def forward(self, z):
        # Reshape z
        z = self.z_linear(z)
        z = z.view(z.size(0), self.in_channels, 4, 4)

        # Forward
        out = self.block1(z)
        out = self.block2(out)
        out = self.block3(out)

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
        out = self.block4(out)
        out = self.block5(out)
        out = self.last(out)

        return torch.tanh(out), attn_map_1
