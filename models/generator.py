import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, image_size=64, z_dim=100, conv_dim=64, n_heads=1):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.z_dim = z_dim
        self.conv_dim = conv_dim
        self.n_heads = n_heads

        self.layer1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(self.z_dim, 8 * self.conv_dim, 4)),
            nn.BatchNorm2d(8 * self.conv_dim),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(8 * self.conv_dim, 4 * self.conv_dim, 4)),
            nn.BatchNorm2d(4 * self.conv_dim),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(4 * self.conv_dim, 2 * self.conv_dim, 4)),
            nn.BatchNorm2d(2 * self.conv_dim),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(2 * self.conv_dim, self.conv_dim, 4)),
            nn.BatchNorm2d(self.conv_dim),
            nn.ReLU(),
        )

        self.output = nn.ConvTranspose2d(self.conv_dim, 3, 4)

        self.attn1 = nn.MultiheadAttention(2 * self.conv_dim, num_heads=self.n_heads)
        self.alpha = nn.Parameter(0, requires_grad=True)
        self.attn2 = nn.MultiheadAttention(self.conv_dim, num_heads=self.n_heads)
        self.beta = nn.Parameter(0, requires_grad=True)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1).contiguous()
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)

        #### Attention 1 ####
        identity = out
        B, C, H, W = out.shape
        q_1 = out.view(H * W, B, C)
        k_1 = out.view(H * W, B, C)
        v_1 = out.view(H * W, B, C)
        out, attn_map_1 = self.attn1(q_1, k_1, v_1)

        # Residual connection
        out = identity + self.alpha * out

        out = self.layer4(out)

        #### Attention 2 ####
        identity = out
        B, C, H, W = out.shape
        q_2 = out.view(H * W, B, C)
        k_2 = out.view(H * W, B, C)
        v_2 = out.view(H * W, B, C)
        out, attn_map_2 = self.attn1(q_2, k_2, v_2)

        out = self.output(out)

        return torch.tanh(out), attn_map_1, attn_map_2
