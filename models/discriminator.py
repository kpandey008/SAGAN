import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, conv_dim=64, n_heads=1):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.n_heads = n_heads

        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(3, self.conv_dim, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )

        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(self.conv_dim, self.conv_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )

        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(2 * self.conv_dim, 4 * self.conv_dim, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )

        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(4 * self.conv_dim, 8 * self.conv_dim, 4, 2, 1)),
            nn.LeakyReLU(0.1),
        )

        self.output = nn.Conv2d(8 * self.conv_dim, 1, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attn1 = nn.MultiheadAttention(4 * self.conv_dim, num_heads=self.n_heads)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.attn2 = nn.MultiheadAttention(8 * self.conv_dim, num_heads=self.n_heads)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

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

        out = self.layer4(out)

        #### Attention 2 ####
        identity = out
        B, C, H, W = out.shape
        q_2 = out.view(H * W, B, C)
        k_2 = out.view(H * W, B, C)
        v_2 = out.view(H * W, B, C)
        out, attn_map_2 = self.attn2(q_2, k_2, v_2)
        out = out.view(B, C, H, W)

        # Residual connection
        out = identity + self.beta * out
        out = self.avg_pool(self.output(out))

        return out.squeeze(), attn_map_1, attn_map_2
