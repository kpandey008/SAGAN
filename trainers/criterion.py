import torch
import torch.nn as nn


class GeneratorHingeLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(GeneratorHingeLoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("Valid values for the reduction param are `mean`, `sum`")
        self.reduction = reduction

    def forward(self, out):
        if self.reduction == "mean":
            return -out.mean()
        else:
            return out.sum()


class DiscriminatorHingeLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(DiscriminatorHingeLoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("Valid values for the reduction param are `mean`, `sum`")
        self.reduction = reduction

    def forward(self, fake_out, real_out):
        real_loss = -torch.minimum(torch.zeros_like(real_out), real_out - 1).mean()
        fake_loss = -torch.minimum(torch.zeros_like(fake_out), -1 - fake_out).mean()
        return real_loss + fake_loss
