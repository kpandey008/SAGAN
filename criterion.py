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
        real_loss = -torch.min(0, real_out - 1)
        fake_loss = -torch.min(0, -1 - fake_out)
        return real_loss + fake_loss
