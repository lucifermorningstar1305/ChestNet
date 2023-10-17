"""
Source: https://discuss.pytorch.org/t/what-is-the-formula-for-cross-entropy-loss-with-label-smoothing/149848
Date: 10-17-2023

"""

from typing import Optional 

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing_factor: Optional[float]=.1, reduction: Optional[str]="mean"):

        super().__init__()

        assert reduction in ["sum", "mean"], f"Expected reduction to be either sum/mean. Found {reduction}"

        eps = smoothing_factor / num_classes

        self.negative = eps
        self.positive = (1 - smoothing_factor) + eps

        self.reduction = reduction

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        pred = yhat.log_softmax(dim=-1)

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        
        true_dist.scatter_(1, y.data.unsqueeze(1), self.positive)

        if self.reduction == "sum":
            return torch.sum(-true_dist * pred, dim=1).sum()
        
        else:
            return torch.sum(-true_dist * pred, dim=1).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[float]=.25, gamma: Optional[float]=2., reduction: Optional[str]="mean"):
        super().__init__()
        
        assert reduction in ["mean", "sum", "none"], f"Expected reduction to be either mean/sum/none. Found {reduction}"

        self.alpha = alpha
        self.gamma = gamma

        self.reduction = reduction

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.view(-1, 1).type_as(yhat)
        bce_loss = F.binary_cross_entropy_with_logits(yhat, y, reduction="none")
        focal_loss = self.alpha * torch.pow((1 - yhat), self.gamma) * bce_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss





