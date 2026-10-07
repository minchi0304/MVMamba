import torch
from torch import nn


class CornerNetFocalLoss(nn.Module):
    """CornerNet-style modified focal loss for a one-channel center heatmap."""

    def forward(self, logits, target):
        if logits.dim() != 4 or logits.size(1) != 1:
            raise ValueError("focal occupancy prediction must be [B,1,H,W]")

        if target.dim() == 3:
            target = target.unsqueeze(1)
        if target.dim() != 4 or target.size(1) != 1:
            raise ValueError("focal occupancy target must be [B,1,H,W]")

        target = target.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
        probability = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)

        positive = target.eq(1.0).to(logits.dtype)
        negative = target.lt(1.0).to(logits.dtype)
        negative_weight = torch.pow(1.0 - target, 4.0)

        positive_loss = (
            torch.log(probability)
            * torch.pow(1.0 - probability, 2.0)
            * positive
        )
        negative_loss = (
            torch.log(1.0 - probability)
            * torch.pow(probability, 2.0)
            * negative_weight
            * negative
        )

        num_positive = positive.sum()
        if num_positive.item() == 0:
            return -negative_loss.sum()
        return -(positive_loss.sum() + negative_loss.sum()) / num_positive
