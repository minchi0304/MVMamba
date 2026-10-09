import torch
from torch import nn
import torch.nn.functional as F


class EvidentialOccupancyLoss(nn.Module):
    def __init__(self, lambda_reg=1e-4, gamma=3.0, eta=4.0):
        super().__init__()
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.eta = float(eta)

    def forward(self, prediction, center_target, gaussian_target):
        evidence = F.softplus(prediction) + 1.0
        alpha = evidence[:, 0:1]
        beta = evidence[:, 1:2]

        if center_target.dim() == 3:
            center_target = center_target.unsqueeze(1)
        if gaussian_target.dim() == 3:
            gaussian_target = gaussian_target.unsqueeze(1)
        center_target = center_target.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).clamp(0.0, 1.0)
        gaussian_target = gaussian_target.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).clamp(0.0, 1.0)

        probability = alpha / (alpha + beta)
        digamma_sum = torch.digamma(alpha + beta)
        positive_loss = (
            center_target
            * (digamma_sum - torch.digamma(alpha))
            * torch.pow(1.0 - probability, self.gamma)
        )
        negative_loss = (
            (1.0 - center_target)
            * (digamma_sum - torch.digamma(beta))
            * torch.pow(probability, self.gamma)
            * torch.pow((1.0 - gaussian_target).clamp_min(0.0), self.eta)
        )

        num_positive = center_target.eq(1.0).to(prediction.dtype).sum().clamp_min(1.0)
        evidential_loss = (positive_loss + negative_loss).sum() / num_positive

        alpha_tilde = center_target + (1.0 - center_target) * alpha
        beta_tilde = (1.0 - center_target) + center_target * beta
        log_beta = (
            torch.lgamma(alpha_tilde)
            + torch.lgamma(beta_tilde)
            - torch.lgamma(alpha_tilde + beta_tilde)
        )
        digamma_alpha_tilde = torch.digamma(alpha_tilde)
        digamma_beta_tilde = torch.digamma(beta_tilde)
        digamma_sum_tilde = torch.digamma(alpha_tilde + beta_tilde)
        regularizer = (
            (alpha_tilde - 1.0)
            * (digamma_alpha_tilde - digamma_sum_tilde)
            + (beta_tilde - 1.0)
            * (digamma_beta_tilde - digamma_sum_tilde)
            - log_beta
        )
        regularizer = regularizer.sum() / num_positive

        return evidential_loss + self.lambda_reg * regularizer
