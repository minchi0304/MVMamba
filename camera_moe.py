from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvExpert(nn.Module):
    """
    Lightweight spatial expert for one projected camera feature map.

    Input and output shape:
        [num_views, channels, height, width]
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: Optional[int] = None,
        kernel_size: int = 3,
        residual: bool = True,
    ):
        super().__init__()
        hidden_channels = channels if hidden_channels is None else int(hidden_channels)
        padding = kernel_size // 2
        self.residual = bool(residual)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False),
            _group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            _group_norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.residual:
            y = x + y
        return y


class CameraWiseMoE(nn.Module):
    """
    Camera-wise MoE for projected BEV features.

    This module does not estimate camera pose explicitly. It lets experts learn
    pose-dependent feature characteristics after camera features are projected
    onto the BEV grid.

    Args:
        channels: Number of feature channels per projected camera feature map.
        num_experts: Number of camera/view experts.
        pose_dim: Optional dimension of a camera pose or camera descriptor vector.
        hidden_channels: Hidden channels inside each convolutional expert.
        gate_hidden_channels: Hidden channels for the feature gate.
        pose_hidden_dim: Hidden dimension for pose conditioning.
        top_k: If set, keep only top-k experts per camera and cell.
        residual_experts: Whether each expert uses a residual connection.

    Inputs:
        x: Projected camera features with shape [num_views, channels, height, width].
        pose: Optional camera descriptor with shape [num_views, pose_dim].

    Returns:
        mixed: Expert-mixed features with the same shape as x.
        expert_weights: Expert weights with shape [num_views, num_experts, height, width].
    """

    def __init__(
        self,
        channels: int,
        num_experts: int,
        pose_dim: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        gate_hidden_channels: Optional[int] = None,
        pose_hidden_dim: int = 64,
        top_k: Optional[int] = None,
        residual_experts: bool = True,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k is not None and (top_k < 1 or top_k > num_experts):
            raise ValueError("top_k must be None or in [1, num_experts]")

        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.pose_dim = None if pose_dim is None else int(pose_dim)
        self.top_k = top_k

        self.experts = nn.ModuleList(
            [
                ConvExpert(
                    channels=self.channels,
                    hidden_channels=hidden_channels,
                    residual=residual_experts,
                )
                for _ in range(self.num_experts)
            ]
        )

        gate_hidden_channels = self.channels if gate_hidden_channels is None else int(gate_hidden_channels)
        gate_in_channels = self.channels
        if self.pose_dim is not None:
            self.pose_mlp = nn.Sequential(
                nn.Linear(self.pose_dim, pose_hidden_dim),
                nn.GELU(),
                nn.Linear(pose_hidden_dim, self.channels),
            )
            gate_in_channels += self.channels
        else:
            self.pose_mlp = None

        self.gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, gate_hidden_channels, kernel_size=1, bias=False),
            _group_norm(gate_hidden_channels),
            nn.GELU(),
            nn.Conv2d(gate_hidden_channels, self.num_experts, kernel_size=1),
        )

    def _gate_input(self, x: torch.Tensor, pose: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pose_mlp is None:
            return x
        if pose is None:
            raise ValueError("pose must be provided when pose_dim is set")
        if pose.dim() != 2 or pose.size(0) != x.size(0) or pose.size(1) != self.pose_dim:
            raise ValueError("pose must have shape [num_views, pose_dim]")

        pose_feature = self.pose_mlp(pose.to(device=x.device, dtype=x.dtype))
        pose_feature = pose_feature[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        return torch.cat([x, pose_feature], dim=1)

    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        if self.top_k is None or self.top_k == self.num_experts:
            return logits

        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=1)
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, top_indices, top_values)
        return filtered_logits

    def forward(
        self,
        x: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError("x must have shape [num_views, channels, height, width]")
        if x.size(1) != self.channels:
            raise ValueError("x channel dimension does not match channels")

        gate_input = self._gate_input(x, pose)
        logits = self.gate(gate_input)
        logits = self._apply_top_k(logits)
        expert_weights = torch.softmax(logits, dim=1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (expert_outputs * expert_weights.unsqueeze(2)).sum(dim=1)
        return mixed, expert_weights


class CameraWiseMoEAggregation(nn.Module):
    """
    Camera-wise MoE followed by learnable view aggregation.

    Inputs:
        x: Projected camera features with shape [num_views, channels, height, width].
        pose: Optional camera descriptor with shape [num_views, pose_dim].

    Returns:
        aggregated: BEV feature with shape [1, channels, height, width].
        expert_weights: Expert weights from CameraWiseMoE.
        view_weights: Camera weights with shape [num_views, 1, height, width].
    """

    def __init__(
        self,
        channels: int,
        num_experts: int,
        pose_dim: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        gate_hidden_channels: Optional[int] = None,
        pose_hidden_dim: int = 64,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.moe = CameraWiseMoE(
            channels=channels,
            num_experts=num_experts,
            pose_dim=pose_dim,
            hidden_channels=hidden_channels,
            gate_hidden_channels=gate_hidden_channels,
            pose_hidden_dim=pose_hidden_dim,
            top_k=top_k,
        )
        self.view_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed, expert_weights = self.moe(x, pose=pose)
        view_logits = self.view_gate(mixed)
        view_weights = torch.softmax(view_logits, dim=0)
        aggregated = (mixed * view_weights).sum(dim=0, keepdim=True)
        return aggregated, expert_weights, view_weights
