import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # Keep non-Mamba baselines importable without the optional package.
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        return sequence + self.dropout(self.mamba(self.norm(sequence)))


class MambaMotionPredictor(nn.Module):
    """Predict the next forward BEV displacement from a track history.

    Each history token is ``[x, y, dx, dy]``. Coordinates and displacements are
    normalized by ``[BEV width, BEV height]`` inside this module. The prediction
    returned by :meth:`generate` is converted back to world-grid units so the
    tracker can update its state as ``previous_xy + predicted_delta``.
    """

    def __init__(
        self,
        history_len=5,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=2,
        dropout=0.0,
        motion_normalization="bev",
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "Mamba motion requires the mamba-ssm package. "
                "Install the project dependencies or rebuild the Docker image."
            ) from _MAMBA_IMPORT_ERROR
        if history_len < 1:
            raise ValueError("history_len must be at least 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if motion_normalization not in ("bev", "none"):
            raise ValueError("motion_normalization must be 'bev' or 'none'")

        self.history_len = int(history_len)
        self.motion_normalization = motion_normalization
        self.input_projection = nn.Linear(4, d_model)
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, self.history_len, d_model)
        )
        self.blocks = nn.ModuleList(
            [
                ResidualMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
        )
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    @staticmethod
    def _coordinate_scale(bev_grid_shape, reference):
        if torch.is_tensor(bev_grid_shape):
            grid_shape = bev_grid_shape.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            grid_shape = reference.new_tensor(bev_grid_shape)
        if grid_shape.dim() == 1:
            if grid_shape.numel() != 2:
                raise ValueError("bev_grid_shape must contain [height, width]")
            return grid_shape[[1, 0]]
        if grid_shape.dim() == 2 and grid_shape.size(-1) == 2:
            return grid_shape[:, [1, 0]]
        raise ValueError("bev_grid_shape must be [2] or [batch, 2]")

    def _normalize_history(self, history, bev_grid_shape):
        if self.motion_normalization == "none":
            return history
        scale = self._coordinate_scale(bev_grid_shape, history)
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
        history = history.clone()
        history[..., :2] = history[..., :2] / scale
        history[..., 2:] = history[..., 2:] / scale
        return history

    def _normalize_delta(self, delta, bev_grid_shape):
        if self.motion_normalization == "none":
            return delta
        return delta / self._coordinate_scale(bev_grid_shape, delta)

    def _denormalize_delta(self, delta, bev_grid_shape):
        if self.motion_normalization == "none":
            return delta
        return delta * self._coordinate_scale(bev_grid_shape, delta)

    def _pad_history(self, history):
        if history.size(1) >= self.history_len:
            return history[:, -self.history_len :]

        pad_count = self.history_len - history.size(1)
        first = history[:, :1].clone()
        first[..., 2:] = 0.0
        padding = first.repeat(1, pad_count, 1)
        return torch.cat((padding, history), dim=1)

    def predict_normalized_delta(self, history, bev_grid_shape=None):
        if history.dim() != 3 or history.size(-1) != 4:
            raise ValueError("history must have shape [batch, time, 4]")
        if self.motion_normalization == "bev" and bev_grid_shape is None:
            raise ValueError("bev_grid_shape is required for BEV normalization")

        history = self._pad_history(history)
        history = self._normalize_history(history, bev_grid_shape)
        last_velocity = history[:, -1, 2:]
        sequence = self.input_projection(history) + self.temporal_embedding
        for block in self.blocks:
            sequence = block(sequence)
        correction = self.delta_head(self.output_norm(sequence[:, -1]))
        return last_velocity + correction

    def forward(self, history, target_delta, bev_grid_shape=None):
        prediction = self.predict_normalized_delta(history, bev_grid_shape)
        target = self._normalize_delta(target_delta, bev_grid_shape)
        return F.smooth_l1_loss(prediction, target, beta=0.01)

    def generate(self, histories, bev_grid_shape=None):
        if len(histories) == 0:
            return np.empty((0, 2), dtype=np.float32)

        device = next(self.parameters()).device
        prepared = []
        for history in histories:
            history_tensor = torch.as_tensor(
                np.asarray(history),
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 4)
            if history_tensor.size(0) == 0:
                raise ValueError("each track must contain at least one history token")
            prepared.append(self._pad_history(history_tensor.unsqueeze(0)))

        history_batch = torch.cat(prepared, dim=0)
        normalized_delta = self.predict_normalized_delta(
            history_batch,
            bev_grid_shape,
        )
        delta = self._denormalize_delta(normalized_delta, bev_grid_shape)
        return delta.detach().cpu().numpy()
