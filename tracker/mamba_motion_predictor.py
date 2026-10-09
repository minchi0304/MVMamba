import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton.language as triton_language
from mamba_ssm import Mamba
from mamba_ssm.modules import mamba_simple
from utils.losses_evidential import (
    evidential_niw_loss_2d,
    niw_predictive_covariance_2d,
)

if not hasattr(triton_language, "math"):
    # Torch 2.0 bundles Triton 2.0, whose language API is too old for the
    # optional Mamba 1.2 state-update kernel. Mamba's step method includes
    # an equivalent PyTorch fallback when this optional kernel is disabled.
    mamba_simple.selective_state_update = None


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
        history_mode="window",
        loss_type="smooth_l1",
        evidential_lambda=1e-3,
        evidential_head="separate",
    ):
        super().__init__()

        self.history_len = int(history_len)
        self.motion_normalization = motion_normalization
        self.history_mode = history_mode
        self.loss_type = loss_type
        self.evidential_lambda = float(evidential_lambda)
        self.evidential_head_type = evidential_head
        self.last_loss_stats = {}
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
        use_joint_head = self.loss_type == "niw" and evidential_head == "joint"
        self.delta_head = None if use_joint_head else nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
        )
        self.evidence_head = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 5),
            )
            if self.loss_type == "niw" and not use_joint_head
            else None
        )
        self.joint_head = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 7),
            )
            if use_joint_head
            else None
        )
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        if self.delta_head is not None:
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
        if self.evidence_head is not None:
            nn.init.zeros_(self.evidence_head[-1].weight)
            nn.init.zeros_(self.evidence_head[-1].bias)
            self.evidence_head[-1].bias.data.copy_(
                torch.tensor([0.5413, 0.5413, -4.6002, 0.0, -4.6002])
            )
        if self.joint_head is not None:
            nn.init.zeros_(self.joint_head[-1].weight)
            nn.init.zeros_(self.joint_head[-1].bias)
            self.joint_head[-1].bias.data[2:].copy_(
                torch.tensor([0.5413, 0.5413, -4.6002, 0.0, -4.6002])
            )

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

    def _denormalize_covariance(self, covariance, bev_grid_shape):
        if self.motion_normalization == "none":
            return covariance
        scale = self._coordinate_scale(bev_grid_shape, covariance)
        return covariance * scale.unsqueeze(-1) * scale.unsqueeze(-2)

    def _pad_history(self, history):
        if history.size(1) >= self.history_len:
            return history[:, -self.history_len :]

        pad_count = self.history_len - history.size(1)
        first = history[:, :1].clone()
        first[..., 2:] = 0.0
        padding = first.repeat(1, pad_count, 1)
        return torch.cat((padding, history), dim=1)

    def _encode_history(self, history, bev_grid_shape=None):
        history = self._pad_history(history)
        history = self._normalize_history(history, bev_grid_shape)
        last_velocity = history[:, -1, 2:]
        sequence = self.input_projection(history)
        if self.history_mode == "window":
            sequence = sequence + self.temporal_embedding
        for block in self.blocks:
            sequence = block(sequence)
        features = self.output_norm(sequence[:, -1])
        return features, last_velocity

    def _predict_distribution_from_features(self, features, last_velocity):
        if self.joint_head is not None:
            joint_output = self.joint_head(features)
            mean = last_velocity + joint_output[:, :2]
            raw = joint_output[:, 2:]
        else:
            mean = last_velocity + self.delta_head(features)
            raw = None if self.evidence_head is None else self.evidence_head(features)

        if raw is None:
            return mean, None, None, None, None

        kappa = F.softplus(raw[:, 0:1]) + 1e-4
        nu = F.softplus(raw[:, 1:2]) + 3.0 + 1e-4
        batch_size = raw.size(0)
        scale_tril = raw.new_zeros((batch_size, 2, 2))
        scale_tril[:, 0, 0] = F.softplus(raw[:, 2]) + 1e-4
        scale_tril[:, 1, 0] = raw[:, 3]
        scale_tril[:, 1, 1] = F.softplus(raw[:, 4]) + 1e-4
        covariance = niw_predictive_covariance_2d(
            kappa,
            nu,
            scale_tril,
        )
        return mean, covariance, kappa, nu, scale_tril

    def predict_normalized_distribution(self, history, bev_grid_shape=None):
        features, last_velocity = self._encode_history(history, bev_grid_shape)
        return self._predict_distribution_from_features(features, last_velocity)

    def predict_normalized_delta(self, history, bev_grid_shape=None):
        mean, _, _, _, _ = self.predict_normalized_distribution(
            history,
            bev_grid_shape,
        )
        return mean

    @torch.no_grad()
    def _stateful_step(self, tokens, states=None, bev_grid_shape=None):
        """Consume one token per track and return its next displacement.

        ``states`` contains one cache per track. Each cache stores the
        convolution and SSM state for every residual Mamba block. A ``None``
        entry starts a new track from zero state.
        """
        if self.history_mode != "stateful":
            raise RuntimeError("stateful_step requires history_mode='stateful'")

        device = next(self.parameters()).device
        tokens = torch.as_tensor(tokens, dtype=torch.float32, device=device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 2 or tokens.size(-1) != 4:
            raise ValueError("tokens must have shape [batch, 4]")
        if self.motion_normalization == "bev" and bev_grid_shape is None:
            raise ValueError("bev_grid_shape is required for BEV normalization")

        batch_size = tokens.size(0)
        if states is None:
            states = [None] * batch_size
        if len(states) != batch_size:
            raise ValueError("states must contain one entry per token")

        normalized_tokens = self._normalize_history(
            tokens.unsqueeze(1),
            bev_grid_shape,
        ).squeeze(1)
        last_velocity = normalized_tokens[:, 2:]
        sequence = self.input_projection(normalized_tokens).unsqueeze(1)
        layer_states = []

        for layer_index, block in enumerate(self.blocks):
            if not hasattr(block.mamba, "step") or not hasattr(
                block.mamba, "allocate_inference_cache"
            ):
                raise RuntimeError(
                    "the installed mamba-ssm version does not support recurrent inference"
                )

            conv_state, ssm_state = block.mamba.allocate_inference_cache(
                batch_size=batch_size,
                max_seqlen=1,
                dtype=sequence.dtype,
            )
            for row_index, track_state in enumerate(states):
                if track_state is None:
                    continue
                if len(track_state) != len(self.blocks):
                    raise ValueError("each track state must contain one cache per block")
                cached_conv, cached_ssm = track_state[layer_index]
                conv_state[row_index : row_index + 1].copy_(
                    cached_conv.to(device=conv_state.device, dtype=conv_state.dtype)
                )
                ssm_state[row_index : row_index + 1].copy_(
                    cached_ssm.to(device=ssm_state.device, dtype=ssm_state.dtype)
                )

            mamba_output, conv_state, ssm_state = block.mamba.step(
                block.norm(sequence),
                conv_state,
                ssm_state,
            )
            sequence = sequence + block.dropout(mamba_output)
            layer_states.append((conv_state, ssm_state))

        features = self.output_norm(sequence[:, -1])
        normalized_delta, normalized_covariance, _, _, _ = (
            self._predict_distribution_from_features(features, last_velocity)
        )
        delta = self._denormalize_delta(normalized_delta, bev_grid_shape)
        covariance = None
        if normalized_covariance is not None:
            covariance = self._denormalize_covariance(
                normalized_covariance,
                bev_grid_shape,
            )

        next_states = [[] for _ in range(batch_size)]
        for conv_state, ssm_state in layer_states:
            for row_index in range(batch_size):
                next_states[row_index].append(
                    (
                        conv_state[row_index : row_index + 1].detach().clone(),
                        ssm_state[row_index : row_index + 1].detach().clone(),
                    )
                )

        covariance_numpy = None
        if covariance is not None:
            covariance_numpy = covariance.detach().cpu().numpy()
        return delta.detach().cpu().numpy(), covariance_numpy, next_states

    def stateful_step(self, tokens, states=None, bev_grid_shape=None):
        delta, _, next_states = self._stateful_step(
            tokens,
            states=states,
            bev_grid_shape=bev_grid_shape,
        )
        return delta, next_states

    def stateful_step_distribution(self, tokens, states=None, bev_grid_shape=None):
        return self._stateful_step(
            tokens,
            states=states,
            bev_grid_shape=bev_grid_shape,
        )

    def forward(self, history, target_delta, bev_grid_shape=None):
        prediction, _, kappa, nu, scale_tril = self.predict_normalized_distribution(
            history,
            bev_grid_shape,
        )
        target = self._normalize_delta(target_delta, bev_grid_shape)
        if self.loss_type == "niw":
            loss, self.last_loss_stats = evidential_niw_loss_2d(
                target,
                prediction,
                kappa,
                nu,
                scale_tril,
                lambda_evi=self.evidential_lambda,
            )
            return loss

        self.last_loss_stats = {}
        return F.smooth_l1_loss(prediction, target, beta=1)

    def generate(self, histories, bev_grid_shape=None):
        delta, _ = self.generate_distribution(histories, bev_grid_shape)
        return delta

    def generate_distribution(self, histories, bev_grid_shape=None):
        if len(histories) == 0:
            return np.empty((0, 2), dtype=np.float32), None

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
        normalized_delta, normalized_covariance, _, _, _ = self.predict_normalized_distribution(
            history_batch,
            bev_grid_shape,
        )
        delta = self._denormalize_delta(normalized_delta, bev_grid_shape)
        covariance_numpy = None
        if normalized_covariance is not None:
            covariance = self._denormalize_covariance(
                normalized_covariance,
                bev_grid_shape,
            )
            covariance_numpy = covariance.detach().cpu().numpy()
        return delta.detach().cpu().numpy(), covariance_numpy
