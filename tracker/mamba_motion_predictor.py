import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton.language as triton_language
from mamba_ssm import Mamba
from mamba_ssm.modules import mamba_simple
from tracker.uncertainty_gated_mamba import UncertaintyGatedMamba
from tracker.motion_history import history_token_width
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
    def __init__(self, d_model, d_state, d_conv, expand, dropout, uncertainty_gate="none"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        mamba_class = Mamba if uncertainty_gate == "none" else UncertaintyGatedMamba
        gate_kwargs = {} if uncertainty_gate == "none" else {"gate_mode": uncertainty_gate}
        self.mamba = mamba_class(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **gate_kwargs,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, reliability=None):
        normalized = self.norm(sequence)
        output = (
            self.mamba(normalized)
            if reliability is None
            else self.mamba(normalized, reliability=reliability)
        )
        return sequence + self.dropout(output)


class MambaMotionPredictor(nn.Module):
    """Predict the next forward BEV displacement from a track history.

    The baseline history token is ``[x, y, dx, dy]``. Implicit uncertainty
    modes use ``[x, y, dx, dy, u_det, cov_xx, cov_yy, cov_xy, is_det,
    is_motion]``. Component selection or explicit gating appends NIW kappa as an eleventh field;
    kappa is used only to split total covariance, never projected as a feature.
    Coordinates, displacements, and covariance are normalized by
    the BEV size inside this module. The prediction returned by :meth:`generate`
    is converted back to world-grid units so the tracker can update its state as
    ``previous_xy + predicted_delta``.
    """

    block_class = ResidualMambaBlock

    def __init__(
        self,
        history_len=5,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=2,
        dropout=0.0,
        history_mode="window",
        loss_type="smooth_l1",
        evidential_head="separate",
        uncertainty_mode="none",
        uncertainty_gate="none",
        motion_uncertainty_type="total",
        uncertainty_weight="cosine",
        niw_reg_weight=1e-3,
        implicit_motion_uncertainty_type="total",
    ):
        super().__init__()

        self.niw_reg_weight = float(niw_reg_weight)
        if not np.isfinite(self.niw_reg_weight) or self.niw_reg_weight < 0:
            raise ValueError("niw_reg_weight must be finite and non-negative")

        self.history_len = int(history_len)
        self.history_mode = history_mode
        self.loss_type = loss_type
        self.evidential_head_type = evidential_head
        self.uncertainty_mode = uncertainty_mode
        self.uncertainty_gate = uncertainty_gate
        if implicit_motion_uncertainty_type not in ("aleatoric", "epistemic", "total"):
            raise ValueError("implicit_motion_uncertainty_type must be aleatoric, epistemic, or total")
        if uncertainty_mode != "none" and implicit_motion_uncertainty_type != "total" and loss_type != "niw":
            raise ValueError("Implicit covariance component selection requires loss_type='niw'")
        self.implicit_motion_uncertainty_type = implicit_motion_uncertainty_type
        self.motion_uncertainty_type = motion_uncertainty_type
        self.uncertainty_weight = uncertainty_weight
        self.last_loss_stats = {}
        self.input_projection = nn.Linear(4, d_model)
        if self.uncertainty_mode == "unified":
            self.uncertainty_input_projection = nn.Linear(6, d_model, bias=False)
            self.detection_input_projection = None
            self.motion_input_projection = None
        elif self.uncertainty_mode == "separate":
            self.uncertainty_input_projection = None
            self.detection_input_projection = nn.Linear(1, d_model, bias=False)
            self.motion_input_projection = nn.Linear(3, d_model, bias=False)
        elif self.uncertainty_mode == "separate_full":
            self.uncertainty_input_projection = None
            self.detection_input_projection = nn.Linear(5, d_model)
            self.motion_input_projection = nn.Linear(7, d_model)
        else:
            self.uncertainty_input_projection = None
            self.detection_input_projection = None
            self.motion_input_projection = None
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, self.history_len, d_model)
        )
        self.blocks = nn.ModuleList(
            [
                self.block_class(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    uncertainty_gate=uncertainty_gate,
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
        if self.uncertainty_input_projection is not None:
            nn.init.zeros_(self.uncertainty_input_projection.weight)
        if self.uncertainty_mode == "separate":
            nn.init.zeros_(self.detection_input_projection.weight)
            nn.init.zeros_(self.motion_input_projection.weight)
        if self.uncertainty_mode == "separate_full":
            with torch.no_grad():
                self.detection_input_projection.weight[:, :4].copy_(
                    self.input_projection.weight
                )
                self.detection_input_projection.weight[:, 4:].zero_()
                self.detection_input_projection.bias.copy_(self.input_projection.bias)
                self.motion_input_projection.weight[:, :4].copy_(
                    self.input_projection.weight
                )
                self.motion_input_projection.weight[:, 4:].zero_()
                self.motion_input_projection.bias.copy_(self.input_projection.bias)

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
        scale = self._coordinate_scale(bev_grid_shape, history)
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
        history = history.clone()
        history[..., :2] = history[..., :2] / scale
        history[..., 2:4] = history[..., 2:4] / scale
        if history.size(-1) >= 10:
            scale_x = scale[..., 0]
            scale_y = scale[..., 1]
            history[..., 5] = history[..., 5] / scale_x.square()
            history[..., 6] = history[..., 6] / scale_y.square()
            history[..., 7] = history[..., 7] / (scale_x * scale_y)
        return history

    def _canonicalize_history(self, history):
        if self.uncertainty_mode == "none" and self.uncertainty_gate == "none":
            return history[..., :4]
        width = history_token_width(
            self.uncertainty_mode, self.uncertainty_gate, self.implicit_motion_uncertainty_type,
        )
        if history.size(-1) >= width:
            return history[..., :width]
        if history.size(-1) != 4:
            raise ValueError("Uncertainty component selection or explicit gating requires NIW kappa as field 11")

        # GT histories have no uncertainty: treat them as reliable observations.
        zeros = history.new_zeros((*history.shape[:-1], width - 4))
        zeros[..., 4] = 1.0
        return torch.cat((history[..., :4], zeros), dim=-1)

    def _project_history(self, history):
        kinematics = history[..., :4]
        projected = self.input_projection(kinematics)
        if self.uncertainty_mode == "none":
            return projected
        # History and the explicit gate retain total covariance. Only the
        # implicit projection sees the selected component, including cov_xy.
        covariance = history[..., 5:8]
        if self.implicit_motion_uncertainty_type != "total":
            kappa = history[..., 10:11]
            if self.implicit_motion_uncertainty_type == "aleatoric":
                covariance = covariance * kappa / (kappa + 1.0)
            else:
                covariance = covariance / (kappa + 1.0)
        if self.uncertainty_mode == "unified":
            uncertainty = torch.cat((history[..., 4:5], covariance, history[..., 8:10]), dim=-1)
            return projected + self.uncertainty_input_projection(uncertainty)
        if self.uncertainty_mode == "separate_full":
            detection_input = torch.cat((kinematics, history[..., 4:5]), dim=-1)
            motion_input = torch.cat((kinematics, covariance), dim=-1)
            detection_features = self.detection_input_projection(detection_input)
            motion_features = self.motion_input_projection(motion_input)
            return (
                history[..., 8:9] * detection_features
                + history[..., 9:10] * motion_features
            )

        detection_features = self.detection_input_projection(history[..., 4:5])
        motion_features = self.motion_input_projection(covariance)
        return (
            projected
            + history[..., 8:9] * detection_features
            + history[..., 9:10] * motion_features
        )

    def _history_reliability(self, history):
        """Reduce covariance after BEV normalization; return one gate per token."""
        if self.uncertainty_gate == "none":
            return None
        trace = (history[..., 5] + history[..., 6]).clamp_min(0.0)
        kappa = history[..., 10]
        if self.motion_uncertainty_type == "epistemic":
            trace = trace / (kappa + 1.0)
        elif self.motion_uncertainty_type == "aleatoric":
            trace = trace * kappa / (kappa + 1.0)
        detection_weight = self._uncertainty_to_weight(history[..., 4].clamp(0.0, 1.0))
        motion_weight = 1.0 - torch.tanh(trace)
        return (
            history[..., 8] * detection_weight
            + history[..., 9] * motion_weight
        )

    def _uncertainty_to_weight(self, uncertainty):
        if self.uncertainty_weight == "linear":
            return 1.0 - uncertainty
        if self.uncertainty_weight == "rational":
            return (1.0 - uncertainty) / (1.0 + uncertainty)
        return 0.5 * (1.0 + torch.cos(torch.pi * uncertainty))

    def _normalize_delta(self, delta, bev_grid_shape):
        return delta / self._coordinate_scale(bev_grid_shape, delta)

    def _denormalize_delta(self, delta, bev_grid_shape):
        return delta * self._coordinate_scale(bev_grid_shape, delta)

    def _denormalize_covariance(self, covariance, bev_grid_shape):
        scale = self._coordinate_scale(bev_grid_shape, covariance)
        return covariance * scale.unsqueeze(-1) * scale.unsqueeze(-2)

    def _pad_history(self, history):
        if history.size(1) >= self.history_len:
            return history[:, -self.history_len :]

        pad_count = self.history_len - history.size(1)
        first = history[:, :1].clone()
        first[..., 2:4] = 0.0
        padding = first.repeat(1, pad_count, 1)
        return torch.cat((padding, history), dim=1)

    def _encode_history(self, history, bev_grid_shape=None):
        history = self._canonicalize_history(history)
        history = self._pad_history(history)
        history = self._normalize_history(history, bev_grid_shape)
        reliability = self._history_reliability(history)
        last_velocity = history[:, -1, 2:4]
        sequence = self._project_history(history)
        if self.history_mode == "window":
            sequence = sequence + self.temporal_embedding
        for block in self.blocks:
            sequence = block(sequence, reliability=reliability)
        features = self.output_norm(self._readout_history(sequence))
        return features, last_velocity

    def _readout_history(self, sequence):
        return sequence[:, -1]

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
    def _stateful_step(self, tokens, states=None, bev_grid_shape=None, return_kappa=False):
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
        tokens = self._canonicalize_history(tokens)
        if bev_grid_shape is None:
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
        last_velocity = normalized_tokens[:, 2:4]
        reliability = self._history_reliability(normalized_tokens)
        sequence = self._project_history(normalized_tokens).unsqueeze(1)
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

            gate_kwargs = {} if reliability is None else {"reliability": reliability}
            mamba_output, conv_state, ssm_state = block.mamba.step(
                block.norm(sequence),
                conv_state,
                ssm_state,
                **gate_kwargs,
            )
            sequence = sequence + block.dropout(mamba_output)
            layer_states.append((conv_state, ssm_state))

        features = self.output_norm(sequence[:, -1])
        normalized_delta, normalized_covariance, kappa, _, _ = (
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
        result = (delta.detach().cpu().numpy(), covariance_numpy, next_states)
        if return_kappa:
            return (*result, None if kappa is None else kappa.detach().cpu().numpy())
        return result

    def stateful_step(self, tokens, states=None, bev_grid_shape=None):
        delta, _, next_states = self._stateful_step(
            tokens,
            states=states,
            bev_grid_shape=bev_grid_shape,
        )
        return delta, next_states

    def stateful_step_distribution(
        self, tokens, states=None, bev_grid_shape=None, return_kappa=False,
    ):
        return self._stateful_step(
            tokens,
            states=states,
            bev_grid_shape=bev_grid_shape,
            return_kappa=return_kappa,
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
                lambda_evi=self.niw_reg_weight,
            )
            return loss

        self.last_loss_stats = {}
        return F.smooth_l1_loss(prediction, target, beta=1)

    def generate(self, histories, bev_grid_shape=None):
        delta, _ = self.generate_distribution(histories, bev_grid_shape)
        return delta

    def generate_distribution(self, histories, bev_grid_shape=None, return_kappa=False):
        if len(histories) == 0:
            result = (np.empty((0, 2), dtype=np.float32), None)
            return (*result, None) if return_kappa else result

        device = next(self.parameters()).device
        prepared = []
        for history in histories:
            history_array = np.asarray(history)
            history_tensor = torch.as_tensor(
                history_array,
                dtype=torch.float32,
                device=device,
            ).reshape(-1, history_array.shape[-1])
            if history_tensor.size(0) == 0:
                raise ValueError("each track must contain at least one history token")
            history_tensor = self._canonicalize_history(history_tensor)
            prepared.append(self._pad_history(history_tensor.unsqueeze(0)))

        history_batch = torch.cat(prepared, dim=0)
        normalized_delta, normalized_covariance, kappa, _, _ = self.predict_normalized_distribution(
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
        result = (delta.detach().cpu().numpy(), covariance_numpy)
        if return_kappa:
            return (*result, None if kappa is None else kappa.detach().cpu().numpy())
        return result
