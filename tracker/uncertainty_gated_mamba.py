"""Mamba-1 with a fixed, per-token reliability gate on B or positive delta."""

import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class UncertaintyGatedMamba(Mamba):
    """Keep Mamba's parameters/checkpoints; change only its SSM recurrence.

    ``reliability`` is [batch, length] for forward and [batch] for step.
    Neither the convolution cache nor the D skip/residual is gated.
    """

    def __init__(self, *args, gate_mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_mode = gate_mode

    def forward(self, hidden_states, reliability=None):
        if reliability is None:
            return super().forward(hidden_states)

        length = hidden_states.size(1)
        x, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :length])
        dt, B, C = self.x_proj(x.transpose(1, 2)).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Gate the positive step size, not the pre-softplus logits/bias.
        dt = F.softplus(F.linear(dt, self.dt_proj.weight, self.dt_proj.bias))
        dt = dt.transpose(1, 2)
        B, C = B.transpose(1, 2), C.transpose(1, 2)
        gate = reliability.to(dtype=x.dtype).unsqueeze(1)
        if self.gate_mode == "b":
            B = B * gate
        else:
            dt = dt * gate

        y = selective_scan_fn(
            x, dt, -self.A_log.float().exp(), B, C,
            self.D.float(), z=z, delta_softplus=False,
        )
        return self.out_proj(y.transpose(1, 2))

    def step(self, hidden_states, conv_state, ssm_state, reliability=None):
        if reliability is None:
            return super().step(hidden_states, conv_state, ssm_state)

        dtype = hidden_states.dtype
        x, z = self.in_proj(hidden_states.squeeze(1)).chunk(2, dim=-1)
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        x = (conv_state * self.conv1d.weight[:, 0, :]).sum(dim=-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = self.act(x).to(dtype=dtype)
        dt, B, C = self.x_proj(x).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(F.linear(dt, self.dt_proj.weight, self.dt_proj.bias))
        gate = reliability.to(dtype=dt.dtype).reshape(-1, 1)
        if self.gate_mode == "b":
            B = B * gate
        else:
            dt = dt * gate

        A = -self.A_log.float().exp()
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + x.unsqueeze(-1) * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = (y + self.D.to(dtype) * x) * self.act(z)
        return self.out_proj(y).unsqueeze(1), conv_state, ssm_state
