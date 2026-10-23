"""Bidirectional encoding of the available past trajectory window."""

from torch import nn
from mamba_ssm import Mamba

from tracker.mamba_motion_predictor import MambaMotionPredictor
from tracker.uncertainty_gated_mamba import UncertaintyGatedMamba


class ResidualBiMambaBlock(nn.Module):
    """Independent forward/reverse Mamba branches with aligned mean fusion.

    Each branch has its own projections, convolution, and SSM parameters.
    This uses the installed Mamba kernels rather than Vim's custom BiMamba
    kernel. Reversing tokens does not change their coordinate/velocity values.
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout, uncertainty_gate="none"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        mamba_class = Mamba if uncertainty_gate == "none" else UncertaintyGatedMamba
        gate_kwargs = {} if uncertainty_gate == "none" else {"gate_mode": uncertainty_gate}
        branch_kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                             **gate_kwargs)
        self.forward_mamba = mamba_class(**branch_kwargs)
        self.backward_mamba = mamba_class(**branch_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, reliability=None):
        normalized = self.norm(sequence)
        reversed_sequence = normalized.flip(1)
        if reliability is None:
            forward = self.forward_mamba(normalized)
            backward = self.backward_mamba(reversed_sequence)
        else:
            forward = self.forward_mamba(normalized, reliability=reliability)
            backward = self.backward_mamba(reversed_sequence, reliability=reliability.flip(1))
        fused = 0.5 * (forward + backward.flip(1))
        return sequence + self.dropout(fused)


class BiMambaMotionPredictor(MambaMotionPredictor):
    """Predict the next displacement using only a bidirectional past window.

    Mean pooling reads both directions over the complete padded history;
    the velocity skip still uses the last chronological token. Padding,
    normalization, uncertainty handling, losses, and heads are inherited.
    """

    block_class = ResidualBiMambaBlock

    def __init__(self, **kwargs):
        if kwargs.get("history_mode", "window") != "window":
            raise ValueError("BiMamba requires history_mode='window'; reverse scans cannot use a one-token stateful cache")
        super().__init__(**kwargs)

    def _readout_history(self, sequence):
        return sequence.mean(dim=1)
