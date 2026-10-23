"""Checkpoint and parameter boundaries for detector-then-motion training."""

import torch


def load_detector_weights(model, checkpoint):
    """Accept detector-only or existing combined state dictionaries."""
    state = torch.load(checkpoint, map_location='cpu')
    detector_state = {
        name: value for name, value in state.items()
        if not name.startswith('motion_predictor.')
        and name not in ('instance_weight', 'motion_weight')
    }
    loaded = model.load_state_dict(detector_state, strict=False)
    missing = [name for name in loaded.missing_keys
               if not name.startswith('motion_predictor.')
               and name not in ('instance_weight', 'motion_weight')]
    if missing or loaded.unexpected_keys:
        raise ValueError('Detector checkpoint mismatch: missing={}, unexpected={}'.format(
            missing, loaded.unexpected_keys))


def freeze_detector(model):
    """Freeze detector parameters and buffers; train only the attached predictor."""
    model.requires_grad_(False)
    model.eval()
    model.motion_predictor.requires_grad_(True)
