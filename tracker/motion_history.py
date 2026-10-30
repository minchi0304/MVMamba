"""Trajectory tokens shared by online tracking and detector-history training."""

import numpy as np


def history_token_width(mode="none", gate="none", implicit_motion_uncertainty_type="total"):
    if gate != "none" or (mode != "none" and implicit_motion_uncertainty_type != "total"):
        return 11  # Total covariance plus NIW kappa for component selection.
    return 4 if mode == "none" else 10


def detection_token(xy, delta, uncertainty, mode="none", gate="none",
                    implicit_motion_uncertainty_type="total"):
    kinematics = np.concatenate((xy, delta)).astype(np.float32)
    if mode == "none" and gate == "none":
        return kinematics
    token = np.concatenate((kinematics, np.asarray(
        [uncertainty, 0., 0., 0., 1., 0.], dtype=np.float32,
    )))
    if history_token_width(mode, gate, implicit_motion_uncertainty_type) == 11:
        token = np.append(token, np.float32(0.))
    return token


def motion_token(xy, delta, covariance, kappa=None, mode="none", gate="none",
                 implicit_motion_uncertainty_type="total"):
    kinematics = np.concatenate((xy, delta)).astype(np.float32)
    if mode == "none" and gate == "none":
        return kinematics
    covariance_values = [0., 0., 0.] if covariance is None else [
        covariance[0, 0], covariance[1, 1], covariance[0, 1],
    ]
    token = np.concatenate((kinematics, np.asarray(
        [0., *covariance_values, 0., 1.], dtype=np.float32,
    )))
    if history_token_width(mode, gate, implicit_motion_uncertainty_type) == 11:
        if kappa is None:
            raise ValueError("Uncertainty component selection or explicit gating requires predicted NIW kappa")
        token = np.append(token, np.float32(kappa))
    return token
