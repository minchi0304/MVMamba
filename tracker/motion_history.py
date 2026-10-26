"""Trajectory tokens shared by online tracking and detector-history training."""

import numpy as np


def detection_token(xy, delta, uncertainty, mode="none", gate="none"):
    kinematics = np.concatenate((xy, delta)).astype(np.float32)
    if mode == "none" and gate == "none":
        return kinematics
    token = np.concatenate((kinematics, np.asarray(
        [uncertainty, 0., 0., 0., 1., 0.], dtype=np.float32,
    )))
    if gate != "none":
        token = np.append(token, np.float32(0.))
    return token


def motion_token(xy, delta, covariance, kappa=None, mode="none", gate="none"):
    kinematics = np.concatenate((xy, delta)).astype(np.float32)
    if mode == "none" and gate == "none":
        return kinematics
    covariance_values = [0., 0., 0.] if covariance is None else [
        covariance[0, 0], covariance[1, 1], covariance[0, 1],
    ]
    token = np.concatenate((kinematics, np.asarray(
        [0., *covariance_values, 0., 1.], dtype=np.float32,
    )))
    if gate != "none":
        if kappa is None:
            raise ValueError("Explicit uncertainty gating requires predicted NIW kappa")
        token = np.append(token, np.float32(kappa))
    return token
