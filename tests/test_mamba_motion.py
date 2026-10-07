import unittest

import numpy as np
import torch
import torch.nn as nn

import tracker.mamba_motion_predictor as mamba_motion
from tracker.multitracker import STrack


class IdentityMamba(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def forward(self, sequence):
        return sequence


class ConstantDeltaPredictor(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))

    def generate(self, histories, bev_grid_shape=None):
        return self.delta.cpu().numpy()[None].repeat(len(histories), axis=0)


class MambaMotionPredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_mamba = mamba_motion.Mamba
        mamba_motion.Mamba = IdentityMamba

    @classmethod
    def tearDownClass(cls):
        mamba_motion.Mamba = cls.original_mamba

    def test_initial_prediction_reuses_last_forward_delta_in_bev_units(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=5,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
        )
        history = [
            np.asarray([100.0, 200.0, 10.0, -20.0], dtype=np.float32)
        ]

        prediction = predictor.generate([history], bev_grid_shape=(1000, 2000))

        np.testing.assert_allclose(prediction[0], [10.0, -20.0], atol=1e-5)

    def test_training_target_uses_forward_delta(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=2,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
        )
        history = torch.tensor(
            [[[100.0, 200.0, 5.0, -2.0], [105.0, 198.0, 5.0, -2.0]]]
        )
        target = torch.tensor([[5.0, -2.0]])

        loss = predictor(history, target, bev_grid_shape=torch.tensor([[1000.0, 2000.0]]))

        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)

    def test_detection_update_replaces_the_predicted_track_state(self):
        track = STrack([10.0, 20.0], 0.9, motion_history_len=5)
        track.activate(None, frame_id=1, motion_model="mamba")
        predictor = ConstantDeltaPredictor([5.0, -2.0])

        STrack.multi_predict_mamba([track], predictor, bev_grid_shape=(100, 200))
        np.testing.assert_allclose(track.xy, [15.0, 18.0])

        detection = STrack([16.0, 19.0], 0.95, motion_history_len=5)
        track.update(detection, frame_id=2, motion_model="mamba")

        np.testing.assert_allclose(track.xy, [16.0, 19.0])
        np.testing.assert_allclose(track.conds[-1], [16.0, 19.0, 6.0, -1.0])


if __name__ == "__main__":
    unittest.main()
