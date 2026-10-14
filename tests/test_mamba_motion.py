import unittest

import numpy as np
import torch
import torch.nn as nn

import tracker.mamba_motion_predictor as mamba_motion
from tracker.multitracker import STrack, Tracker


class IdentityMamba(nn.Module):
    def __init__(self, d_model, d_conv=4, d_state=16, **_):
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.d_state = d_state

    def forward(self, sequence):
        return sequence

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        del max_seqlen
        return (
            torch.zeros(batch_size, self.d_model, self.d_conv, dtype=dtype),
            torch.zeros(batch_size, self.d_model, self.d_state, dtype=dtype),
        )

    def step(self, hidden_states, conv_state, ssm_state):
        conv_state.add_(1.0)
        ssm_state.add_(1.0)
        return hidden_states, conv_state, ssm_state


class ConstantDeltaPredictor(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))

    def generate(self, histories, bev_grid_shape=None):
        return self.delta.cpu().numpy()[None].repeat(len(histories), axis=0)


class ConstantStatefulPredictor(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))
        self.seen_tokens = []

    def stateful_step(self, tokens, states=None, bev_grid_shape=None):
        del bev_grid_shape
        tokens = np.asarray(tokens, dtype=np.float32)
        self.seen_tokens.extend(tokens.copy())
        if states is None:
            states = [None] * len(tokens)
        next_states = [1 if state is None else state + 1 for state in states]
        motion = self.delta.cpu().numpy()[None].repeat(len(tokens), axis=0)
        return motion, next_states


class ConstantDistributionPredictor(ConstantDeltaPredictor):
    def __init__(self, delta, covariance):
        super().__init__(delta)
        self.register_buffer(
            "covariance",
            torch.tensor(covariance, dtype=torch.float32),
        )

    def generate_distribution(self, histories, bev_grid_shape=None):
        del bev_grid_shape
        count = len(histories)
        motion = self.delta.cpu().numpy()[None].repeat(count, axis=0)
        covariance = self.covariance.cpu().numpy()[None].repeat(count, axis=0)
        return motion, covariance


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

    def test_implicit_padding_repeats_first_source_and_only_zeros_velocity(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=3,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
            uncertainty_mode="unified",
        )
        history = torch.tensor([[[
            100.0, 200.0, 5.0, -2.0,
            0.25, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]]])

        padded = predictor._pad_history(history)

        self.assertEqual(tuple(padded.shape), (1, 3, 10))
        torch.testing.assert_close(padded[0, 0, :2], history[0, 0, :2])
        torch.testing.assert_close(padded[0, 0, 2:4], torch.zeros(2))
        torch.testing.assert_close(padded[0, 0, 4:], history[0, 0, 4:])

    def test_implicit_covariance_is_normalized_by_bev_axis_scales(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=1,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
            uncertainty_mode="unified",
        )
        history = torch.tensor([[[
            100.0, 50.0, 20.0, 10.0,
            0.0, 400.0, 100.0, 20.0, 0.0, 1.0,
        ]]])

        normalized = predictor._normalize_history(history, (100, 200))

        torch.testing.assert_close(
            normalized[0, 0],
            torch.tensor([0.5, 0.5, 0.1, 0.1, 0.0, 0.01, 0.01, 0.001, 0.0, 1.0]),
        )

    def test_separate_implicit_projection_routes_tokens_by_source(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=2,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
            uncertainty_mode="separate",
        )
        history = torch.tensor([[
            [100.0, 200.0, 5.0, -2.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0],
            [105.0, 198.0, 5.0, -2.0, 0.0, 4.0, 9.0, 1.0, 0.0, 1.0],
        ]])

        projected = predictor._project_history(history)

        self.assertEqual(tuple(projected.shape), (1, 2, 8))
        torch.testing.assert_close(
            projected[:, 0],
            predictor.input_projection(history[:, 0, :4])
            + predictor.detection_input_projection(history[:, 0, 4:5]),
        )
        torch.testing.assert_close(
            projected[:, 1],
            predictor.input_projection(history[:, 1, :4])
            + predictor.motion_input_projection(history[:, 1, 5:8]),
        )

    def test_full_separate_projection_routes_complete_source_tokens(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=2,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
            uncertainty_mode="separate_full",
        )
        history = torch.tensor([[
            [100.0, 200.0, 5.0, -2.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0],
            [105.0, 198.0, 5.0, -2.0, 0.0, 4.0, 9.0, 1.0, 0.0, 1.0],
        ]])

        projected = predictor._project_history(history)
        detection_input = history[:, 0, [0, 1, 2, 3, 4]]
        motion_input = history[:, 1, [0, 1, 2, 3, 5, 6, 7]]

        self.assertEqual(predictor.detection_input_projection.in_features, 5)
        self.assertEqual(predictor.motion_input_projection.in_features, 7)
        torch.testing.assert_close(
            projected[:, 0],
            predictor.detection_input_projection(detection_input),
        )
        torch.testing.assert_close(
            projected[:, 1],
            predictor.motion_input_projection(motion_input),
        )

    def test_unified_implicit_generate_accepts_mixed_history(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=2,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
            uncertainty_mode="unified",
        )
        history = np.asarray([
            [100.0, 200.0, 5.0, -2.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0],
            [105.0, 198.0, 6.0, -1.0, 0.0, 4.0, 9.0, 1.0, 0.0, 1.0],
        ], dtype=np.float32)

        prediction = predictor.generate([history], bev_grid_shape=(1000, 2000))

        np.testing.assert_allclose(prediction[0], [6.0, -1.0], atol=1e-5)

    def _check_niw_head(self, head_type):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=2,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
            loss_type="niw",
            evidential_head=head_type,
        )
        history = torch.tensor(
            [[[100.0, 200.0, 5.0, -2.0], [105.0, 198.0, 5.0, -2.0]]]
        )
        target = torch.tensor([[5.0, -2.0]])
        grid_shape = torch.tensor([[1000.0, 2000.0]])

        loss = predictor(history, target, bev_grid_shape=grid_shape)
        delta, covariance = predictor.generate_distribution(
            [history[0].numpy()],
            bev_grid_shape=(1000.0, 2000.0),
        )

        self.assertTrue(torch.isfinite(loss))
        np.testing.assert_allclose(delta[0], [5.0, -2.0], atol=1e-5)
        self.assertEqual(covariance.shape, (1, 2, 2))
        self.assertGreater(covariance[0, 0, 0], 0.0)
        self.assertGreater(covariance[0, 1, 1], 0.0)
        np.testing.assert_allclose(covariance[0], covariance[0].T, atol=1e-6)

    def test_separate_niw_head_returns_finite_loss_and_bev_covariance(self):
        self._check_niw_head("separate")

    def test_joint_niw_head_returns_finite_loss_and_bev_covariance(self):
        self._check_niw_head("joint")

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

    def test_tracker_records_motion_covariance_without_using_it_for_update(self):
        predictor = ConstantDistributionPredictor(
            [5.0, -2.0],
            [[4.0, 1.0], [1.0, 9.0]],
        )
        tracker = Tracker(
            high_cls_thres=0.4,
            match_thresh=30.0,
            motion_model="mamba",
            motion_normalization="bev",
            bev_grid_shape=(100, 200),
            motion_predictor=predictor,
            motion_history_len=5,
        )

        tracker.update([[10.0, 20.0]], scores=[0.9])
        tracks = tracker.update([[16.0, 19.0]], scores=[0.95])

        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose(tracks[0].xy, [16.0, 19.0])
        self.assertEqual(len(tracker.last_motion_predictions), 1)
        row = tracker.last_motion_predictions[0]
        self.assertEqual(row["matched"], 1)
        self.assertAlmostEqual(row["cov_xx"], 4.0)
        self.assertAlmostEqual(row["cov_xy"], 1.0)
        self.assertAlmostEqual(row["cov_yy"], 9.0)

    def test_implicit_tracker_uses_detection_uncertainty_and_motion_covariance(self):
        predictor = ConstantDistributionPredictor(
            [5.0, -2.0],
            [[4.0, 1.0], [1.0, 9.0]],
        )
        tracker = Tracker(
            high_cls_thres=0.4,
            match_thresh=30.0,
            motion_model="mamba",
            motion_normalization="bev",
            bev_grid_shape=(100, 200),
            motion_predictor=predictor,
            motion_history_len=5,
            mamba_uncertainty_mode="unified",
        )

        tracks = tracker.update(
            [[10.0, 20.0]],
            scores=[0.9],
            uncertainties=[0.25],
        )
        np.testing.assert_allclose(
            tracks[0].conds[-1],
            [10.0, 20.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

        tracks = tracker.update(
            [[16.0, 19.0]],
            scores=[0.95],
            uncertainties=[0.4],
        )
        np.testing.assert_allclose(
            tracks[0].conds[-1],
            [16.0, 19.0, 6.0, -1.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

        tracker.update([], scores=[], uncertainties=[])
        motion_token = tracker.lost_stracks[0].conds[-1]
        np.testing.assert_allclose(
            motion_token,
            [21.0, 17.0, 5.0, -2.0, 0.0, 4.0, 9.0, 1.0, 0.0, 1.0],
        )

    def test_stateful_step_carries_each_layer_cache_between_tokens(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=5,
            d_model=8,
            d_state=3,
            d_conv=2,
            num_layers=2,
            motion_normalization="bev",
            history_mode="stateful",
        )

        first_delta, states = predictor.stateful_step(
            [[100.0, 200.0, 5.0, -2.0]],
            bev_grid_shape=(1000, 2000),
        )
        second_delta, states = predictor.stateful_step(
            [[105.0, 198.0, 6.0, -1.0]],
            states=states,
            bev_grid_shape=(1000, 2000),
        )

        np.testing.assert_allclose(first_delta[0], [5.0, -2.0], atol=1e-5)
        np.testing.assert_allclose(second_delta[0], [6.0, -1.0], atol=1e-5)
        self.assertEqual(len(states[0]), 2)
        self.assertTrue(torch.all(states[0][0][0] == 2.0))
        self.assertTrue(torch.all(states[0][1][1] == 2.0))

    def test_stateful_step_accepts_implicit_motion_token(self):
        predictor = mamba_motion.MambaMotionPredictor(
            history_len=5,
            d_model=8,
            num_layers=1,
            motion_normalization="bev",
            history_mode="stateful",
            uncertainty_mode="unified",
        )
        token = [[
            105.0, 198.0, 6.0, -1.0,
            0.0, 4.0, 9.0, 1.0, 0.0, 1.0,
        ]]

        delta, states = predictor.stateful_step(
            token,
            bev_grid_shape=(1000, 2000),
        )

        np.testing.assert_allclose(delta[0], [6.0, -1.0], atol=1e-5)
        self.assertEqual(len(states[0]), 1)

    def test_stateful_tracker_commits_detection_corrected_token(self):
        predictor = ConstantStatefulPredictor([5.0, -2.0])
        tracker = Tracker(
            high_cls_thres=0.4,
            match_thresh=30.0,
            motion_model="mamba",
            motion_normalization="bev",
            bev_grid_shape=(100, 200),
            motion_predictor=predictor,
            motion_history_len=5,
            mamba_history_mode="stateful",
        )

        tracks = tracker.update([[10.0, 20.0]], scores=[0.9])
        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose(predictor.seen_tokens[-1], [10.0, 20.0, 0.0, 0.0])

        tracks = tracker.update([[16.0, 19.0]], scores=[0.95])
        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose(tracks[0].xy, [16.0, 19.0])
        np.testing.assert_allclose(predictor.seen_tokens[-1], [16.0, 19.0, 6.0, -1.0])
        self.assertEqual(tracks[0].mamba_state, 2)


if __name__ == "__main__":
    unittest.main()
