import math
import unittest
from itertools import product

import numpy as np
import torch
from mamba_ssm import Mamba

from tracker.mamba_motion_predictor import MambaMotionPredictor
from tracker.multitracker import Tracker
from tracker.uncertainty_gated_mamba import UncertaintyGatedMamba


class ReliabilityTests(unittest.TestCase):
    def predictor(self, **kwargs):
        return MambaMotionPredictor(
            d_model=8, d_state=3, num_layers=1, loss_type="niw",
            uncertainty_gate="b", **kwargs,
        )

    def test_detection_weight_choices_and_source_selection(self):
        history = torch.zeros(1, 5, 11)
        history[..., 4] = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        history[..., 8] = 1
        # An inactive branch must not change a detection's reliability.
        history[..., 5:7] = 100
        expected_weights = {
            "linear": [1, .75, .5, .25, 0],
            "rational": [1, .6, 1 / 3, 1 / 7, 0],
            "cosine": [1, .853553390593, .5, .146446609407, 0],
        }
        for weighting, expected in expected_weights.items():
            with self.subTest(weighting=weighting):
                predictor = self.predictor(uncertainty_weight=weighting)
                gate = predictor._history_reliability(history)
                torch.testing.assert_close(gate, torch.tensor([expected]))

    def test_covariance_components_and_bev_units(self):
        # Grid [H=100,W=200], total covariance diag[4,9], kappa=3.
        # Normalized total trace = 4/200^2 + 9/100^2 = .001.
        token = torch.tensor([[[10, 20, 1, 2, 0.9, 4, 9, 1, 0, 1, 3.0]]])
        for component, trace in (("total", .001), ("aleatoric", .00075), ("epistemic", .00025)):
            with self.subTest(component=component):
                predictor = self.predictor(motion_uncertainty_type=component)
                normalized = predictor._normalize_history(token, (100, 200))
                self.assertEqual(normalized[0, 0, 10].item(), 3.0)
                self.assertAlmostEqual(normalized[0, 0, 7].item(), 1 / 20000)
                expected = 1 - math.tanh(trace)
                self.assertAlmostEqual(predictor._history_reliability(normalized).item(), expected, places=6)

    def test_gt_gate_one_and_padding_preserves_uncertainty(self):
        predictor = self.predictor(history_len=3)
        gt = predictor._canonicalize_history(torch.ones(2, 2, 4))
        torch.testing.assert_close(predictor._history_reliability(gt), torch.ones(2, 2))
        token = torch.tensor([[[10, 20, 1, 2, 0, 4, 9, 1, 0, 1, 3.0]]])
        padded = predictor._pad_history(token)
        torch.testing.assert_close(padded[0, :2, 2:4], torch.zeros(2, 2))
        torch.testing.assert_close(padded[0, :, 4:], token[0, :, 4:].expand(3, -1))

    def test_motion_tanh_ignores_detection_mapping(self):
        history = torch.zeros(1, 4, 11)
        history[..., 4] = .9  # Inactive detection uncertainty must be ignored.
        history[..., 5] = torch.tensor([0.0, .5, 1.0, 10.0])
        history[..., 9] = 1
        history[..., 10] = 3
        expected = torch.tensor([[1.0, .53788284274, .23840584404, 0.0]])
        for weighting in ("linear", "rational", "cosine"):
            with self.subTest(weighting=weighting):
                predictor = self.predictor(uncertainty_weight=weighting)
                torch.testing.assert_close(predictor._history_reliability(history), expected)

    def test_linear_detection_and_tanh_motion_use_correct_source_and_covariance(self):
        history = torch.tensor([[
            [10, 20, 1, 2, .2, 4, 9, 1, 1, 0, 3],
            [10, 20, 1, 2, .9, 4, 9, 1, 0, 1, 3],
        ]])
        # Grid [H=3,W=2] makes normalized total trace=2; kappa=3 splits
        # it into aleatoric=1.5 and epistemic=.5. Detection must stay at .8.
        for component, motion_weight in (
            ("total", .03597241992),
            ("aleatoric", .09485174636),
            ("epistemic", .53788284274),
        ):
            with self.subTest(component=component):
                predictor = self.predictor(
                    uncertainty_weight="linear",
                    motion_uncertainty_type=component,
                )
                normalized = predictor._normalize_history(history, (3, 2))
                torch.testing.assert_close(
                    predictor._history_reliability(normalized),
                    torch.tensor([[.8, motion_weight]]),
                )
                gt = predictor._canonicalize_history(history[..., :4])
                torch.testing.assert_close(predictor._history_reliability(gt), torch.ones(1, 2))


@unittest.skipUnless(torch.cuda.is_available(), "real Mamba sequence kernels require CUDA")
class GatedMambaCudaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def layer(self, mode):
        return UncertaintyGatedMamba(
            d_model=8, d_state=3, d_conv=4, expand=2, gate_mode=mode,
        ).cuda()

    def test_gate_one_matches_original_outputs_and_gradients(self):
        baseline = Mamba(d_model=8, d_state=3, d_conv=4, expand=2).cuda()
        inputs = torch.randn(2, 7, 8, device="cuda")
        for mode in ("b", "delta"):
            with self.subTest(mode=mode):
                gated = self.layer(mode)
                gated.load_state_dict(baseline.state_dict(), strict=True)
                x_base = inputs.clone().requires_grad_()
                x_gate = inputs.clone().requires_grad_()
                baseline.zero_grad()
                expected = baseline(x_base)
                actual = gated(x_gate, torch.ones(2, 7, device="cuda"))
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                expected.square().sum().backward()
                actual.square().sum().backward()
                torch.testing.assert_close(x_gate.grad, x_base.grad, atol=2e-5, rtol=2e-4)
                for name, parameter in gated.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad, dict(baseline.named_parameters())[name].grad,
                        atol=2e-5, rtol=2e-4,
                    )

    def test_sequence_matches_recurrent_steps_with_mixed_gates(self):
        inputs = torch.randn(2, 7, 8, device="cuda")
        gates = torch.tensor([[0, .2, .5, 1, .8, 0, 1], [1, .5, 0, .3, 1, .9, .4]], device="cuda")
        for mode in ("b", "delta"):
            with self.subTest(mode=mode), torch.no_grad():
                layer = self.layer(mode)
                conv, ssm = layer.allocate_inference_cache(2, 7)
                outputs = []
                for index in range(7):
                    y, conv, ssm = layer.step(inputs[:, index:index + 1], conv, ssm, gates[:, index])
                    outputs.append(y)
                torch.testing.assert_close(
                    torch.cat(outputs, dim=1), layer(inputs, gates), atol=2e-5, rtol=2e-4,
                )

    def test_zero_delta_freezes_ssm_but_zero_b_keeps_decay(self):
        token = torch.randn(2, 1, 8, device="cuda")
        for mode in ("b", "delta"):
            with self.subTest(mode=mode), torch.no_grad():
                layer = self.layer(mode)
                # Constant positive delta lets us check exact recurrence semantics.
                layer.dt_proj.weight.zero_()
                layer.dt_proj.bias.fill_(math.log(math.expm1(.1)))
                conv, ssm = layer.allocate_inference_cache(2, 1)
                ssm.normal_()
                before = ssm.clone()
                before_conv = conv.clone()
                layer.step(token, conv, ssm, torch.zeros(2, device="cuda"))
                expected = before if mode == "delta" else before * torch.exp(-.1 * layer.A_log.exp())
                torch.testing.assert_close(ssm, expected)
                self.assertFalse(torch.equal(conv, before_conv))

    def test_baseline_step_and_gated_step_agree_at_one(self):
        token = torch.randn(2, 1, 8, device="cuda")
        for mode in ("b", "delta"):
            with self.subTest(mode=mode), torch.no_grad():
                layer = self.layer(mode)
                conv, ssm = layer.allocate_inference_cache(2, 1)
                conv.normal_()
                ssm.normal_()
                expected = layer.step(token, conv.clone(), ssm.clone())
                actual = layer.step(token, conv.clone(), ssm.clone(), torch.ones(2, device="cuda"))
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)

    def predictor(self, mode, **kwargs):
        return MambaMotionPredictor(
            d_model=8, d_state=3, num_layers=2, history_len=3,
            loss_type="niw", uncertainty_gate=mode, **kwargs,
        ).cuda()

    def test_gate_changes_features_without_implicit_inputs(self):
        history = torch.tensor([[
            [10, 20, 1, 2, 0, 0, 0, 0, 1, 0, 0],
            [11, 22, 1, 2, 0, 4, 9, 1, 0, 1, 3],
            [12, 24, 1, 2, 0, 0, 0, 0, 1, 0, 0],
        ]], dtype=torch.float32, device="cuda")
        for mode in ("b", "delta"):
            with self.subTest(mode=mode), torch.no_grad():
                predictor = self.predictor(mode)
                gated, _ = predictor._encode_history(history, (100, 200))
                reliable = history.clone()
                reliable[..., 5:8] = 0
                ungated, _ = predictor._encode_history(reliable, (100, 200))
                self.assertGreater((gated - ungated).abs().max().item(), 1e-7)

    def test_niw_training_backpropagates_for_both_gates_and_heads(self):
        for mode in ("b", "delta"):
            for head in ("separate", "joint"):
                with self.subTest(mode=mode, head=head):
                    predictor = self.predictor(mode, evidential_head=head)
                    output_head = predictor.delta_head if head == "separate" else predictor.joint_head
                    torch.nn.init.normal_(output_head[-1].weight, std=.01)
                    history = torch.randn(2, 3, 4, device="cuda") * 10
                    target = torch.randn(2, 2, device="cuda")
                    loss = predictor(history, target, (100, 200))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    gradient = predictor.blocks[0].mamba.in_proj.weight.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)

    def test_predictor_sequence_matches_stateful_with_covariance_and_kappa(self):
        tokens = np.array([
            [10, 20, 1, 2, .2, 0, 0, 0, 1, 0, 0],
            [11, 22, 1, 2, 0, 4, 9, 1, 0, 1, 3],
            [12, 24, 1, 2, .7, 0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)
        for mode, weighting in product(
            ("b", "delta"), ("linear", "rational", "cosine"),
        ):
            for implicit in ("none", "unified", "separate", "separate_full"):
                with self.subTest(
                    mode=mode, implicit=implicit, weighting=weighting,
                ), torch.no_grad():
                    predictor = self.predictor(
                        mode, history_mode="stateful", uncertainty_mode=implicit,
                        uncertainty_weight=weighting,
                    )
                    torch.nn.init.normal_(predictor.delta_head[-1].weight, std=.01)
                    torch.nn.init.normal_(predictor.evidence_head[-1].weight, std=.01)
                    expected = predictor.generate_distribution([tokens], (100, 200), return_kappa=True)
                    states = None
                    for token in tokens:
                        delta, covariance, states, kappa = predictor.stateful_step_distribution(
                            token, states, (100, 200), return_kappa=True,
                        )
                    for a, b in zip((delta, covariance, kappa), expected):
                        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-4)

    def test_tracker_routes_current_covariance_through_misses_and_reacquisition(self):
        for mode in ("b", "delta"):
            for history_mode in ("window", "stateful"):
                with self.subTest(mode=mode, history_mode=history_mode):
                    predictor = self.predictor(
                        mode, history_mode=history_mode, uncertainty_weight="linear",
                    )
                    tracker = Tracker(
                        motion_model="mamba", bev_grid_shape=(100, 200), device="cuda",
                        motion_predictor=predictor, motion_history_len=3,
                        mamba_history_mode=history_mode, mamba_uncertainty_gate=mode,
                    )
                    track = tracker.update([[10, 20]], scores=[.9], uncertainties=[.25])[0]
                    track_id = track.track_id
                    self.assertEqual(len(track.conds[-1]), 11)
                    for index in range(2):
                        cached_kappa = track.mamba_next_kappa
                        tracker.update([], scores=[], uncertainties=[])
                        token = track.conds[-1]
                        np.testing.assert_allclose(token[8:10], [0, 1])
                        covariance = track.motion_covariance
                        np.testing.assert_allclose(token[5:8], [covariance[0, 0], covariance[1, 1], covariance[0, 1]])
                        self.assertAlmostEqual(token[10], track.motion_kappa, places=6)
                        if history_mode == "stateful":
                            self.assertAlmostEqual(token[10], cached_kappa, places=6)
                        if index == 1:
                            self.assertGreater(np.trace(track.motion_position_covariance), np.trace(covariance))
                    tracks = tracker.update([track.xy], scores=[.9], uncertainties=[.6])
                    self.assertEqual(tracks[0].track_id, track_id)
                    np.testing.assert_allclose(track.conds[-1][4:], [.6, 0, 0, 0, 1, 0, 0])
                    self.assertIsNone(track.motion_position_covariance)


if __name__ == "__main__":
    unittest.main()
