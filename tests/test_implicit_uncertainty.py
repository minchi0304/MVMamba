import unittest
from itertools import product
from types import SimpleNamespace

import numpy as np
import torch

from tracker.bimamba_motion_predictor import BiMambaMotionPredictor
from tracker.mamba_motion_predictor import MambaMotionPredictor
from tracker.motion_history import detection_token, motion_token
from tracker.multitracker import Tracker
from utils.detection_motion_training import DetectionHistory, build_detection_motion_batch


COMPONENTS = ("total", "aleatoric", "epistemic")
MODES = ("unified", "separate", "separate_full")


class ImplicitUncertaintyTests(unittest.TestCase):
    def predictor(self, component="total", mode="unified", predictor_class=MambaMotionPredictor, **kwargs):
        # No CUDA kernels are needed to exercise token routing, projections,
        # the NIW head/loss, or the shared window/stateful input pipeline.
        return predictor_class(
            history_len=3, d_model=8, num_layers=0, loss_type="niw",
            uncertainty_mode=mode, implicit_motion_uncertainty_type=component, **kwargs,
        )

    def history(self):
        return torch.tensor([[
            [10., 20., 2., -1., .25, 0., 0., 0., 1., 0., 0.],
            [12., 19., 2., -1., 0., 400., 900., -100., 0., 1., 2.],
            [14., 18., 2., -1., 0., 800., 1800., -200., 0., 1., 4.],
        ]])

    def test_all_projections_receive_selected_full_covariance_and_backpropagate(self):
        for mode, component in product(MODES, COMPONENTS):
            with self.subTest(mode=mode, component=component):
                model = self.predictor(component, mode)
                history = model._normalize_history(self.history(), (100, 200)).requires_grad_()
                original = history.detach().clone()
                covariance = history[..., 5:8].detach().clone()
                if component != "total":
                    kappa = history[..., 10:11].detach()
                    covariance *= (kappa if component == "aleatoric" else 1.) / (kappa + 1.)
                if mode == "unified":
                    projection = model.uncertainty_input_projection
                    expected = torch.cat((history[..., 4:5].detach(), covariance, history[..., 8:10].detach()), -1)
                else:
                    projection = model.motion_input_projection
                    expected = covariance if mode == "separate" else torch.cat((history[..., :4].detach(), covariance), -1)
                seen = []
                handle = projection.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].detach().clone()))
                with torch.no_grad():
                    projection.weight.fill_(1.)
                model._project_history(history).sum().backward()
                handle.remove()
                torch.testing.assert_close(seen[0], expected)
                torch.testing.assert_close(history.detach(), original)
                self.assertTrue(torch.isfinite(history.grad).all())
                self.assertTrue((history.grad[:, 1:, 5:8].abs() > 0).all())

    def test_implicit_selection_does_not_change_explicit_gate(self):
        for implicit, explicit in product(COMPONENTS, COMPONENTS):
            with self.subTest(implicit=implicit, explicit=explicit):
                model = self.predictor(implicit, uncertainty_gate="b", motion_uncertainty_type=explicit)
                history = model._normalize_history(self.history(), (100, 200))
                trace = history[..., 5] + history[..., 6]
                kappa = history[..., 10]
                factor = {"total": torch.ones_like(kappa), "aleatoric": kappa / (kappa + 1.),
                          "epistemic": 1. / (kappa + 1.)}[explicit]
                before = model._history_reliability(history)
                model._project_history(history)
                torch.testing.assert_close(model._history_reliability(history), before)
                torch.testing.assert_close(before[:, 1:], (1. - torch.tanh(trace * factor))[:, 1:])

    def test_default_total_and_gt_inputs_keep_existing_behavior(self):
        total = self.predictor()
        history = self.history()[..., :10]
        expected = total.input_projection(history[..., :4]) + total.uncertainty_input_projection(history[..., 4:10])
        torch.testing.assert_close(total._project_history(history), expected)
        gt = self.history()[..., :4]
        baseline = total._encode_history(gt, (100, 200))[0]
        for component in COMPONENTS:
            model = self.predictor(component)
            model.load_state_dict(total.state_dict(), strict=True)
            canonical = model._canonicalize_history(gt)
            self.assertEqual(canonical.shape[-1], 10 if component == "total" else 11)
            self.assertTrue((canonical[..., 5:8] == 0).all())
            torch.testing.assert_close(model._encode_history(gt, (100, 200))[0], baseline)

    def test_component_selection_rejects_missing_kappa_and_deterministic_head(self):
        for component in ("aleatoric", "epistemic"):
            with self.subTest(component=component):
                with self.assertRaisesRegex(ValueError, "kappa"):
                    self.predictor(component)._canonicalize_history(self.history()[..., :10])
                with self.assertRaisesRegex(ValueError, "kappa"):
                    motion_token([1., 2.], [0., 0.], np.eye(2), mode="unified",
                                 implicit_motion_uncertainty_type=component)
                with self.assertRaisesRegex(ValueError, "niw"):
                    MambaMotionPredictor(uncertainty_mode="unified", implicit_motion_uncertainty_type=component)

    def test_training_window_and_stateful_share_component_projection(self):
        for component, predictor_class in product(COMPONENTS, (MambaMotionPredictor, BiMambaMotionPredictor)):
            with self.subTest(component=component, model=predictor_class.__name__):
                model = self.predictor(component, predictor_class=predictor_class)
                history = self.history()
                seen = []
                handle = model.uncertainty_input_projection.register_forward_pre_hook(
                    lambda _, inputs: seen.append(inputs[0].detach().clone()))
                loss = model(history, torch.tensor([[3., -2.]]), bev_grid_shape=(100, 200))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                model.generate_distribution(history.numpy(), bev_grid_shape=(100, 200), return_kappa=True)
                torch.testing.assert_close(seen[1], seen[0])
                if predictor_class is MambaMotionPredictor:
                    model.history_mode = "stateful"
                    model.stateful_step_distribution(history[:, -1].numpy(), bev_grid_shape=(100, 200), return_kappa=True)
                    torch.testing.assert_close(seen[2], seen[0][:, -1])
                handle.remove()

    def test_tracker_misses_and_reacquisition_preserve_total_covariance_and_kappa(self):
        for component, history_mode in product(("aleatoric", "epistemic"), ("window", "stateful")):
            with self.subTest(component=component, history_mode=history_mode):
                model = self.predictor(component, history_mode=history_mode)
                tracker = Tracker(motion_model="mamba", bev_grid_shape=(100, 200),
                                  motion_predictor=model, mamba_history_mode=history_mode,
                                  mamba_uncertainty_mode="unified", implicit_motion_uncertainty_type=component)
                track = tracker.update([[10., 20.]], [.9], [.25])[0]
                self.assertEqual(len(track.conds[-1]), 11)
                for _ in range(2):
                    tracker.update([], [], [])
                    token = track.conds[-1]
                    np.testing.assert_allclose(token[5:8], track.motion_covariance[[0, 1, 0], [0, 1, 1]])
                    self.assertEqual(token[10], track.motion_kappa)
                    self.assertGreater(token[10], 0.)
                restored = tracker.update([track.xy], [.9], [.4])[0]
                self.assertEqual(restored.track_id, track.track_id)
                np.testing.assert_array_equal(restored.conds[-1][5:8], [0., 0., 0.])
                self.assertEqual(restored.conds[-1][10], 0.)

    def test_replay_carries_kappa_without_explicit_gate_and_keeps_total_statistics(self):
        for component in COMPONENTS:
            with self.subTest(component=component):
                model = self.predictor(component).eval()
                replay = DetectionHistory(3, .4, 5., 2, "unified", "none", component)
                replay.advance([[10., 20.]], [.9], [.25], [[10., 20.]], [7], model, (100, 200))
                for _ in range(2):
                    replay.advance([], [], [], [[10., 20.]], [7], model, (100, 200))
                histories, targets, stats = replay.samples([[11., 21.]], [7])
                self.assertEqual(histories[0].shape, (3, 10 if component == "total" else 11))
                np.testing.assert_array_equal(histories[0][0], detection_token(
                    [10., 20.], [0., 0.], .25, "unified", "none", component))
                if component != "total":
                    self.assertTrue((histories[0][1:, 10] > 0).all())
                np.testing.assert_allclose(stats['motion_trace_sum'], histories[0][1:, 5:7].sum())
                loss = model(torch.tensor(np.asarray(histories)), torch.tensor(np.asarray(targets)), (100, 200))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()

    def test_training_batch_width_includes_kappa_with_and_without_samples(self):
        frames = SimpleNamespace(
            world_grid_shape=(100, 200), history_frames=lambda target: range(target),
            world_gt={frame: ([[10., 20.]], [7]) for frame in range(3)},
        )
        dataset = SimpleNamespace(dicts={'train': {'base': frames}})
        cache = {('train', 0): ([[10., 20.]], [.9], [.25]), ('train', 1): ([], [], [])}
        for component, frame in product(COMPONENTS, (0, 2)):
            with self.subTest(component=component, frame=frame):
                model = torch.nn.Module()
                model.motion_predictor = self.predictor(component)
                args = SimpleNamespace(
                    motion_history_len=3, high_cls_thres=.4, motion_train_match_thresh=5., track_buffer=2,
                    mamba_uncertainty_mode='unified', mamba_uncertainty_gate='none',
                    implicit_motion_uncertainty_type=component,
                )
                batch, _ = build_detection_motion_batch(
                    model, dataset, {'root': ['train'], 'frame': [frame]}, 'cpu', args, cache,
                )
                width = 10 if component == 'total' else 11
                self.assertEqual(tuple(batch['motion_condition'].shape), (int(frame > 0), 3, width))
                if frame > 0 and component != 'total':
                    self.assertGreater(batch['motion_condition'][0, -1, 10].item(), 0.)


if __name__ == "__main__":
    unittest.main()
