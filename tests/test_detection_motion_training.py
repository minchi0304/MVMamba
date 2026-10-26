import contextlib
import io
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import main
from dataloader.dataloader import GetDataset
from tracker.mamba_motion_predictor import MambaMotionPredictor
from tracker.bimamba_motion_predictor import BiMambaMotionPredictor
from tracker.multitracker import STrack
from utils.detection_motion_training import (
    DetectionHistory, build_detection_motion_batch, history_inference,
)


def arguments(**overrides):
    values = dict(
        motion_model='mamba', motion_train_history='detection', motion_history_len=3,
        high_cls_thres=.4, motion_train_match_thresh=5., track_buffer=2,
        mamba_uncertainty_mode='unified', mamba_uncertainty_gate='none',
        dataset='multiviewx', multi_task_loss='plain',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class RecordingMotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(10, 2, bias=False)
        self.rollouts = []
        self.last_loss_stats = {}

    def generate_distribution(self, histories, bev_grid_shape, return_kappa):
        self.rollouts.append([np.array(history, copy=True) for history in histories])
        count = len(histories)
        delta = np.tile([2., 0.], (count, 1)).astype(np.float32)
        covariance = np.tile(np.array([[4., .5], [.5, 9.]], dtype=np.float32), (count, 1, 1))
        return delta, covariance, np.full((count,), 2., dtype=np.float32)

    def forward(self, history, target, bev_grid_shape):
        self.training_input = history.detach().clone()
        return (self.projection(history[:, -1]) - target).square().mean()


class ReplayTests(unittest.TestCase):
    def replay(self, mode='unified', gate='none', max_gap=2):
        return DetectionHistory(4, .4, 5., max_gap, mode, gate)

    def advance(self, replay, motion, xy=(), scores=(), gt_xy=((10., 20.),), ids=(7,)):
        replay.advance(xy, scores, [.25] * len(scores), gt_xy, ids, motion, (100, 200))

    def test_consecutive_misses_use_previous_prediction_covariance_and_target_residual(self):
        replay, motion = self.replay(gate='b'), RecordingMotion()
        self.advance(replay, motion, ((12., 20.),), (.9,))
        self.advance(replay, motion)
        self.advance(replay, motion)
        # Second rollout consumes first prediction, including its covariance.
        np.testing.assert_allclose(motion.rollouts[1][0][-1],
                                   [14., 20., 2., 0., 0., 4., 9., .5, 0., 1., 2.])
        conditions, targets, stats = replay.samples([[19., 20.], [50., 50.]], [7, 99])
        np.testing.assert_allclose(targets, [[3., 0.]])  # GT(19) - predicted last position(16)
        np.testing.assert_allclose(conditions[0][-1, :2], [16., 20.])
        self.assertEqual((stats['motion_samples'], stats['det_tokens'], stats['motion_tokens']), (1, 1, 2))
        self.assertEqual(conditions[0].shape, (4, 11))
        np.testing.assert_allclose(conditions[0][0], conditions[0][1])  # short-history padding

    def test_reacquisition_uses_predicted_previous_position_and_tracker_token_format(self):
        replay, motion = self.replay(), RecordingMotion()
        self.advance(replay, motion, ((12., 20.),), (.9,))
        self.advance(replay, motion)
        self.advance(replay, motion, ((17., 20.),), (.35,), ((17., 20.),))
        last = replay.histories[7][-1]
        track = STrack([17., 20.], .35, detection_uncertainty=.25, mamba_uncertainty_mode='unified')
        np.testing.assert_array_equal(last, track._detection_condition(np.array([17., 20.]), np.array([3., 0.])))
        np.testing.assert_allclose(last, [17., 20., 3., 0., .25, 0., 0., 0., 1., 0.])

    def test_low_confidence_cannot_birth_and_expired_history_restarts_from_detection(self):
        replay, motion = self.replay(max_gap=1), RecordingMotion()
        self.advance(replay, motion, ((10., 20.),), (.35,))
        self.assertFalse(replay.histories)
        self.advance(replay, motion, ((10., 20.),), (.9,))
        self.advance(replay, motion)
        self.advance(replay, motion)
        self.assertFalse(replay.histories)
        self.assertFalse(replay.samples([[10., 20.]], [7])[0])
        self.advance(replay, motion, ((12., 20.),), (.9,))
        self.assertEqual(len(replay.histories[7]), 1)
        np.testing.assert_array_equal(replay.histories[7][-1][2:4], [0., 0.])

    def test_assignment_is_one_to_one_gated_and_never_inserts_gt_positions(self):
        replay, motion = self.replay(), RecordingMotion()
        self.advance(replay, motion, ((11., 20.), (100., 100.)), (.9, .9),
                     ((10., 20.), (13., 20.), (50., 50.)), (7, 8, 9))
        self.assertEqual(len(replay.histories), 1)
        np.testing.assert_array_equal(next(iter(replay.histories.values()))[0][:2], [11., 20.])

    def test_uncertainty_disabled_still_rolls_out_and_counts_sources(self):
        replay, motion = self.replay(mode='none'), RecordingMotion()
        self.advance(replay, motion, ((10., 20.),), (.9,))
        self.advance(replay, motion)
        conditions, _, stats = replay.samples([[14., 20.]], [7])
        self.assertEqual(conditions[0].shape, (4, 4))
        self.assertEqual(stats['motion_tokens'], 1)

    def test_history_stops_at_split_start_and_frame_gap(self):
        dataset = SimpleNamespace(frames=[10, 15, 20, 35, 40], frame_step=5, motion_history_len=2)
        dataset.frame_indices = {frame: index for index, frame in enumerate(dataset.frames)}
        for frame, expected in ((10, []), (20, [10, 15]), (35, []), (40, [35])):
            self.assertEqual(GetDataset.history_frames(dataset, frame), expected)
        with self.assertRaises(KeyError):
            GetDataset.history_frames(dataset, 0)

    def test_history_image_loading_disables_augmentation_without_mutating_dataset(self):
        dataset = SimpleNamespace(frame_indices={5: 0}, root='train', train=True, data_aug=True)
        with patch.object(dataset, '_load_images', create=True,
                          return_value=(torch.zeros(1), torch.eye(3), [])) as load:
            GetDataset.load_history_frame(dataset, 5)
        self.assertFalse(load.call_args.kwargs['augment'])
        np.testing.assert_array_equal(load.call_args.args[1], np.eye(3))
        self.assertTrue(dataset.data_aug)


class TinyFrames:
    root = 'train'
    world_grid_shape = (64, 64)
    grid_reduce = 2

    def __init__(self):
        self.world_gt = {frame: (np.array([[10. + 2 * frame, 20.]]), np.array([7])) for frame in range(5)}
        self.requested = []

    def history_frames(self, frame):
        return list(range(max(0, frame - 3), frame))

    def load_history_frame(self, frame):
        self.requested.append(frame)
        return {'img': torch.full((1, 1, 16, 16), float(frame)), 'proj_mats': torch.eye(3).unsqueeze(0)}


class TinyModel(nn.Module):
    def __init__(self, motion=None):
        super().__init__()
        self.detector_weight = nn.Parameter(torch.tensor(.1))
        self.bn = nn.BatchNorm2d(1)
        self.motion_predictor = RecordingMotion() if motion is None else motion
        self.instance_weight = nn.Parameter(torch.tensor(0.))
        self.motion_weight = nn.Parameter(torch.tensor(0.))
        self.calls = []

    def forward(self, dataset, roots, images, projection):
        frame = int(images[0, 0, 0, 0, 0])
        self.calls.append((frame, self.training, torch.is_grad_enabled()))
        self.bn(images[:, 0])
        logits = self.detector_weight * torch.ones(1, 2, 16, 16)
        logits[:, 1] = logits[:, 1] + 6.
        # Real decoder: exactly one high-confidence detection at reduced xy=(6,10)
        # in frame zero, followed by misses. Gradients still reach the detector.
        if frame == 0:
            logits[:, 0, 10, 6] = logits[:, 0, 10, 6] + 12.
            logits[:, 1, 10, 6] = logits[:, 1, 10, 6] - 6.
        return logits


def target_batch(frame=3):
    return {
        'root': ['train'], 'frame': torch.tensor([frame]),
        'img': torch.full((1, 1, 1, 16, 16), float(frame)),
        'proj_mats': torch.eye(3).reshape(1, 1, 3, 3),
    }, {'center_bev': torch.ones(1, 1, 16, 16), 'valid_bev': torch.ones(1, 1, 16, 16)}


class JointTrainingTests(unittest.TestCase):
    def test_fresh_past_only_replay_restores_modes_and_does_not_update_bn_or_gradients(self):
        frames, model = TinyFrames(), TinyModel()
        dataset = SimpleNamespace(dicts={'train': {'base': frames}})
        model.train()
        model.motion_predictor.eval()  # mixed modes must be restored exactly
        item, _ = target_batch()
        batch, stats = build_detection_motion_batch(model, dataset, item, 'cpu', arguments())
        self.assertEqual(model.calls, [(0, False, False), (1, False, False), (2, False, False)])
        self.assertEqual(model.bn.num_batches_tracked.item(), 0)
        self.assertTrue(model.training)
        self.assertFalse(model.motion_predictor.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertFalse(batch['motion_condition'].requires_grad)
        torch.testing.assert_close(batch['motion_condition'][0, -1, :2], torch.tensor([16., 20.]))
        torch.testing.assert_close(batch['motion_target'], torch.zeros(1, 2))
        self.assertEqual(stats['motion_tokens'], 2)
        before = batch['motion_condition'][0, 0, 4].item()
        with torch.no_grad():
            model.detector_weight.add_(1.)
        batch, _ = build_detection_motion_batch(model, dataset, item, 'cpu', arguments())
        self.assertNotEqual(before, batch['motion_condition'][0, 0, 4].item())
        self.assertEqual(frames.requested, [0, 1, 2, 0, 1, 2])

    def test_each_step_updates_both_tasks_once_and_empty_history_updates_only_detection(self):
        for model_name in ('mamba', 'bimamba'):
            for weighting in ('plain', 'uncertainty'):
                with self.subTest(model=model_name, weighting=weighting):
                    model, frames = TinyModel(), TinyFrames()
                    dataset = SimpleNamespace(dicts={'train': {'base': frames}})
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
                    args = arguments(motion_model=model_name, multi_task_loss=weighting)
                    with patch.multiple(main, args=args, device=torch.device('cpu'), train_dataset=dataset,
                                        criterion=lambda raw, valid, target: raw.square().mean(), create=True), \
                         patch.object(optimizer, 'step', wraps=optimizer.step) as step, \
                         contextlib.redirect_stdout(io.StringIO()):
                        before = model.detector_weight.detach().clone()
                        main.train(model, 1, [target_batch()], optimizer, 1)
                        self.assertEqual(step.call_count, 1)
                        self.assertFalse(torch.equal(before, model.detector_weight))
                        self.assertGreater(model.motion_predictor.projection.weight.grad.abs().sum().item(), 0.)
                        self.assertEqual(model.bn.num_batches_tracked.item(), 1)
                        self.assertEqual(model.calls[-1], (3, True, True))
                        motion_before = model.motion_predictor.projection.weight.detach().clone()
                        weight_before = model.motion_weight.detach().clone()
                        main.train(model, 1, [target_batch(frame=0)], optimizer, 1)
                        torch.testing.assert_close(model.motion_predictor.projection.weight, motion_before)
                        torch.testing.assert_close(model.motion_weight, weight_before)
                        self.assertIsNone(model.motion_predictor.projection.weight.grad)

    def test_nonzero_covariance_inputs_train_real_niw_uncertainty_projections(self):
        for predictor_class in (MambaMotionPredictor, BiMambaMotionPredictor):
            for mode in ('unified', 'separate_full'):
                with self.subTest(model=predictor_class.__name__, mode=mode):
                    torch.manual_seed(4)
                    predictor = predictor_class(history_len=3, d_model=8, num_layers=0,
                                                loss_type='niw', uncertainty_mode=mode)
                    model, frames = TinyModel(predictor), TinyFrames()
                    dataset = SimpleNamespace(dicts={'train': {'base': frames}})
                    args = arguments(mamba_uncertainty_mode=mode)
                    optimizer = torch.optim.SGD(predictor.parameters(), lr=.01)
                    # Output heads start at zero, so the first update learns the
                    # heads; the second must reach the covariance input columns.
                    for _ in range(2):
                        batch, _ = build_detection_motion_batch(model, dataset, target_batch()[0], 'cpu', args)
                        self.assertGreater(batch['motion_condition'][0, -1, 5:7].sum().item(), 0.)
                        optimizer.zero_grad(set_to_none=True)
                        loss = predictor(batch['motion_condition'], batch['motion_target'], batch['motion_bev_grid_shape'])
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        optimizer.step()
                    projection = predictor.uncertainty_input_projection if mode == 'unified' else predictor.motion_input_projection
                    gradient = projection.weight.grad[:, 1:4] if mode == 'unified' else projection.weight.grad[:, 4:7]
                    self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_modes_restore_on_failure_and_future_frames_are_rejected(self):
        model, frames = TinyModel(), TinyFrames()
        model.train()
        with self.assertRaisesRegex(RuntimeError, 'replay failed'):
            with history_inference(model):
                raise RuntimeError('replay failed')
        self.assertTrue(model.training)
        self.assertTrue(torch.is_grad_enabled())
        dataset = SimpleNamespace(dicts={'train': {'base': frames}})
        with patch.object(frames, 'history_frames', return_value=[3]), \
             self.assertRaisesRegex(ValueError, 'strictly before'):
            build_detection_motion_batch(model, dataset, target_batch()[0], 'cpu', arguments())
        self.assertFalse(frames.requested)

    def test_cli_defaults_to_detection_history_and_retains_target_augmentation(self):
        class ReachedDataset(Exception):
            pass

        command = ['main.py', '-d', 'multiviewx', '--aug']
        with contextlib.redirect_stdout(io.StringIO()) as output, \
             patch.object(sys, 'argv', command), patch('os.makedirs'), \
             patch('utils.logger.Logger', return_value=output), \
             patch('dataloader.multiviewx.MultiviewX', return_value=SimpleNamespace()), \
             patch('dataloader.dataloader.GetDataset', side_effect=ReachedDataset) as dataset:
            with self.assertRaises(ReachedDataset):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')
        self.assertFalse(dataset.call_args.kwargs['build_motion_samples'])
        self.assertTrue(dataset.call_args.kwargs['data_aug'])
        self.assertEqual(dataset.call_args.kwargs['motion_history_len'], 10)


if __name__ == '__main__':
    unittest.main()
