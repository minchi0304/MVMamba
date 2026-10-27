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
    DetectionHistory, build_detection_motion_batch, history_inference, precompute_detection_cache,
)


def arguments(**overrides):
    values = dict(
        motion_model='mamba', motion_train_history='detection', motion_train_cache='epoch', motion_history_len=3,
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

    def __init__(self, history_len=3, num_frames=5):
        self.history_len = history_len
        self.world_gt = {frame: (np.array([[10. + 2 * frame, 20.]]), np.array([7])) for frame in range(num_frames)}
        self.requested = []

    def history_frames(self, frame):
        return list(range(max(0, frame - self.history_len), frame))

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
    def test_precompute_uses_fixed_detector_only_and_refreshes_next_epoch(self):
        frames, model = TinyFrames(), TinyModel()
        dataset = SimpleNamespace(dicts={'train': {'base': frames}})
        args = arguments(motion_train_cache='precompute')
        model.train()
        model.motion_predictor.eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with patch.object(model.motion_predictor, 'generate_distribution',
                          side_effect=AssertionError('Motion must not run during detector precomputation')):
            cache = precompute_detection_cache(model, dataset, 'cpu', args, epoch=1)
        self.assertEqual(frames.requested, list(range(5)))
        self.assertEqual(model.calls, [(frame, False, False) for frame in range(5)])
        self.assertTrue(model.training)
        self.assertFalse(model.motion_predictor.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(set(cache), {('train', frame) for frame in range(5)})
        for values in cache.values():
            for array in values:
                self.assertIsInstance(array, np.ndarray)
                self.assertTrue(array.flags.owndata)

        old_uncertainty = cache[('train', 0)][2].copy()
        with torch.no_grad():
            model.detector_weight.add_(1.)
        result, stats = build_detection_motion_batch(model, dataset, target_batch()[0], 'cpu', args, cache)
        # Current and future frames are cached too, but only t-3..t-1 enter the input.
        self.assertEqual(len(model.calls), 5)
        self.assertEqual((stats['history_inferences'], stats['history_cache_hits']), (0, 3))
        torch.testing.assert_close(result['motion_condition'][0, -1, :2], torch.tensor([16., 20.]))
        torch.testing.assert_close(result['motion_target'], torch.zeros(1, 2))
        self.assertEqual(len(model.motion_predictor.rollouts), 2)
        refreshed = precompute_detection_cache(model, dataset, 'cpu', args, epoch=2)
        self.assertIsNot(refreshed, cache)
        self.assertFalse(np.array_equal(old_uncertainty, refreshed[('train', 0)][2]))
        np.testing.assert_array_equal(old_uncertainty, cache[('train', 0)][2])
        self.assertEqual(frames.requested, list(range(5)) * 2)

    def test_missing_precomputed_frame_fails_without_mixing_in_new_detector_weights(self):
        frames, model = TinyFrames(), TinyModel()
        dataset = SimpleNamespace(dicts={'train': {'base': frames}})
        args = arguments(motion_train_cache='precompute')
        cache = precompute_detection_cache(model, dataset, 'cpu', args)
        del cache[('train', 1)]
        with patch.object(frames, 'load_history_frame', side_effect=AssertionError('Unexpected inference')), \
             self.assertRaisesRegex(KeyError, 'Missing precomputed detection'):
            build_detection_motion_batch(model, dataset, target_batch()[0], 'cpu', args, cache)
        self.assertTrue(model.training)

    def test_cache_reuses_cpu_observations_but_recomputes_motion_with_current_model(self):
        frames, model, cache = TinyFrames(), TinyModel(), {}
        dataset = SimpleNamespace(dicts={'train': {'base': frames}})
        args = arguments(mamba_uncertainty_gate='b')
        first, stats = build_detection_motion_batch(model, dataset, target_batch()[0], 'cpu', args, cache)
        self.assertEqual((stats['history_inferences'], stats['history_cache_hits']), (3, 0))
        for values in cache.values():
            self.assertEqual(len(values), 3)
            for array in values:
                self.assertIsInstance(array, np.ndarray)
                self.assertEqual(array.dtype, np.float32)
                self.assertTrue(array.flags.owndata)
        with torch.no_grad():
            model.detector_weight.add_(1.)
        original = model.motion_predictor.generate_distribution

        def updated_motion(*args, **kwargs):
            delta, covariance, kappa = original(*args, **kwargs)
            return delta * 2, covariance * 3, kappa * 2

        with patch.object(model.motion_predictor, 'generate_distribution', side_effect=updated_motion) as rollout:
            second, stats = build_detection_motion_batch(model, dataset, target_batch()[0], 'cpu', args, cache)
        self.assertEqual((stats['history_inferences'], stats['history_cache_hits']), (0, 3))
        self.assertEqual(frames.requested, [0, 1, 2])
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(rollout.call_count, 2)
        torch.testing.assert_close(first['motion_condition'][0, 0], second['motion_condition'][0, 0])
        torch.testing.assert_close(second['motion_condition'][0, -1, :2], torch.tensor([20., 20.]))
        torch.testing.assert_close(second['motion_condition'][0, -1, 5:8], torch.tensor([12., 27., 1.5]))
        self.assertEqual(second['motion_condition'][0, -1, 10].item(), 4.)

    def test_shuffled_overlapping_windows_match_fresh_outputs_and_decode_each_frame_once(self):
        fresh_frames, cached_frames = (TinyFrames(history_len=10, num_frames=25) for _ in range(2))
        fresh_model, cached_model = TinyModel(), TinyModel()
        fresh_dataset = SimpleNamespace(dicts={'train': {'base': fresh_frames}})
        cached_dataset = SimpleNamespace(dicts={'train': {'base': cached_frames}})
        cache, inference_count, cache_hits = {}, 0, 0
        args = arguments(motion_history_len=10)
        # Reverse order also checks that cached later frames never leak into a target's history.
        for frame in reversed(range(25)):
            item, _ = target_batch(frame)
            expected, _ = build_detection_motion_batch(fresh_model, fresh_dataset, item, 'cpu', args)
            actual, stats = build_detection_motion_batch(cached_model, cached_dataset, item, 'cpu', args, cache)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name])
            inference_count += stats['history_inferences']
            cache_hits += stats['history_cache_hits']
        self.assertEqual(len(fresh_model.calls), 195)
        self.assertEqual(len(cached_model.calls), 24)
        self.assertEqual((inference_count, cache_hits), (24, 171))
        self.assertEqual(sorted(cached_frames.requested), list(range(24)))
        self.assertEqual(len(cache), 24)

    def test_cache_separates_roots_with_same_frame_ids_and_keeps_world_coordinates(self):
        model, cache = TinyModel(), {}
        frames, other = TinyFrames(), TinyFrames()
        other.grid_reduce = 3
        other.world_gt = {frame: (xy * 1.5, ids) for frame, (xy, ids) in other.world_gt.items()}
        dataset = SimpleNamespace(dicts={'train': {'base': frames}, 'other': {'base': other}})
        item, _ = target_batch()
        build_detection_motion_batch(model, dataset, item, 'cpu', arguments(), cache)
        item['root'] = ['other']
        result, stats = build_detection_motion_batch(model, dataset, item, 'cpu', arguments(), cache)
        self.assertEqual((len(cache), stats['history_inferences']), (6, 3))
        self.assertEqual(other.requested, [0, 1, 2])
        torch.testing.assert_close(result['motion_condition'][0, 0, :2], torch.tensor([18., 30.]))

    def test_train_resets_cache_each_epoch_and_keeps_one_joint_update_per_batch(self):
        for mode, inference_frames in (
            ('precompute', [0, 1, 2, 3, 4]), ('epoch', [0, 1, 2]), ('none', [0, 1, 2, 0, 1]),
        ):
            with self.subTest(cache=mode):
                model, frames = TinyModel(), TinyFrames()
                dataset = SimpleNamespace(dicts={'train': {'base': frames}})
                optimizer = torch.optim.SGD(model.parameters(), lr=.0001)
                args = arguments(motion_train_cache=mode)
                with patch.multiple(main, args=args, device=torch.device('cpu'), train_dataset=dataset,
                                    criterion=lambda raw, valid, target: raw.square().mean(), create=True), \
                     patch.object(optimizer, 'step', wraps=optimizer.step) as step, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    for epoch in (1, 2):
                        before = [parameter.detach().clone() for parameter in
                                  (model.detector_weight, model.motion_predictor.projection.weight)]
                        main.train(model, epoch, [target_batch(3), target_batch(2)], optimizer, 1)
                        self.assertFalse(torch.equal(model.detector_weight, before[0]))
                        self.assertFalse(torch.equal(model.motion_predictor.projection.weight, before[1]))
                self.assertEqual(step.call_count, 4)
                self.assertEqual(model.bn.num_batches_tracked.item(), 4)
                self.assertEqual(frames.requested, inference_frames * 2)
                self.assertEqual([frame for frame, training, grad in model.calls if training and grad], [3, 2, 3, 2])
                self.assertIn('cache_hits', output.getvalue())
                if mode == 'precompute':
                    expected_epoch = [(frame, False, False) for frame in range(5)] + [(3, True, True), (2, True, True)]
                    self.assertEqual(model.calls, expected_epoch * 2)
                    self.assertIn('precompute 5, during training 0, total 5', output.getvalue())

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
        self.assertIn('Detector history cache: precompute', output.getvalue())


if __name__ == '__main__':
    unittest.main()
