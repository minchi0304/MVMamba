import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.motion_training_data import (
    DetectionMotionDataset, match_detections_to_gt, motion_inputs,
    prepare_motion_dataset, rollout_detection_history, _cache_signature,
)


def sequence(root='train', count=8):
    records = []
    for index in range(count):
        gt = np.array([[2 * index, 20]], dtype=np.float32)
        records.append({
            'frame': index * 5, 'gt_xy': gt, 'gt_ids': np.array([7]),
            'xy': gt + np.array([[0.5, -0.25]], dtype=np.float32),
            'scores': np.array([0.8]), 'uncertainty': np.array([0.1 + index * 0.01]),
        })
    return {'root': root, 'dataset_name': 'Wildtrack', 'metric_scale': .025,
            'bev_grid_shape': [480, 1440], 'records': records}


class DetectionHistoryTests(unittest.TestCase):
    def test_observation_positions_uncertainty_and_gt_endpoint(self):
        data = DetectionMotionDataset([sequence()], 2, .4)
        row = data[0]
        history = row['motion_condition']
        torch.testing.assert_close(history[:, :2], torch.tensor([[2.5, 19.75], [4.5, 19.75]]))
        torch.testing.assert_close(history[:, 2:4], torch.tensor([[2., 0.], [2., 0.]]))
        torch.testing.assert_close(history[:, 4], torch.tensor([.11, .12]))
        torch.testing.assert_close(history[:, 8], torch.ones(2))
        torch.testing.assert_close(history[:, [5, 6, 7, 9, 10]], torch.zeros(2, 5))
        torch.testing.assert_close(row['motion_target'], torch.tensor([1.5, .25]))
        # The observation plus target reconstructs the next GT position exactly.
        torch.testing.assert_close(history[-1, :2] + row['motion_target'], torch.tensor([6., 20.]))
        self.assertEqual(data.sample_keys[0], (0, 15, 7))

    def test_missing_detection_breaks_history_but_not_gt_target(self):
        seq = sequence()
        seq['records'][3].update(xy=np.empty((0, 2)), scores=np.array([]), uncertainty=np.array([]))
        data = DetectionMotionDataset([seq], 2, .4)
        self.assertEqual([key[1] for key in data.sample_keys], [15, 35])
        self.assertEqual(data.summary[0]['skipped_missing_history'], 3)

    def test_missing_timestamp_is_not_compressed(self):
        seq = sequence()
        del seq['records'][3]
        data = DetectionMotionDataset([seq], 2, .4)
        self.assertEqual([key[1] for key in data.sample_keys], [35])
        self.assertGreater(data.summary[0]['skipped_time_gap_windows'], 0)

    def test_future_detection_never_enters_input_or_target(self):
        seq = sequence(count=4)
        expected = DetectionMotionDataset([seq], 2, .4)[0]
        seq['records'][-1].update(xy=np.array([[999., 999.]]), uncertainty=np.array([.99]))
        actual = DetectionMotionDataset([seq], 2, .4)[0]
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_sequence_and_gt_id_boundaries(self):
        first = sequence('a', 3)
        second = sequence('b', 3)
        data = DetectionMotionDataset([first, second], 2, .4)
        self.assertEqual(len(data), 0)
        seq = sequence(count=4)
        seq['records'][1]['gt_ids'] = np.array([99])
        self.assertEqual(len(DetectionMotionDataset([seq], 2, .4)), 0)

    def test_metric_gate_confidence_and_unique_matches(self):
        record = {
            'gt_xy': np.array([[0., 0.], [2., 0.], [100., 0.]]),
            'gt_ids': np.array([1, 2, 3]),
            'xy': np.array([[.2, 0.], [99.9, 0.], [2., 0.]]),
            'scores': np.array([.8, .29, .8]), 'uncertainty': np.array([.1, .2, .3]),
        }
        matched = match_detections_to_gt(record, .1, .05, .3)
        self.assertEqual(set(matched), {1, 2})
        np.testing.assert_allclose(matched[1][0], [.2, 0.])
        np.testing.assert_allclose(matched[2][0], [2., 0.])
        self.assertEqual(len(match_detections_to_gt({**record, 'xy': record['xy'][:1],
            'scores': record['scores'][:1], 'uncertainty': record['uncertainty'][:1]}, .1, 1., .3)), 1)


class TinyMotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5, 2, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.last_loss_stats = {}

    def forward(self, history, target, bev_grid_shape):
        return (self.projection(history[:, -1, :5]) - target).square().mean() - 3.0

    def predict_normalized_distribution(self, history, bev_grid_shape):
        count = history.size(0)
        return (history.new_tensor([[2., 0.]]).expand(count, -1),
                history.new_tensor([[[4., 1.], [1., 9.]]]).expand(count, -1, -1),
                history.new_full((count, 1), 2.), None, None)

    def _denormalize_delta(self, mean, shape):
        return mean

    def _denormalize_covariance(self, covariance, shape):
        return covariance


class RolloutTests(unittest.TestCase):
    def test_consecutive_misses_reacquisition_and_masks(self):
        seq = sequence(count=5)
        for index in (1, 2):
            seq['records'][index].update(xy=np.empty((0, 2)), scores=np.array([]), uncertainty=np.array([]))
        data = DetectionMotionDataset([seq], 3, .4, missing='motion')
        motion = TinyMotion().train()
        inputs = data[torch.tensor([0])]
        history, target, _ = motion_inputs(SimpleNamespace(motion_predictor=motion), inputs, 'cpu')
        torch.testing.assert_close(history[0, :, :2], torch.tensor([[2.5, 19.75], [4.5, 19.75], [6.5, 19.75]]))
        torch.testing.assert_close(history[0, :, 8:10], torch.tensor([[0., 1.], [0., 1.], [1., 0.]]))
        torch.testing.assert_close(history[0, :2, 4], torch.zeros(2))
        torch.testing.assert_close(history[0, :2, 5:8], torch.tensor([[4., 9., 1.], [4., 9., 1.]]))
        torch.testing.assert_close(history[0, :2, 10], torch.tensor([2., 2.]))
        torch.testing.assert_close(history[0, 2, 5:8], torch.zeros(3))
        torch.testing.assert_close(history[0, 2, 4], torch.tensor(.13))
        torch.testing.assert_close(target, torch.tensor([[1.5, .25]]))
        self.assertFalse(history.requires_grad)
        self.assertTrue(motion.training)

    def test_target_is_relative_to_last_motion_position_not_last_detection(self):
        seq = sequence(count=4)
        for index in (1, 2):
            seq['records'][index].update(xy=np.empty((0, 2)), scores=np.array([]), uncertainty=np.array([]))
        data = DetectionMotionDataset([seq], 2, .4, missing='motion')
        history, target, _ = motion_inputs(SimpleNamespace(motion_predictor=TinyMotion()),
                                          data[torch.tensor([0])], 'cpu')
        torch.testing.assert_close(history[0, -1, :2] + target[0], torch.tensor([6., 20.]))
        torch.testing.assert_close(target, torch.tensor([[1.5, .25]]))

    def test_first_observation_padding_and_no_gt_bootstrap(self):
        seq = sequence(count=4)
        for index in (0, 1):
            seq['records'][index].update(xy=np.empty((0, 2)), scores=np.array([]), uncertainty=np.array([]))
        data = DetectionMotionDataset([seq], 2, .4, missing='motion')
        row = data[0]
        torch.testing.assert_close(row['motion_observations'][:, :2], torch.tensor([[4.5, 19.75]]).expand(3, -1))
        seq['records'][2].update(xy=np.empty((0, 2)), scores=np.array([]), uncertainty=np.array([]))
        self.assertEqual(len(DetectionMotionDataset([seq], 2, .4, missing='motion')), 0)


class CacheTests(unittest.TestCase):
    def test_cache_creation_uses_real_decode_units_and_beta_uncertainty(self):
        class Frames(Dataset):
            def __init__(self):
                self.root = 'synthetic_training_sequence'
                self.train, self.data_aug = True, True
                self.dataset_name = 'Wildtrack'
                self.world_grid_shape = [64, 64]
                self.grid_reduce, self.img_reduce, self.num_cam = 4, 8, 1
                self.img_shape = [64, 64]
                self.base = SimpleNamespace(grid_cell=2.5)
                self.world_gt = {i * 5: (np.array([[4 * (2 + i) + .5, 12.25]], dtype=np.float32),
                                        np.array([7])) for i in range(4)}
                self.datasets, self.dataset_list = [self], [self.root]
                self.dicts = {self.root: {'proj_mats': torch.eye(3).unsqueeze(0)}}

            def __len__(self):
                return 4

            def __getitem__(self, index):
                if self.data_aug:
                    raise AssertionError('History snapshot must use original coordinates')
                return {'img': torch.tensor([float(index)]), 'root': self.root, 'frame': index * 5}, {}

        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(0.))

            def forward(self, dataset, root, image, projection):
                index = int(image.flatten()[0])
                logits = torch.empty(1, 2, 16, 16)
                logits[:, 0], logits[:, 1] = -5., 5.
                logits[0, :, 3, 2 + index] = torch.tensor([5., -5.])
                return logits + self.bias

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'source.pth'
            torch.save({}, checkpoint)
            args = SimpleNamespace(detector_checkpoint=str(checkpoint), dataset='wildtrack',
                num_workers=0, motion_detection_cache=None, motion_history_len=2,
                high_cls_thres=.4, motion_history_missing='skip')
            dataset = Frames()
            detector = Detector()
            with contextlib.redirect_stdout(io.StringIO()):
                data = prepare_motion_dataset(detector, dataset, args, 'cpu', directory)
            self.assertTrue(dataset.data_aug)
            metadata = json.loads((Path(directory) / 'motion_training_samples.json').read_text())
            self.assertTrue(metadata['checkpoint'])
            self.assertIsNotNone(metadata['checkpoint_sha256'])
            self.assertEqual(len(data), 1)
            history = data[0]['motion_condition']
            torch.testing.assert_close(history[:, :2], torch.tensor([[12., 12.], [16., 12.]]))
            self.assertTrue(torch.all((history[:, 4] > 0) & (history[:, 4] < 1)))
            torch.testing.assert_close(data[0]['motion_target'], torch.tensor([4.5, .25]))
            self.assertTrue((Path(directory) / 'motion_detections_train.pt').exists())
            old_signature = _cache_signature(dataset, args, detector)
            with patch.object(detector, 'forward', side_effect=AssertionError('cache should be reused')), \
                 contextlib.redirect_stdout(io.StringIO()):
                reused = prepare_motion_dataset(detector, dataset, args, 'cpu', directory)
            torch.testing.assert_close(history, reused[0]['motion_condition'])
            with torch.no_grad():
                detector.bias.add_(1.)
            self.assertNotEqual(old_signature, _cache_signature(dataset, args, detector))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                prepare_motion_dataset(detector, dataset, args, 'cpu', directory)
            args.motion_detection_cache = str(Path(directory) / 'new_detector.pt')
            with patch('utils.motion_training_data.utils.decode_frame', side_effect=RuntimeError('decode failed')):
                with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                    prepare_motion_dataset(detector, dataset, args, 'cpu', directory)
            self.assertTrue(dataset.data_aug)

    def test_matching_cache_is_reused_and_mismatched_cache_rejected(self):
        seq = sequence()
        dataset = SimpleNamespace(datasets=[SimpleNamespace(train=True, data_aug=True)])
        model = nn.Module()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'detections.pt'
            torch.save({'signature': {'split': 'train'}, 'sequences': [seq]}, cache)
            args = SimpleNamespace(motion_detection_cache=str(cache), detector_checkpoint='weights.pth',
                motion_history_len=2, high_cls_thres=.4)
            signature = {'split': 'train', 'checkpoint_sha256': 'test', 'detector_sha256': 'test-detector'}
            torch.save({'signature': signature, 'sequences': [seq]}, cache)
            with patch('utils.motion_training_data._cache_signature', return_value=signature), \
                 patch('utils.motion_training_data.utils.decode_frame', side_effect=AssertionError('cache should be reused')), \
                 contextlib.redirect_stdout(io.StringIO()):
                data = prepare_motion_dataset(model, dataset, args, 'cpu', directory)
            self.assertEqual(len(data), 5)
            self.assertTrue(dataset.datasets[0].data_aug)
            self.assertTrue((Path(directory) / 'motion_training_samples.json').exists())
            with patch('utils.motion_training_data._cache_signature', return_value={'split': 'other'}):
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    prepare_motion_dataset(model, dataset, args, 'cpu', directory)

    def test_empty_frozen_detector_cache_stops_motion_training(self):
        seq = sequence()
        for record in seq['records']:
            record['scores'] = np.zeros_like(record['scores'])
        dataset = SimpleNamespace(datasets=[SimpleNamespace(train=True, data_aug=True)])
        signature = {'checkpoint_sha256': None, 'detector_sha256': 'initial-detector'}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'detections.pt'
            torch.save({'signature': signature, 'sequences': [seq]}, cache)
            args = SimpleNamespace(motion_detection_cache=str(cache), detector_checkpoint='weights.pth',
                motion_history_len=2, high_cls_thres=.4)
            with patch('utils.motion_training_data._cache_signature', return_value=signature), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'No motion training samples'):
                    prepare_motion_dataset(nn.Module(), dataset, args, 'cpu', directory)
            self.assertTrue(dataset.datasets[0].data_aug)
            metadata = json.loads((Path(directory) / 'motion_training_samples.json').read_text())
            self.assertEqual(metadata['sequences'][0]['samples'], 0)
            self.assertTrue(metadata['checkpoint'])

    def test_test_split_cannot_become_training_cache(self):
        dataset = SimpleNamespace(datasets=[SimpleNamespace(train=False)])
        with self.assertRaisesRegex(ValueError, 'training splits'):
            prepare_motion_dataset(None, dataset, None, 'cpu', '.')

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual Mamba training needs CUDA')
    def test_real_mamba_learns_from_detection_uncertainty_and_explicit_gates(self):
        from tracker.mamba_motion_predictor import MambaMotionPredictor

        seq = sequence(count=5)
        seq['records'][1].update(xy=np.empty((0, 2)), scores=np.array([]), uncertainty=np.array([]))
        data = DetectionMotionDataset([seq], 2, .4, missing='motion')
        for mode, gate in [('unified', 'none'), ('separate', 'none'), ('separate_full', 'none'),
                           ('none', 'b'), ('none', 'delta')]:
            with self.subTest(mode=mode, gate=gate):
                torch.manual_seed(8)
                model = nn.Module()
                model.detector = nn.Linear(2, 2)
                model.motion_predictor = MambaMotionPredictor(
                    history_len=2, d_model=8, d_state=3, num_layers=1,
                    loss_type='niw', evidential_head='joint', uncertainty_mode=mode,
                    uncertainty_gate=gate, uncertainty_weight='linear',
                )
                model.cuda()
                optimizer = torch.optim.Adam(model.motion_predictor.parameters(), lr=1e-3)
                loader = DataLoader(data, batch_size=len(data))
                for _ in range(2):
                    for item in loader:
                        optimizer.zero_grad()
                        history, target, shape = motion_inputs(model, item, 'cuda')
                        loss = model.motion_predictor(history, target, shape)
                        loss.backward()
                        optimizer.step()
                self.assertTrue(torch.isfinite(loss))
                predictor = model.motion_predictor
                if mode == 'unified':
                    uncertainty_weights = predictor.uncertainty_input_projection.weight[:, 0]
                elif mode in ('separate', 'separate_full'):
                    uncertainty_weights = predictor.detection_input_projection.weight[:, -1]
                    self.assertGreater(predictor.motion_input_projection.weight.grad[:, -3:].abs().sum().item(), 0)
                else:
                    normalized = predictor._normalize_history(history, shape)
                    reliability = predictor._history_reliability(normalized)
                    self.assertTrue(torch.all((reliability > 0) & (reliability < 1)))
                    continue
                self.assertGreater(uncertainty_weights.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
