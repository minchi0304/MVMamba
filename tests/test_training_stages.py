import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

import main
from train_detector import train_detector
from train_motion import train_motion
from utils.motion_training_data import DetectionMotionDataset
from utils.training_stages import freeze_detector, load_detector_weights
from test_detection_motion_training import TinyMotion, sequence


class TinyModel(nn.Module):
    def __init__(self, *unused, multi_task_loss='plain', use_motion_loss=False, **kwargs):
        super().__init__()
        self.detector = nn.Linear(1, 2, bias=False)
        self.norm = nn.BatchNorm1d(2)
        nn.init.zeros_(self.detector.weight)
        if multi_task_loss == 'uncertainty':
            self.instance_weight = nn.Parameter(torch.tensor(0.))
            if use_motion_loss:
                self.motion_weight = nn.Parameter(torch.tensor(0.))

    def forward(self, dataset, roots, images, projection):
        return self.norm(self.detector(images)).reshape(-1, 2, 1, 1)


def occupancy_loss(raw, valid, target):
    return .5 * (raw - 1.).square().mean()


def image_batch():
    item = {'img': torch.ones(2, 1), 'root': ['train', 'train'], 'frame': torch.tensor([15, 20])}
    target = {'center_bev': torch.ones(2, 1, 1, 1), 'valid_bev': torch.ones(2, 1, 1, 1)}
    return item, target


class StageTests(unittest.TestCase):
    def test_cli_stages_are_separate_and_old_joint_detection_flag_is_removed(self):
        detector = main.parse_args('detector', ['-d', 'multiviewx', '-pr', '--aug'])
        self.assertTrue(detector.aug)
        self.assertTrue(detector.pretrained)
        self.assertEqual(detector.multi_task_loss, 'plain')
        motion = main.parse_args('motion', ['--detector_checkpoint', 'weights.pth'])
        self.assertEqual(motion.batch_size, 256)
        self.assertEqual(motion.motion_history_missing, 'motion')
        self.assertEqual(motion.multi_task_loss, 'plain')
        self.assertFalse(motion.aug)
        self.assertEqual(main.parse_args('joint', []).multi_task_loss, 'uncertainty')
        for stage, argv in [('joint', ['--motion_train_history', 'detection']),
                            ('motion', []),
                            ('motion', ['--detector_checkpoint', 'x', '--aug']),
                            ('motion', ['--detector_checkpoint', 'x', '--multi_task_loss', 'uncertainty']),
                            ('detector', ['--motion_loss', 'niw']),
                            ('motion', ['--detector_checkpoint', 'x', '-d', 'gmvd_test'])]:
            with self.subTest(stage=stage, argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main.parse_args(stage, argv)

    def test_detector_stage_optimizes_occupancy_without_motion(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with contextlib.redirect_stdout(io.StringIO()):
            loss = train_detector(model, 1, [image_batch()], optimizer, 1,
                                  dataset=None, device='cpu', criterion=occupancy_loss, args=None)
        self.assertAlmostEqual(loss, .5)
        self.assertGreater(model.norm.bias.abs().sum().item(), 0)
        self.assertFalse(hasattr(model, 'motion_predictor'))
        self.assertFalse(hasattr(model, 'instance_weight'))

    def test_gt_joint_stage_still_trains_both_tasks_with_selected_weighting(self):
        item, target = image_batch()
        item.update(motion_condition=torch.tensor([[[2., 1., .5, .2], [3., 1., 1., 0.]]]),
                    motion_target=torch.tensor([[1., 0.]]), motion_bev_grid_shape=torch.tensor([[64., 64.]]))
        # TinyMotion accepts five features, whereas real GT histories have four.
        item['motion_condition'] = torch.cat((item['motion_condition'], torch.zeros(1, 2, 1)), dim=-1)
        for weighting in ('plain', 'uncertainty'):
            model = TinyModel(multi_task_loss=weighting, use_motion_loss=True)
            model.motion_predictor = TinyMotion()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            args = SimpleNamespace(motion_model='mamba', multi_task_loss=weighting)
            with contextlib.redirect_stdout(io.StringIO()):
                loss = main.train(model, 1, [(item, target)], optimizer, 1,
                                  dataset=None, device='cpu', criterion=occupancy_loss, args=args)
            self.assertAlmostEqual(loss, -2.)
            self.assertGreater(model.norm.bias.abs().sum().item(), 0)
            self.assertGreater(model.motion_predictor.projection.weight.abs().sum().item(), 0)
            if weighting == 'uncertainty':
                self.assertNotEqual(model.motion_weight.item(), 0)
                self.assertNotEqual(model.instance_weight.item(), 0)

    def test_motion_stage_preserves_detector_parameters_and_buffers_for_multiple_epochs(self):
        model = TinyModel()
        model.motion_predictor = TinyMotion()
        freeze_detector(model)
        original = {name: value.clone() for name, value in model.state_dict().items()
                    if not name.startswith('motion_predictor.')}
        seq = sequence(count=5)
        seq['records'][1]['scores'][:] = 0
        data = DetectionMotionDataset([seq], 2, .4, missing='motion')
        loader = DataLoader(data, batch_size=2)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        with patch.object(model, 'forward', side_effect=AssertionError('Motion training must not run detector')), \
             contextlib.redirect_stdout(io.StringIO()):
            for epoch in (1, 2):
                model.train()  # The stage must restore eval mode even after an external train() call.
                train_motion(model, epoch, loader, optimizer, 1,
                             dataset=None, device='cpu', criterion=None, args=None)
        for name, expected in original.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
        self.assertFalse(model.norm.training)
        self.assertTrue(model.motion_predictor.training)
        self.assertGreater(model.motion_predictor.projection.weight.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.detector.parameters()))

    def test_detector_checkpoint_loading_keeps_fresh_motion_parameters(self):
        model = TinyModel(multi_task_loss='uncertainty', use_motion_loss=True)
        model.motion_predictor = TinyMotion()
        with torch.no_grad():
            model.detector.weight.fill_(.5)
            model.motion_predictor.projection.weight.fill_(.7)
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / 'combined.pth'
            torch.save(model.state_dict(), full)
            other = TinyModel()
            other.motion_predictor = TinyMotion()
            load_detector_weights(other, full)
            torch.testing.assert_close(other.detector.weight, model.detector.weight)
            self.assertEqual(other.motion_predictor.projection.weight.abs().sum().item(), 0)
            broken = Path(directory) / 'broken.pth'
            torch.save({'detector.weight': model.detector.weight}, broken)
            with self.assertRaisesRegex(ValueError, 'Detector checkpoint mismatch'):
                load_detector_weights(other, broken)

    def test_motion_run_prepares_cache_once_and_saves_loadable_checkpoints(self):
        data = DetectionMotionDataset([sequence()], 2, .4, missing='motion')
        frames = SimpleNamespace(datasets=[SimpleNamespace(root='train', dataset_name='MultiviewX')])

        class Frames:
            datasets = frames.datasets

            def __len__(self):
                return 4

        def evaluate(model, epoch, *unused, **kwargs):
            model.eval()
            main.score.append([60., 70., 80., 90., 50. + epoch, 40., .1, 90., 20., 10.])

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            detector_path = Path(directory) / 'detector.pth'
            torch.save(TinyModel().state_dict(), detector_path)
            run_dir = Path(directory) / 'motion'
            with patch.object(main, 'Logger', return_value=output), \
                 patch.object(main, 'MultiviewX', return_value=SimpleNamespace()), \
                 patch.object(main, 'GetDataset', return_value=Frames()), \
                 patch.object(main, 'ConcatDataset', side_effect=lambda *datasets: datasets[0]), \
                 patch.object(main, 'resnet18', return_value=None), \
                 patch.object(main, 'MultiView_Detection', side_effect=TinyModel), \
                 patch.object(main, 'MambaMotionPredictor', side_effect=lambda **kwargs: TinyMotion()), \
                 patch('utils.motion_training_data.prepare_motion_dataset', return_value=data) as prepare, \
                 patch.object(main, 'test', side_effect=evaluate):
                main.run('motion', train_motion, ['-d', 'multiviewx', '--detector_checkpoint', str(detector_path),
                         '--output_dir', str(run_dir), '--epochs', '2', '-j', '0', '-b', '2'])
            prepare.assert_called_once()
            with (run_dir / 'metrics.csv').open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[-1]['idf1']), 52.)
            self.assertTrue((run_dir / 'history_preparation.json').exists())
            full = run_dir / 'Best_Multiview_Tracking_multiviewx.pth'
            motion = run_dir / 'Best_Motion_multiviewx.pth'
            restored = TinyModel()
            restored.motion_predictor = TinyMotion()
            restored.load_state_dict(torch.load(full, map_location='cpu'), strict=True)
            initial = torch.load(detector_path, map_location='cpu')
            for name, value in initial.items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            motion_state = torch.load(motion, map_location='cpu')
            self.assertEqual(motion_state.keys(), restored.motion_predictor.state_dict().keys())
            for name, value in restored.motion_predictor.state_dict().items():
                torch.testing.assert_close(motion_state[name], value, rtol=0, atol=0)

    def test_detector_evaluation_uses_detection_metrics_without_tracker(self):
        with tempfile.TemporaryDirectory() as directory:
            gt_path = Path(directory) / 'gt.txt'
            np.savetxt(gt_path, [[0, 12, 8], [10, 12, 8], [11, 12, 8]])
            base = SimpleNamespace(gt_fname=str(gt_path), indexing='yx')
            sub = SimpleNamespace(base=base, grid_reduce=4, world_gt={10: None, 11: None})

            class Loader(list):
                dataset = SimpleNamespace(dataset_list=['scene'], dicts={'scene': {'base': sub}})

            loader = Loader([({'root': ['scene'], 'frame': torch.tensor([frame])}, {}) for frame in (10, 11)])
            decoded = [{'eval_xy': torch.tensor([[2., 3.]]), 'eval_scores': torch.tensor([[.9]])},
                       {'eval_xy': torch.empty(0, 2), 'eval_scores': torch.empty(0, 1)}]
            score = []
            with patch.multiple(main, args=SimpleNamespace(), device='cpu', logdir=directory, score=score, create=True), \
                 patch.object(main.utils, 'decode_frame', side_effect=decoded), \
                 patch.object(main.utils, 'update_tracking_for_frame', side_effect=AssertionError('Detector stage has no tracker')), \
                 contextlib.redirect_stdout(io.StringIO()):
                main.test(nn.Module(), 1, loader, str(Path(directory) / 'detections.txt'), detection_only=True)
            self.assertEqual(len(score), 1)
            self.assertEqual(score[0][0], 50.)
            self.assertEqual(score[0][2], 100.)
            np.testing.assert_allclose(np.loadtxt(Path(directory) / 'detections.txt')[:3], [10, 12, 8])

    def test_detector_run_saves_detector_only_checkpoint_and_keeps_augmentation(self):
        class Frames:
            datasets = [SimpleNamespace(root='train', dataset_name='MultiviewX')]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                item = {'img': torch.ones(1), 'root': 'train', 'frame': index}
                target = {'center_bev': torch.ones(1, 1, 1), 'valid_bev': torch.ones(1, 1, 1)}
                return item, target

        def evaluate(model, epoch, *unused, **kwargs):
            self.assertTrue(kwargs['detection_only'])
            self.assertFalse(hasattr(model, 'motion_predictor'))
            main.score.append([60. + epoch, 70., 80., 90.])

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            with patch.object(main, 'Logger', return_value=output), \
                 patch.object(main, 'MultiviewX', return_value=SimpleNamespace()), \
                 patch.object(main, 'GetDataset', return_value=Frames()) as get_dataset, \
                 patch.object(main, 'ConcatDataset', side_effect=lambda *datasets: datasets[0]), \
                 patch.object(main, 'resnet18', return_value=None), \
                 patch.object(main, 'MultiView_Detection', side_effect=TinyModel), \
                 patch.object(main, 'MambaMotionPredictor', side_effect=AssertionError('Detector-only training must not create Mamba')), \
                 patch.object(main, 'test', side_effect=evaluate):
                main.run('detector', train_detector, ['-d', 'multiviewx', '--output_dir', directory,
                         '--epochs', '2', '-j', '0', '-b', '2', '--aug'])
            self.assertTrue(get_dataset.call_args_list[0].kwargs['data_aug'])
            self.assertFalse(get_dataset.call_args_list[0].kwargs['build_motion_samples'])
            state = torch.load(Path(directory) / 'Best_Multiview_Detection_multiviewx.pth', map_location='cpu')
            TinyModel().load_state_dict(state, strict=True)
            self.assertFalse(any(name.startswith('motion_predictor.') or name.endswith('_weight') for name in state))
            with (Path(directory) / 'metrics.csv').open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertNotIn('idf1', rows[0])


if __name__ == '__main__':
    unittest.main()
