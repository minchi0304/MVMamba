import contextlib
import io
import runpy
import sys
import unittest
from pathlib import Path
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import main
from dataloader.dataloader import GetDataset


class TinyMotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 2, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.last_loss_stats = {}

    def forward(self, history, target, bev_grid_shape):
        return (self.projection(history[:, -1]) - target).square().mean() - 3.0


def gt_batch(empty=False):
    dataset = SimpleNamespace(motion_history_len=2, world_gt={
        frame: (np.array([[frame / 5 * 2, 20]], dtype=np.float32), np.array([7]))
        for frame in (15, 0, 10, 5)
    })
    history, target = GetDataset._build_motion_samples(dataset)[15][0]
    item = {
        'img': torch.ones(1, 1), 'root': ['train'], 'frame': torch.tensor([15]),
        'motion_condition': torch.from_numpy(history).unsqueeze(0),
        'motion_target': torch.from_numpy(target).unsqueeze(0),
        'motion_bev_grid_shape': torch.tensor([[64., 64.]]),
    }
    if empty:
        for name in ('motion_condition', 'motion_target', 'motion_bev_grid_shape'):
            item[name] = item[name][:0]
    return item, {'center_bev': torch.ones(1, 1, 1, 1), 'valid_bev': torch.ones(1, 1, 1, 1)}


class GTJointTrainingTests(unittest.TestCase):
    def test_both_tasks_update_uncertainty_weights_and_plain_has_unit_coefficients(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.detector = nn.Linear(1, 2, bias=False)
                nn.init.zeros_(self.detector.weight)
                self.motion_predictor = TinyMotion()
                self.instance_weight = nn.Parameter(torch.tensor(0.))
                self.motion_weight = nn.Parameter(torch.tensor(0.))

            def forward(self, dataset, roots, images, projection):
                return self.detector(images).reshape(-1, 2, 1, 1)

        def occupancy_loss(raw, valid, target):
            return (raw - 1.).square().mean() * .5

        item, targets = gt_batch()
        torch.testing.assert_close(item['motion_condition'],
                                   torch.tensor([[[2., 20., 2., 0.], [4., 20., 2., 0.]]]))
        torch.testing.assert_close(item['motion_condition'][:, -1, :2] + item['motion_target'],
                                   torch.tensor([[6., 20.]]))
        for motion_model, weighting in product(('mamba', 'bimamba'), ('uncertainty', 'plain')):
            with self.subTest(motion_model=motion_model, weighting=weighting):
                model = Model()
                before = model.detector.weight.detach().clone()
                args = SimpleNamespace(motion_model=motion_model, multi_task_loss=weighting, motion_train_history='gt')
                optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                with patch.multiple(main, args=args, device=torch.device('cpu'), train_dataset=None,
                                    criterion=occupancy_loss, create=True), contextlib.redirect_stdout(io.StringIO()):
                    loss = main.train(model, 1, [(item, targets)], optimizer, 1)
                self.assertAlmostEqual(loss, -.5, places=5)
                self.assertFalse(torch.equal(model.detector.weight, before))
                self.assertGreater(model.motion_predictor.projection.weight.abs().sum().item(), 0)
                if weighting == 'uncertainty':
                    self.assertLess(model.instance_weight.item(), 0)
                    self.assertLess(model.motion_weight.item(), 0)
                else:
                    self.assertIsNone(model.instance_weight.grad)
                    self.assertIsNone(model.motion_weight.grad)

    def test_empty_gt_history_batch_does_not_update_motion_or_abort_training(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.occupancy = nn.Parameter(torch.zeros(1, 2, 1, 1))
                self.motion_predictor = TinyMotion()
                self.instance_weight = nn.Parameter(torch.tensor(0.))
                self.motion_weight = nn.Parameter(torch.tensor(0.))

            def forward(self, dataset, roots, images, projection):
                return self.occupancy

        args = SimpleNamespace(motion_model='mamba', multi_task_loss='uncertainty', motion_train_history='gt')
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
        criterion = lambda raw, valid, target: .5 * (raw - 1.).square().mean()
        with patch.multiple(main, args=args, device=torch.device('cpu'), train_dataset=None,
                            criterion=criterion, create=True), contextlib.redirect_stdout(io.StringIO()):
            main.train(model, 1, [gt_batch(empty=True)], optimizer, 1)
            self.assertGreater(model.occupancy.abs().sum().item(), 0)
            self.assertEqual(model.motion_predictor.projection.weight.abs().sum().item(), 0)
            self.assertEqual(model.motion_weight.item(), 0)
            main.train(model, 1, [gt_batch()], optimizer, 1)
            self.assertGreater(model.motion_predictor.projection.weight.abs().sum().item(), 0)
            motion_before = model.motion_predictor.projection.weight.detach().clone()
            weight_before = model.motion_weight.detach().clone()
            # Momentum/weight decay must not advance a task absent from the next batch.
            main.train(model, 1, [gt_batch(empty=True)], optimizer, 1)
            torch.testing.assert_close(model.motion_predictor.projection.weight, motion_before)
            torch.testing.assert_close(model.motion_weight, weight_before)

    def test_gt_history_cli_keeps_aug_and_builds_motion_windows(self):
        class ReachedTrainingDataset(Exception):
            pass

        command = ('main.py -d multiviewx -pr --data_root /dataset --motion_model mamba '
                   '--multi_task_loss uncertainty --epochs 100 --motion_train_history gt '
                   '--track_match_thresh 50 --track_low_match_thresh 80 --bev_gaussian_sigma 1.5 '
                   '--motion_loss niw --evidential_head separate --mamba_history_mode window '
                   '--mamba_uncertainty_mode separate_full --mamba_uncertainty_gate none --aug')
        with contextlib.redirect_stdout(io.StringIO()) as output, \
             patch.object(sys, 'argv', command.split()), patch('os.makedirs'), \
             patch('utils.logger.Logger', return_value=output), \
             patch('dataloader.multiviewx.MultiviewX', return_value=SimpleNamespace()), \
             patch('dataloader.dataloader.GetDataset', side_effect=ReachedTrainingDataset) as get_dataset:
            with self.assertRaises(ReachedTrainingDataset):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')
        self.assertTrue(get_dataset.call_args.kwargs['data_aug'])
        self.assertTrue(get_dataset.call_args.kwargs['build_motion_samples'])


if __name__ == '__main__':
    unittest.main()
