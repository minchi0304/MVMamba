import contextlib
import io
import runpy
import sys
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import tracker.bimamba_motion_predictor as bimamba
from dataloader.dataloader import motion_joint_collate
from tracker.multitracker import Tracker
from utils.utils import build_tracker


class RecordingScan(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, sequence, reliability=None):
        self.seen_sequence = sequence.detach().clone()
        self.seen_reliability = None if reliability is None else reliability.detach().clone()
        return (sequence if reliability is None else sequence * reliability.unsqueeze(-1)).cumsum(1)


class BiMambaWiringTests(unittest.TestCase):
    def test_bidirectional_scan_aligns_tokens_reliability_and_residual(self):
        with patch.object(bimamba, 'Mamba', RecordingScan), \
             patch.object(bimamba, 'UncertaintyGatedMamba', RecordingScan):
            for gate in ('none', 'b', 'delta'):
                with self.subTest(gate=gate):
                    block = bimamba.ResidualBiMambaBlock(4, 3, 2, 2, 0., gate)
                    block.norm = nn.Identity()
                    sequence = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
                    reliability = None if gate == 'none' else torch.tensor([[1., .2, .7], [.1, .8, .3]])
                    result = block(sequence, reliability)
                    weighted = sequence if reliability is None else sequence * reliability.unsqueeze(-1)
                    expected = sequence + .5 * (weighted.cumsum(1) + weighted.flip(1).cumsum(1).flip(1))
                    torch.testing.assert_close(result, expected)
                    torch.testing.assert_close(block.backward_mamba.seen_sequence, sequence.flip(1))
                    if reliability is not None:
                        torch.testing.assert_close(block.backward_mamba.seen_reliability, reliability.flip(1))
                    changed = sequence.clone()
                    changed[:, -1] += 5.
                    self.assertGreater((block(changed, reliability)[:, 0] - result[:, 0]).abs().sum().item(), 0)

    def test_window_readout_pools_history_but_preserves_last_forward_velocity(self):
        with patch.object(bimamba, 'Mamba', RecordingScan):
            model = bimamba.BiMambaMotionPredictor(history_len=3, d_model=4, num_layers=1)
        model.blocks = nn.ModuleList()
        model.input_projection = nn.Identity()
        model.output_norm = nn.Identity()
        with torch.no_grad():
            model.temporal_embedding.zero_()
        history = torch.tensor([[[10., 20., 2., 3.], [12., 23., 2., 3.]]])
        features, velocity = model._encode_history(history, (100, 200))
        expected = torch.tensor([[[.05, .2, 0., 0.], [.05, .2, .01, .03], [.06, .23, .01, .03]]])
        torch.testing.assert_close(features, expected.mean(1))
        torch.testing.assert_close(velocity, torch.tensor([[.01, .03]]))

    def test_cli_constructs_bimamba_and_gt_loader_with_motion_task(self):
        class ReachedPredictor(Exception):
            pass

        class Frames:
            root = 'scene'
            dataset_name = 'MultiviewX'
            bev_gaussian_sigma = 1.5

            def __init__(self):
                self.datasets = [self]

            def __len__(self):
                return 2

        command = ('main.py -d multiviewx --data_root /dataset --motion_model bimamba '
                   '--motion_train_history gt '
                   '--mamba_history_mode window --mamba_uncertainty_gate b '
                   '--niw_reg_weight 0.25 '
                   '--motion_loss niw --aug -j 0 --mamba_d_model 8 --mamba_layers 2')
        with contextlib.redirect_stdout(io.StringIO()) as output, \
             patch.object(sys, 'argv', command.split()), patch('os.makedirs'), \
             patch('utils.logger.Logger', return_value=output), \
             patch('dataloader.multiviewx.MultiviewX', return_value=SimpleNamespace()), \
             patch('dataloader.dataloader.GetDataset', return_value=Frames()) as dataset, \
             patch('dataloader.concat_dataset.ConcatDataset', side_effect=lambda *items: items[0]), \
             patch('torch.utils.data.DataLoader', wraps=torch.utils.data.DataLoader) as loader, \
             patch('detector.encoder.resnet18', return_value=None), \
             patch('detector.multiview_model.MultiView_Detection', return_value=nn.Module()) as detector, \
             patch.object(bimamba, 'BiMambaMotionPredictor', side_effect=ReachedPredictor) as predictor:
            with self.assertRaises(ReachedPredictor):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')
        self.assertTrue(dataset.call_args_list[0].kwargs['build_motion_samples'])
        self.assertTrue(dataset.call_args_list[0].kwargs['data_aug'])
        self.assertIs(loader.call_args_list[0].kwargs['collate_fn'], motion_joint_collate)
        self.assertTrue(detector.call_args.kwargs['use_motion_loss'])
        self.assertEqual(predictor.call_args.kwargs['d_model'], 8)
        self.assertEqual(predictor.call_args.kwargs['num_layers'], 2)
        self.assertEqual(predictor.call_args.kwargs['uncertainty_gate'], 'b')
        self.assertEqual(predictor.call_args.kwargs['niw_reg_weight'], .25)

    def test_stateful_bimamba_is_rejected_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, 'window'):
            bimamba.BiMambaMotionPredictor(history_mode='stateful')
        with self.assertRaisesRegex(ValueError, 'window'):
            Tracker(motion_model='bimamba', mamba_history_mode='stateful')
        with patch.object(sys, 'argv', ['main.py', '--motion_model', 'bimamba', '--mamba_history_mode', 'stateful']), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as exit_error:
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')
        self.assertEqual(exit_error.exception.code, 2)
        self.assertIn('--mamba_history_mode window', error.getvalue())

    def test_tracker_routes_bimamba_covariance_through_misses_and_reacquisition(self):
        class Predictor(nn.Module):
            def generate_distribution(self, histories, bev_grid_shape=None, return_kappa=False):
                count = len(histories)
                result = (np.tile([1., 0.], (count, 1)), np.tile(np.diag([4., 9.]), (count, 1, 1)))
                return (*result, np.full((count, 1), 3.)) if return_kappa else result

        args = SimpleNamespace(high_cls_thres=.4, track_match_thresh=30, track_low_match_thresh=50,
                               motion_model='bimamba', motion_history_len=3,
                               mamba_uncertainty_gate='b', mamba_uncertainty_mode='separate_full')
        config = {'base': SimpleNamespace(base=SimpleNamespace(world_grid_shape=(100, 200)))}
        tracker = build_tracker(args, config, 'cpu', Predictor())
        track = tracker.update([[10., 20.]], [.9], uncertainties=[.2])[0]
        identity = track.track_id
        tracker.update(np.empty((0, 2)), np.empty(0))
        np.testing.assert_allclose(track.xy, [11., 20.])
        np.testing.assert_array_equal(track.conds[-1][8:10], [0., 1.])
        np.testing.assert_allclose(track.conds[-1][5:8], [4., 9., 0.])
        self.assertEqual(track.motion_kappa, 3.)
        reacquired = tracker.update([[12.2, 20.]], [.9], uncertainties=[.4])[0]
        self.assertEqual(reacquired.track_id, identity)
        np.testing.assert_array_equal(reacquired.conds[-1][8:10], [1., 0.])
        np.testing.assert_allclose(reacquired.xy, [12.2, 20.])


@unittest.skipUnless(torch.cuda.is_available(), 'Real Mamba kernels require CUDA')
class BiMambaCudaTests(unittest.TestCase):
    def test_both_directions_train_with_niw_heads_implicit_inputs_and_explicit_gates(self):
        history = torch.tensor([[
            [10, 20, 1, 2, .2, 0, 0, 0, 1, 0, 0],
            [11, 22, 1, 2, 0, 4, 9, 1, 0, 1, 3],
            [12, 24, 1, 2, .7, 0, 0, 0, 1, 0, 0],
        ]], dtype=torch.float32, device='cuda').repeat(2, 1, 1)
        shape = torch.tensor([[100., 200.], [120., 240.]], device='cuda')
        for gate, head, implicit in product(('none', 'b', 'delta'), ('joint', 'separate'),
                                             ('none', 'unified', 'separate', 'separate_full')):
            with self.subTest(gate=gate, head=head, implicit=implicit):
                model = bimamba.BiMambaMotionPredictor(
                    history_len=3, d_model=8, d_state=3, d_conv=2, num_layers=2,
                    loss_type='niw', evidential_head=head, uncertainty_mode=implicit,
                    uncertainty_gate=gate,
                ).cuda()
                output_head = model.joint_head if head == 'joint' else model.delta_head
                nn.init.normal_(output_head[-1].weight, std=.01)
                loss = model(history, torch.tensor([[2., 1.], [1., 3.]], device='cuda'), shape)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for block in model.blocks:
                    for branch in (block.forward_mamba, block.backward_mamba):
                        gradient = branch.in_proj.weight.grad
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(gradient.abs().sum().item(), 0)
                    self.assertIsNot(block.forward_mamba.in_proj.weight, block.backward_mamba.in_proj.weight)
                delta, covariance, kappa = model.generate_distribution(
                    [history[0, -1:].cpu().numpy(), history[1].cpu().numpy()],
                    bev_grid_shape=shape, return_kappa=True,
                )
                self.assertEqual(delta.shape, (2, 2))
                self.assertTrue(np.isfinite(delta).all())
                self.assertTrue((np.linalg.eigvalsh(covariance) > 0).all())
                self.assertTrue((kappa > 0).all())

    def test_smooth_l1_window_truncation_padding_and_checkpoint_round_trip(self):
        kwargs = dict(history_len=3, d_model=8, d_state=3, d_conv=2, num_layers=2,
                      loss_type='smooth_l1')
        model = bimamba.BiMambaMotionPredictor(**kwargs).cuda().eval()
        long_history = np.array([[10 + i, 20 + 2*i, 1, 2] for i in range(5)], dtype=np.float32)
        histories = [long_history[-1:], long_history]
        np.testing.assert_allclose(model.generate(histories, (100, 200)), [[1., 2.], [1., 2.]], atol=1e-5)
        nn.init.normal_(model.delta_head[-1].weight, std=.01)
        loss = model(torch.from_numpy(long_history[-3:]).cuda().unsqueeze(0),
                     torch.tensor([[2., 1.]], device='cuda'), (100, 200))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        clone = bimamba.BiMambaMotionPredictor(**kwargs).cuda().eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        expected = model.generate(histories, (100, 200))
        np.testing.assert_allclose(clone.generate(histories, (100, 200)), expected, atol=1e-5)
        for index, history in enumerate(histories):
            np.testing.assert_allclose(model.generate([history[-3:]], (100, 200))[0], expected[index], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
