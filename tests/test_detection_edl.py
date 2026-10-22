import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from dataloader.dataloader import GetDataset
from detector.multiview_model import MultiView_Detection
from utils.loss import EvidentialOccupancyLoss
from utils.utils import decoder, get_occupancy_prob, get_occupancy_uncertainty


class DetectionEvidentialTests(unittest.TestCase):
    def test_detector_returns_batched_occupancy_and_trains_center_head(self):
        backbone = nn.Sequential(nn.Conv2d(3, 512, 1), nn.Identity(), nn.Identity())
        # Use real decoder layers without downloading pretrained test weights.
        with patch('detector.decoder.torchvision.models.resnet18', side_effect=lambda **_: resnet18(weights=None)):
            model = MultiView_Detection(backbone, '.', device=torch.device('cpu'),
                                        multi_task_loss='uncertainty', use_motion_loss=True)
        dataset = SimpleNamespace(dicts={'scene': {
            'upsample_shape': [16, 16], 'reducedgrid_shape': [16, 16],
            'proj_mats': torch.eye(3).unsqueeze(0), 'coord_map': torch.zeros(1, 2, 16, 16),
        }})
        # Both single-frame and recursive multi-frame forwards must return tensors.
        prediction = model(dataset, ['scene', 'scene'], torch.randn(2, 1, 3, 16, 16))
        self.assertEqual(prediction.shape, (2, 2, 16, 16))
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4, 5] = 1.
        loss = EvidentialOccupancyLoss()(prediction, target, target)
        loss.backward()
        gradient = model.instance_center_head[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        task_weights = {name for name, _ in model.named_parameters() if name.endswith('_weight')}
        self.assertEqual(task_weights, {'instance_weight', 'motion_weight'})

    def test_bev_targets_and_decoding_use_cell_coordinates(self):
        dataset = SimpleNamespace(reducedgrid_shape=[8, 8], grid_reduce=4, bev_gaussian_sigma=1.5)
        center, valid, identities = GetDataset.get_bev_gt(dataset, np.array([[9.25, 13.5]]), [7])
        self.assertTrue(valid[0, 3, 2])
        self.assertEqual(int(identities[0, 3, 2]), 7)
        xy, scores, indices = decoder(center.unsqueeze(0), K=1, return_inds=True)
        torch.testing.assert_close(xy, torch.tensor([[[2., 3.]]]))
        torch.testing.assert_close(scores, torch.ones(1, 1, 1))
        self.assertEqual(int(indices[0, 0]), 3 * 8 + 2)

    def test_probability_and_uncertainty_follow_beta_evidence(self):
        neutral = torch.zeros(1, 2, 1, 1)
        probability = get_occupancy_prob(neutral)
        uncertainty = get_occupancy_uncertainty(neutral)

        self.assertAlmostEqual(float(probability.item()), 0.5, places=7)
        expected_uncertainty = 1.0 / (np.log(2.0) + 1.0)
        self.assertAlmostEqual(
            float(uncertainty.item()),
            expected_uncertainty,
            places=7,
        )

        center_evidence = torch.tensor([[[[6.0]], [[-6.0]]]])
        self.assertGreater(float(get_occupancy_prob(center_evidence).item()), 0.8)

        strong_evidence = torch.full((1, 2, 1, 1), 6.0)
        self.assertLess(
            float(get_occupancy_uncertainty(strong_evidence).item()),
            float(uncertainty.item()),
        )

    def test_evidential_loss_is_finite_and_backpropagates(self):
        prediction = torch.randn(2, 2, 3, 4, requires_grad=True)
        center_target = torch.zeros(2, 3, 4)
        center_target[0, 1, 2] = 1.0
        center_target[1, 2, 1] = 1.0
        gaussian_target = center_target.clone()

        loss = EvidentialOccupancyLoss()(
            prediction,
            center_target,
            gaussian_target,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
