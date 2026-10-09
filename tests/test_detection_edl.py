import unittest

import numpy as np
import torch

from utils.loss import EvidentialOccupancyLoss
from utils.utils import get_occupancy_prob, get_occupancy_uncertainty


class DetectionEvidentialTests(unittest.TestCase):
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
