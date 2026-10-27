import unittest

import torch

from utils.losses_evidential import evidential_niw_loss_2d


class NIWRegularizationTests(unittest.TestCase):
    def inputs(self):
        return (
            torch.tensor([[3., 4.], [-3., 4.]], dtype=torch.float64),
            torch.zeros(2, 2, dtype=torch.float64, requires_grad=True),
            torch.tensor([[1.], [2.]], dtype=torch.float64, requires_grad=True),
            torch.tensor([[5.], [6.]], dtype=torch.float64, requires_grad=True),
            torch.tensor([[[2., 0.], [0., 4.]]] * 2, dtype=torch.float64, requires_grad=True),
        )

    def test_euclidean_penalty_is_l2_error_times_evidence(self):
        loss, stats = evidential_niw_loss_2d(
            *self.inputs(), lambda_evi=.25,
        )
        # Both residuals have length 5; their evidence values are 6 and 8.
        self.assertAlmostEqual(stats["reg"], 35.)
        self.assertAlmostEqual(loss.item(), stats["nll"] + .25 * 35.)

    def test_default_uses_euclidean_penalty_and_default_weight(self):
        loss, stats = evidential_niw_loss_2d(*self.inputs())
        self.assertAlmostEqual(stats["reg"], 35.)
        self.assertAlmostEqual(loss.item(), stats["nll"] + 1e-3 * 35.)

    def test_regularization_weight_preserves_nll_and_evidence_statistics(self):
        inputs = self.inputs()
        nll_loss, unregularized = evidential_niw_loss_2d(*inputs, lambda_evi=0.)
        regularized_loss, regularized = evidential_niw_loss_2d(*inputs, lambda_evi=.25)
        self.assertAlmostEqual(nll_loss.item(), unregularized["nll"])
        self.assertAlmostEqual((regularized_loss - nll_loss).item(), .25 * 35.)
        for key in ("nll", "kappa_mean", "nu_mean"):
            self.assertEqual(unregularized[key], regularized[key])

    def test_euclidean_penalty_is_independent_of_scale_matrix(self):
        target, mean, kappa, nu, scale = self.inputs()
        _, original = evidential_niw_loss_2d(
            target, mean, kappa, nu, scale,
        )
        # Change both the size and correlation of Psi while holding error/evidence fixed.
        changed_scale = torch.tensor([[[20., 0.], [7., 40.]]] * 2, dtype=torch.float64)
        _, changed = evidential_niw_loss_2d(
            target, mean, kappa, nu, changed_scale,
        )
        self.assertEqual(original["reg"], changed["reg"])
        self.assertNotEqual(original["nll"], changed["nll"])

    def test_euclidean_penalty_gradients_reach_mean_and_evidence_but_not_scale(self):
        target, mean, kappa, nu, scale = self.inputs()
        regularized, _ = evidential_niw_loss_2d(
            target, mean, kappa, nu, scale, lambda_evi=.25,
        )
        nll, _ = evidential_niw_loss_2d(
            target, mean, kappa, nu, scale, lambda_evi=0.,
        )
        gradients = torch.autograd.grad((regularized - nll) / .25, (mean, kappa, nu, scale))
        torch.testing.assert_close(
            gradients[0], torch.tensor([[-1.8, -2.4], [2.4, -3.2]], dtype=torch.float64),
        )
        for gradient in gradients[1:3]:
            torch.testing.assert_close(gradient, torch.full((2, 1), 2.5, dtype=torch.float64))
        torch.testing.assert_close(gradients[3], torch.zeros_like(scale), atol=1e-12, rtol=0.)

    def test_exact_prediction_has_zero_euclidean_penalty_and_finite_gradients(self):
        _, mean, kappa, nu, scale = self.inputs()
        loss, stats = evidential_niw_loss_2d(
            mean.detach().clone(), mean, kappa, nu, scale,
        )
        self.assertEqual(stats["reg"], 0.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (mean, kappa, nu, scale):
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_weight_is_rejected(self):
        for weight in (-1., float("inf"), float("nan")):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "lambda_evi"):
                evidential_niw_loss_2d(*self.inputs(), lambda_evi=weight)


if __name__ == "__main__":
    unittest.main()
