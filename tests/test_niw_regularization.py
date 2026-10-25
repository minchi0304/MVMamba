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
            *self.inputs(), lambda_evi=.25, regularizer_type="euclidean",
        )
        # Both residuals have length 5; their evidence values are 6 and 8.
        self.assertAlmostEqual(stats["reg"], 35.)
        self.assertAlmostEqual(loss.item(), stats["nll"] + .25 * 35.)

    def test_default_preserves_psi_whitened_penalty(self):
        inputs = self.inputs()
        default_loss, stats = evidential_niw_loss_2d(*inputs)
        explicit_loss, _ = evidential_niw_loss_2d(*inputs, regularizer_type="mahalanobis")
        # Whitening by diag(2, 4) gives residuals (+/-1.5, 1).
        self.assertAlmostEqual(stats["reg"], (1.5 ** 2 + 1. + 1e-8) ** .5 * 7.)
        torch.testing.assert_close(default_loss, explicit_loss)

    def test_switching_regularizer_preserves_nll_and_evidence_statistics(self):
        inputs = self.inputs()
        maha_loss, maha = evidential_niw_loss_2d(*inputs, lambda_evi=0.)
        euclidean_loss, euclidean = evidential_niw_loss_2d(
            *inputs, lambda_evi=0., regularizer_type="euclidean",
        )
        torch.testing.assert_close(maha_loss, euclidean_loss)
        for key in ("nll", "kappa_mean", "nu_mean"):
            self.assertEqual(maha[key], euclidean[key])

    def test_euclidean_penalty_is_independent_of_scale_matrix(self):
        target, mean, kappa, nu, scale = self.inputs()
        _, original = evidential_niw_loss_2d(
            target, mean, kappa, nu, scale, regularizer_type="euclidean",
        )
        # Change both the size and correlation of Psi while holding error/evidence fixed.
        changed_scale = torch.tensor([[[20., 0.], [7., 40.]]] * 2, dtype=torch.float64)
        _, changed = evidential_niw_loss_2d(
            target, mean, kappa, nu, changed_scale, regularizer_type="euclidean",
        )
        self.assertEqual(original["reg"], changed["reg"])
        self.assertNotEqual(original["nll"], changed["nll"])

    def test_euclidean_penalty_gradients_reach_mean_and_evidence_but_not_scale(self):
        target, mean, kappa, nu, scale = self.inputs()
        regularized, _ = evidential_niw_loss_2d(
            target, mean, kappa, nu, scale, lambda_evi=.25, regularizer_type="euclidean",
        )
        nll, _ = evidential_niw_loss_2d(
            target, mean, kappa, nu, scale, lambda_evi=0., regularizer_type="euclidean",
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
            mean.detach().clone(), mean, kappa, nu, scale, regularizer_type="euclidean",
        )
        self.assertEqual(stats["reg"], 0.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (mean, kappa, nu, scale):
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_regularizer_or_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "regularizer_type"):
            evidential_niw_loss_2d(*self.inputs(), regularizer_type="invalid")
        for weight in (-1., float("inf"), float("nan")):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "lambda_evi"):
                evidential_niw_loss_2d(*self.inputs(), lambda_evi=weight)


if __name__ == "__main__":
    unittest.main()
