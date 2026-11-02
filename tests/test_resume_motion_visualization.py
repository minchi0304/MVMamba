"""Resume plots consume the same evaluation pass, without rerunning inference."""

import argparse
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import visualize_motion_uncertainty as viz

try:
    import torch
except ImportError:
    torch = None


def plot_settings(*arguments):
    parser = argparse.ArgumentParser()
    viz.add_resume_plot_arguments(parser)
    args = parser.parse_args(arguments)
    args.resume = "weights.pth"
    return args


class ResumePlotSelectionTests(unittest.TestCase):
    def test_default_and_disabled_plots_and_sequence_frame_selection(self):
        dataset = SimpleNamespace(dataset_list=["seq1", "seq2"], datasets=[
            SimpleNamespace(frames=[10, 20]), SimpleNamespace(frames=[100, 110])])
        self.assertTrue(plot_settings().plot_motion_uncertainty)
        self.assertFalse(plot_settings("--no-plot_motion_uncertainty").plot_motion_uncertainty)
        self.assertEqual(viz.resume_plot_plan(plot_settings(), dataset), {0: [10, 20], 1: [100, 110]})
        args = plot_settings("--motion_plot_seq_ids", "1", "--motion_plot_frames", "110")
        self.assertEqual(viz.resume_plot_plan(args, dataset), {1: [110]})
        for options in (("--motion_plot_seq_ids", "2"), ("--motion_plot_frames", "15"),
                        ("--motion_plot_max_objects", "0"), ("--motion_plot_xlim", "5", "4")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                viz.resume_plot_plan(plot_settings(*options), dataset)

    def test_no_motion_predictions_do_not_break_completed_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(viz.plot_resume_results(plot_settings(), None, 1, folder, {0: [10]}), [])


@unittest.skipIf(torch is None, "PyTorch unavailable in standalone plotting environment")
class ResumeEvaluationIntegrationTests(unittest.TestCase):
    def test_resume_uses_one_evaluation_pass_and_preserves_tracking_results(self):
        import main
        from motion_uncertainty_inference import OPTIONS
        from tracker.multitracker import STrack
        from utils import utils

        class Predictor(torch.nn.Module):
            def generate_distribution(self, histories, bev_grid_shape=None, return_kappa=False):
                count = len(histories)
                result = (np.tile([2., 3.], (count, 1)), np.tile(np.eye(2)[None] * 4., (count, 1, 1)))
                return (*result, np.full((count, 1), 3.)) if return_kappa else result

        class Dataset:
            dataset_list = ["/dataset/scene/config1/seq1", "/dataset/scene/config2/seq1"]

            def __init__(self):
                base = SimpleNamespace(indexing="xy", world_grid_shape=(100, 200), metric_grid_cell=.1, grid_cell=10.)
                self.datasets = [SimpleNamespace(frames=[10, 20, 30], base=base, grid_reduce=2) for _ in range(2)]
                self.dicts = {name: {"base": sub} for name, sub in zip(self.dataset_list, self.datasets)}

            def __len__(self):
                return 6

            def __getitem__(self, index):
                return dict(root=self.dataset_list[index // 3], frame=[10, 20, 30][index % 3],
                            grid_gt=np.array([[14., 25., 99.]], dtype=np.float32), grid_gt_count=1), {}

        def decode(model, dataset, item, index, device, args):
            xy = torch.tensor([[5. + index % 3, 10. + index % 3]])
            scores = torch.tensor([[.9]])
            return dict(xy_all=xy, scores_all=scores, uncertainty_all=torch.tensor([.1]),
                        eval_xy=xy, eval_scores=scores)

        def finalize(epoch, loader, res_path, plot_tracking, measure_fps, all_res_rows,
                     tracking, gt, uncertainty, *args, **kwargs):
            folder = str(Path(res_path).parent)
            utils.write_tracking_result_files(folder, loader, utils.stack_detection_results(all_res_rows, "cpu"), tracking, gt)
            utils.write_motion_uncertainty_file(folder, epoch, loader, uncertainty)
            return 50.

        settings = plot_settings("--motion_plot_seq_ids", "1", "--motion_plot_frames", "30",
                                 "--motion_plot_component", "epistemic", "--motion_plot_formats", "png")
        for key, (default, _, _) in OPTIONS.items():
            setattr(settings, key, default)
        settings.dataset = "gmvd_test"
        model = SimpleNamespace(eval=Mock(), motion_predictor=Predictor())
        dataset = Dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        previous_count = STrack._count
        self.addCleanup(setattr, STrack, "_count", previous_count)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for enabled in (False, True):
                STrack._count = 0
                folder = root / str(enabled)
                folder.mkdir()
                with patch.multiple(main, create=True, args=settings, device=torch.device("cpu"), logdir=str(folder)), \
                        patch.object(utils, "decode_frame", side_effect=decode) as decoder, \
                        patch.object(utils, "finalize_results", side_effect=finalize), \
                        patch.object(viz, "run_checkpoint", side_effect=AssertionError("Must not rerun the model")):
                    score = main.test(model, 1, loader, str(folder / "eval.txt"), plot_motion_uncertainty=enabled)
                self.assertEqual(score, 50.)
                self.assertEqual(decoder.call_count, 6)  # Both sequences and all frames, irrespective of plot filters.
            for filename in ("mota_pred.txt", "mota_gt.txt", "all_res.txt"):
                self.assertEqual((root / "False" / filename).read_bytes(), (root / "True" / filename).read_bytes())
            with (root / "False" / "motion_uncertainty_epoch_1.csv").open() as stream:
                baseline = list(csv.DictReader(stream))
            with (root / "True" / "motion_uncertainty_epoch_1.csv").open() as stream:
                plotted = list(csv.DictReader(stream))
            self.assertEqual(baseline, [{key: value for key, value in row.items() if key != "kappa"} for row in plotted])
            self.assertEqual({row["kappa"] for row in plotted if row["sequence"] == dataset.dataset_list[1]}, {"3.0"})
            self.assertEqual({row["kappa"] for row in plotted if row["sequence"] == dataset.dataset_list[0]}, {""})
            output = root / "True" / "motion_uncertainty_plots" / "seq_1_seq1"
            self.assertEqual(len(list(output.glob("*.png"))), 1)
            self.assertFalse((output.parent / "seq_0_seq1").exists())
            with (output / "plotted_points_epistemic.csv").open() as stream:
                points = list(csv.DictReader(stream))
            motion = next(row for row in points if row["source"] == "motion")
            self.assertEqual(float(motion["cov_xx"]), 1.)
            self.assertEqual({row["frame"] for row in points}, {"30"})
            metadata = json.loads((output / "plot_settings_epistemic.json").read_text())
            self.assertEqual(metadata["evaluation_checkpoint"], "weights.pth")
            self.assertEqual(metadata["evaluation_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
