"""Resume plots consume the same evaluation pass, without rerunning inference."""

import argparse
import ast
import csv
import io
import json
import os
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
        self.assertFalse(plot_settings().plot_motion_uncertainty)
        self.assertEqual(plot_settings().motion_plot_object_selection, "id")
        self.assertIsNone(plot_settings().motion_plot_object_selection_seed)
        self.assertTrue(plot_settings("--plot_motion_uncertainty").plot_motion_uncertainty)
        self.assertFalse(plot_settings("--no-plot_motion_uncertainty").plot_motion_uncertainty)
        self.assertEqual(viz.resume_plot_plan(plot_settings(), dataset), {0: [10, 20], 1: [100, 110]})
        args = plot_settings("--motion_plot_seq_ids", "1", "--motion_plot_frames", "110")
        self.assertEqual(viz.resume_plot_plan(args, dataset), {1: [110]})
        for options in (("--motion_plot_seq_ids", "2"), ("--motion_plot_frames", "15"),
                        ("--motion_plot_max_objects", "0"), ("--motion_plot_xlim", "5", "4"),
                        ("--motion_plot_object_selection", "random", "--motion_plot_object_selection_seed", "-1"),
                        ("--motion_plot_object_selection", "both", "--motion_plot_max_objects", "1")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                viz.resume_plot_plan(plot_settings(*options), dataset)

    def test_no_motion_predictions_do_not_break_completed_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(viz.plot_resume_results(plot_settings(), None, 1, folder, {0: [10]}), [])

    def test_automatic_selection_uses_real_consecutive_ids_after_initial_observation(self):
        dataset = SimpleNamespace(dataset_list=["seq"], datasets=[SimpleNamespace(frames=list(range(200, 210)))])
        args = plot_settings("--motion_plot_consecutive", "5")
        self.assertEqual(viz.resume_plot_plan(args, dataset), {0: [201, 202, 203, 204, 205]})
        args.motion_plot_start_frame = 204
        self.assertEqual(viz.resume_plot_plan(args, dataset), {0: [204, 205, 206, 207, 208]})
        dataset.datasets[0].frames = [10, 11, 13, 20, 21, 22, 23, 24]
        args.motion_plot_start_frame = None
        self.assertEqual(viz.resume_plot_plan(args, dataset), {0: [20, 21, 22, 23, 24]})

    def test_sampling_is_not_silently_treated_as_original_frame_continuity(self):
        dataset = SimpleNamespace(dataset_list=["seq"], datasets=[SimpleNamespace(frames=[0, 10, 20, 30, 40, 50])])
        args = plot_settings("--motion_plot_consecutive", "5")
        args.dataset = "gmvd_train"
        with self.assertRaisesRegex(ValueError, "no run of 5.*steps=\\[10\\].*gmvd_eval_all_frames"):
            viz.resume_plot_plan(args, dataset)
        for options in (("--motion_plot_consecutive", "0"), ("--motion_plot_start_frame", "0")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                viz.resume_plot_plan(plot_settings(*options), dataset)
        dataset.datasets[0].frames = [0]
        with self.assertRaisesRegex(ValueError, "no run of 5"):
            viz.resume_plot_plan(args, dataset)

    def test_invalid_explicit_ids_report_actual_range_and_examples(self):
        dataset = SimpleNamespace(dataset_list=["seq"], datasets=[SimpleNamespace(frames=list(range(100)))])
        with self.assertRaisesRegex(ValueError, "count=100, range=0..99.*IDs=.*motion_plot_consecutive"):
            viz.resume_plot_plan(plot_settings("--motion_plot_frames", "100", "101", "102", "103", "104"), dataset)

    def test_all_frames_override_applies_only_to_gmvd_evaluation_rows(self):
        # Execute the actual CLI dataset construction branch with lightweight
        # constructors; avoid opening images or importing GPU libraries.
        source = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        entry = next(node for node in tree.body if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
        branch = next(node for node in entry.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "'wildtrack' in args.dataset")
        code = compile(ast.Module(body=[branch], type_ignores=[]), str(source), "exec")
        for enabled in (False, True):
            args = SimpleNamespace(dataset="gmvd_train", resume=None, gmvd_eval_all_frames=enabled, aug=False,
                                   motion_model="mamba", motion_train_history="detection",
                                   motion_history_len=15, bev_gaussian_sigma=3.)
            constructor = Mock(side_effect=lambda base, **kwargs: kwargs)
            namespace = dict(args=args, os=os, csv=csv, GetDataset=constructor,
                             MultiviewX=lambda path: SimpleNamespace(root=path), train_transform=None)
            with patch("builtins.open", return_value=io.StringIO(
                    "train,./scene/train,1,100\ntest,./scene/test,0,100\n")):
                exec(code, namespace)
            self.assertEqual(namespace["train_dataset_list"][0]["sample_require"], 100)
            self.assertEqual(namespace["test_dataset_list"][0]["sample_require"], 0 if enabled else 100)


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
                self.datasets = [SimpleNamespace(frames=[10, 11, 12], base=base, grid_reduce=2) for _ in range(2)]
                self.dicts = {name: {"base": sub} for name, sub in zip(self.dataset_list, self.datasets)}

            def __len__(self):
                return 6

            def __getitem__(self, index):
                return dict(root=self.dataset_list[index // 3], frame=[10, 11, 12][index % 3],
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

        settings = plot_settings("--motion_plot_seq_ids", "1", "--motion_plot_consecutive", "2",
                                 "--motion_plot_component", "epistemic", "--motion_plot_formats", "png",
                                 "--motion_plot_object_selection", "random", "--motion_plot_object_selection_seed", "0",
                                 "--motion_plot_max_objects", "1")
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
                    options = {"plot_motion_uncertainty": True} if enabled else {}
                    score = main.test(model, 1, loader, str(folder / "eval.txt"), **options)
                self.assertEqual(score, 50.)
                self.assertEqual(decoder.call_count, 6)  # Both sequences and all frames, irrespective of plot filters.
            for filename in ("mota_pred.txt", "mota_gt.txt", "all_res.txt"):
                self.assertEqual((root / "False" / filename).read_bytes(), (root / "True" / filename).read_bytes())
            self.assertFalse((root / "False" / "motion_uncertainty_plots").exists())
            with (root / "False" / "motion_uncertainty_epoch_1.csv").open() as stream:
                baseline = list(csv.DictReader(stream))
            with (root / "True" / "motion_uncertainty_epoch_1.csv").open() as stream:
                plotted = list(csv.DictReader(stream))
            self.assertEqual(baseline, [{key: value for key, value in row.items() if key != "kappa"} for row in plotted])
            self.assertEqual({row["kappa"] for row in plotted if row["sequence"] == dataset.dataset_list[1]}, {"3.0"})
            self.assertEqual({row["kappa"] for row in plotted if row["sequence"] == dataset.dataset_list[0]}, {""})
            output = root / "True" / "motion_uncertainty_plots" / "seq_1_seq1"
            self.assertEqual(len(list(output.glob("*.png"))), 1)
            self.assertTrue((output / "trajectory_000011_000012_epistemic.png").is_file())
            self.assertEqual(list(output.glob("frame_*")), [])
            self.assertFalse((output.parent / "seq_0_seq1").exists())
            with (output / "plotted_points_epistemic.csv").open() as stream:
                points = list(csv.DictReader(stream))
            motion = next(row for row in points if row["source"] == "motion")
            self.assertEqual(float(motion["cov_xx"]), 1.)
            self.assertEqual({row["frame"] for row in points}, {"11", "12"})
            metadata = json.loads((output / "plot_settings_epistemic.json").read_text())
            self.assertEqual(metadata["object_selection"], "random")
            self.assertEqual(metadata["effective_object_selection_seed"], 0)
            self.assertEqual(metadata["evaluation_checkpoint"], "weights.pth")
            self.assertEqual(metadata["evaluation_epoch"], 1)
            self.assertEqual(metadata["layout"], "trajectory")


if __name__ == "__main__":
    unittest.main()
