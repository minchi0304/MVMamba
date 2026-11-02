"""Inference orchestration tests; CUDA kernels are exercised on the target GPU."""

import argparse
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import motion_uncertainty_inference as inference
import visualize_motion_uncertainty as viz

try:
    import torch
except ImportError:
    torch = None


class InferenceSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "weights.pth"
        self.checkpoint.touch()

    def arguments(self, *options):
        return viz.build_parser().parse_args(["--checkpoint", str(self.checkpoint), *options])

    def test_saved_architecture_and_explicit_overrides(self):
        (self.root / "log.txt").write_text("Settings:\n" + repr(dict(
            dataset="gmvd_train", mamba_d_model=1024, evidential_head="joint",
            mamba_uncertainty_mode="separate_full", motion_history_len=13,
            implicit_motion_uncertainty_type="aleatoric", track_buffer=7, pretrained=True,
        )), encoding="utf-8")
        args = self.arguments("--data_root", str(self.root), "--track_buffer", "9")
        settings, config = inference.resolve_settings(args)
        self.assertEqual(config, self.root / "log.txt")
        self.assertEqual(settings.mamba_d_model, 1024)
        self.assertEqual(settings.evidential_head, "joint")
        self.assertEqual(settings.implicit_motion_uncertainty_type, "aleatoric")
        self.assertEqual(settings.motion_history_len, 13)
        self.assertEqual(settings.track_buffer, 9)
        self.assertEqual(settings.data_root, str(self.root))
        self.assertFalse(hasattr(settings, "pretrained"))

    def test_json_config_and_invalid_settings_fail_early(self):
        config = self.root / "model.json"
        config.write_text(json.dumps(dict(dataset="citytrack", motion_model="bimamba", mamba_history_mode="stateful")))
        with self.assertRaisesRegex(ValueError, "BiMamba"):
            inference.resolve_settings(self.arguments("--model-config", str(config)))
        with self.assertRaisesRegex(ValueError, "Specify --dataset"):
            inference.resolve_settings(self.arguments())
        with self.assertRaisesRegex(ValueError, "positive"):
            inference.resolve_settings(self.arguments("-d", "citytrack", "--motion_history_len", "0"))
        config.write_text("[]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            inference.read_settings(config)

    def test_log_values_are_not_executed(self):
        config = self.root / "log.txt"
        config.write_text("Settings:\n__import__('os').getcwd()\n")
        with self.assertRaisesRegex(ValueError, "Settings dictionary"):
            inference.read_settings(config)

    def test_sequence_mapping_uses_evaluation_order_and_data_root(self):
        split = self.root / "split.csv"
        split.write_text("train,./scene1/config1/seq1,1,100\n"
                         "test,./scene5/config2/seq1,0,100\n"
                         "test,./scene5/config1/seq1,0,100\n")
        settings, _ = inference.resolve_settings(self.arguments("-d", "gmvd_train", "--data_root", str(self.root)))
        specs = inference.sequence_specs(settings, split)
        self.assertEqual([s["sequence_id"] for s in specs], [0, 1])
        self.assertEqual(Path(specs[0]["root"]), self.root / "scene5/config2/seq1")
        self.assertEqual(inference.choose_sequence(specs, sequence_id=1), specs[1])
        self.assertEqual(inference.choose_sequence(specs, "scene5/config1/seq1"), specs[1])
        with self.assertRaisesRegex(ValueError, "one sequence"):
            inference.choose_sequence(specs, "seq1")
        with self.assertRaisesRegex(ValueError, "matching"):
            inference.choose_sequence(specs, specs[0]["root"], 1)
        # Listing does not require importing a model or calling torch.load.
        args = self.arguments("-d", "gmvd_train", "--gmvd-csv", str(split), "--list-sequences")
        with patch.object(inference, "build_model", side_effect=AssertionError("model must not load")):
            self.assertEqual(viz.run(args), [])

    def test_sparse_plot_request_preserves_all_prior_evaluation_frames(self):
        processed, selected = inference.inference_frames([0, 10, 20, 30, 40], [30, 10, 30])
        self.assertEqual(processed, [0, 10, 20, 30])
        self.assertEqual(selected, [10, 30])
        with self.assertRaisesRegex(ValueError, "sampling"):
            inference.inference_frames([0, 10, 20], [11])

    def test_checkpoint_cli_plots_only_fresh_results_with_component_decomposition(self):
        args = self.arguments("--frames", "20", "--component", "aleatoric", "--max-objects", "1",
                              "--output-dir", str(self.root / "plots"))
        result = dict(motion=[dict(sequence="/dataset/scene", frame=20, track_id=42, matched=1,
                                  predicted_x=10., predicted_y=20., mean_dx=1., mean_dy=2.,
                                  cov_xx=4., cov_xy=-1., cov_yy=9., kappa=3.)],
                      gt=np.array([[7, 20, 9, -1, -1, -1, -1, 1, 11, 21, -1]]),
                      detections=np.array([[20, 12, 22, .9, 7]]), sequence="/dataset/scene",
                      processed_frames=[0, 10, 20], selected_frames=[20], detection_order="xy")

        def fresh_inference(arguments):
            paths = inference.write_inference(self.root / "fresh", result, {"checkpoint": str(self.checkpoint)})
            for key, value in paths.items():
                setattr(arguments, key, value)
            arguments.sequence, arguments.sequence_id = "/dataset/scene", 7
            return True

        with patch.object(viz, "run_checkpoint", side_effect=fresh_inference) as runner:
            files = viz.run(args)
        runner.assert_called_once()
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].is_file())
        with (files[0].parent / "plotted_points_aleatoric.csv").open() as stream:
            points = list(csv.DictReader(stream))
        self.assertEqual({p["source"] for p in points}, {"motion", "gt", "detection"})
        motion = next(p for p in points if p["source"] == "motion")
        self.assertEqual(float(motion["cov_xx"]), 3.)
        self.assertEqual(motion["sequence_id"], "7")

    def test_checkpoint_rejects_mixing_old_gt_or_detections(self):
        with self.assertRaisesRegex(ValueError, "omit --gt"):
            viz.run(self.arguments("--gt", "old_gt.txt"))


@unittest.skipIf(torch is None, "PyTorch is not installed in the standalone plotting environment")
class InferenceTrackerTests(unittest.TestCase):
    def settings(self, **overrides):
        defaults = {key: value[0] for key, value in inference.OPTIONS.items()}
        defaults.update(dataset="gmvd_test", num_workers=0, motion_history_len=3)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def predictor(self):
        class Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.token_widths = []

            def generate_distribution(self, histories, bev_grid_shape=None, return_kappa=False):
                self.token_widths.extend(len(history[-1]) for history in histories)
                count = len(histories)
                result = (np.tile([2., 3.], (count, 1)), np.tile(np.eye(2)[None] * 4., (count, 1, 1)))
                return (*result, np.full((count, 1), 3.)) if return_kappa else result

            def stateful_step_distribution(self, tokens, states=None, bev_grid_shape=None, return_kappa=False):
                motion, covariance, kappa = self.generate_distribution([[token] for token in tokens], return_kappa=True)
                next_states = [1 if state is None else state + 1 for state in states]
                result = (motion, covariance, next_states)
                return (*result, kappa) if return_kappa else result

        return Predictor()

    def test_logging_preserves_window_and_stateful_predictions_and_history_tokens(self):
        from utils.utils import build_tracker
        config = {"base": SimpleNamespace(base=SimpleNamespace(world_grid_shape=(100, 200)))}
        for mode in ("window", "stateful"):
            with self.subTest(mode=mode):
                settings = self.settings(mamba_history_mode=mode)
                predictor = self.predictor()
                logging = inference.build_logging_tracker(settings, config, "cpu", predictor)
                baseline = build_tracker(settings, config, "cpu", self.predictor())
                for detections in ([[10., 20.]], [[13., 24.]], [], [[17., 30.]]):
                    for tracker in (logging, baseline):
                        tracker.update(detections, [.9] * len(detections))
                    self.assertEqual(len(logging.last_motion_predictions), len(baseline.last_motion_predictions))
                    for actual, expected in zip(logging.last_motion_predictions, baseline.last_motion_predictions):
                        self.assertEqual(actual["kappa"], 3.)
                        for key in ("predicted_x", "predicted_y", "cov_xx", "cov_xy", "cov_yy", "matched"):
                            self.assertEqual(actual[key], expected[key])
                    if logging.tracked_stracks:
                        np.testing.assert_array_equal(logging.tracked_stracks[0].conds, baseline.tracked_stracks[0].conds)
                self.assertEqual(set(predictor.token_widths), {10})

    def test_shared_tracking_pipeline_warms_history_and_preserves_original_sequence_id(self):
        from utils import utils

        class Dataset:
            def __init__(self):
                self.frames = [10, 20, 30, 40]
                self.base = SimpleNamespace(indexing="xy", world_grid_shape=(100, 200), metric_grid_cell=.1, grid_cell=10.)
                self.grid_reduce = 2
                self.datasets = [self]
                self.dataset_list = ["/dataset/scene5/config2/seq1"]
                self.dicts = {self.dataset_list[0]: {"base": self}}

            def __len__(self):
                return len(self.frames)

            def __getitem__(self, index):
                return dict(root=self.dataset_list[0], frame=self.frames[index],
                            grid_gt=np.array([[11., 22., 99.]], dtype=np.float32), grid_gt_count=1), {}

        seen = []

        def decode(model, dataset, item, index, device, settings):
            seen.append(int(item["frame"][0]))
            xy = torch.tensor([[5. + index, 10. + index]])
            scores = torch.tensor([[.9]])
            return dict(xy_all=xy, scores_all=scores, uncertainty_all=torch.tensor([.1]), eval_xy=xy, eval_scores=scores)

        with patch.object(utils, "decode_frame", side_effect=decode):
            result = inference.infer_sequence(SimpleNamespace(motion_predictor=self.predictor()), Dataset(),
                                              self.settings(), {"sequence_id": 5}, torch.device("cpu"), [30])
        self.assertEqual(seen, [10, 20, 30])
        self.assertEqual(result["selected_frames"], [30])
        self.assertEqual([r["frame"] for r in result["motion"]], [20, 30])
        self.assertEqual(result["motion"][-1]["predicted_y"], 25.)  # Last detection y=22 + predicted dy=3.
        self.assertEqual(result["motion"][-1]["kappa"], 3.)
        np.testing.assert_array_equal(result["gt"][:, 0], [5, 5, 5])
        np.testing.assert_array_equal(result["detections"][:, 4], [5, 5, 5])
        np.testing.assert_array_equal(result["detections"][-1, 1:3], [14., 24.])

    def test_strict_checkpoint_load_and_dataparallel_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pth"
            model = torch.nn.Linear(2, 3)
            torch.save({"state_dict": {"module." + k: v for k, v in model.state_dict().items()}}, path)
            restored = torch.nn.Linear(2, 3)
            inference.load_checkpoint(restored, path)
            torch.testing.assert_close(model.weight, restored.weight)
            torch.save({"weight": model.weight}, path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                inference.load_checkpoint(restored, path)


if __name__ == "__main__":
    unittest.main()
