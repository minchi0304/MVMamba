"""Regression tests for complete detection evaluation within the selected split."""

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation.CLEAR_MOD_HUN import CLEAR_MOD_HUN
from evaluation.mod import modMetricsCalculator
from utils.utils import evaluate_detection_results


class DetectionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gt_path = self.root / "gt.txt"
        self.pred_path = self.root / "pred.txt"

    def evaluate(self, gt, pred, **kwargs):
        np.savetxt(self.gt_path, np.asarray(gt, dtype=float).reshape(-1, 3))
        np.savetxt(self.pred_path, np.asarray(pred, dtype=float).reshape(-1, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return modMetricsCalculator(self.pred_path, self.gt_path, **kwargs)

    def test_missing_first_middle_and_last_prediction_frame(self):
        gt = [[frame, 0, 0] for frame in [10, 20, 30]]
        for missing in [10, 20, 30]:
            with self.subTest(missing=missing):
                pred = [row for row in gt if row[0] != missing]
                np.testing.assert_allclose(self.evaluate(gt, pred),
                                           [200 / 3, 100, 200 / 3, 100])

    def test_prediction_only_frames_before_and_after_gt(self):
        gt = [[10, 0, 0], [20, 0, 0]]
        for extra_frame in [0, 30]:
            with self.subTest(extra_frame=extra_frame):
                np.testing.assert_allclose(self.evaluate(gt, gt + [[extra_frame, 0, 0]]),
                                           [100, 200 / 3, 50, 100])

    def test_misses_and_false_positives_in_different_frames(self):
        result = self.evaluate([[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]])
        np.testing.assert_allclose(result, [50, 50, 0, 100])

    def test_empty_gt_predictions_and_both_empty(self):
        for gt, pred in [([], []), ([[0, 0, 0]], []), ([], [[0, 0, 0]])]:
            with self.subTest(gt=gt, pred=pred):
                np.testing.assert_array_equal(self.evaluate(gt, pred), [0, 0, 0, 0])

    def test_no_matches_has_finite_zero_metrics(self):
        np.testing.assert_array_equal(self.evaluate([[0, 0, 0]], [[0, 100, 0]]),
                                      [0, 0, 0, 0])

    def test_explicit_frames_exclude_training_and_skipped_frames(self):
        gt = [[0, 0, 0], [4800, 0, 0], [4801, 0, 0], [4802, 0, 0]]
        pred = [[0, 0, 0], [4800, 0, 0], [6000, 0, 0]]
        result = self.evaluate(gt, pred, frame_ids=range(4800, 4803, 2))
        np.testing.assert_allclose(result, [50, 100, 50, 100])

    def test_explicit_frames_keep_prediction_only_frame(self):
        result = self.evaluate([[0, 0, 0], [1, 0, 0]],
                               [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                               frame_ids=[0, 1, 2])
        np.testing.assert_allclose(result, [100, 200 / 3, 50, 100])

    def test_empty_evaluation_frame_selection(self):
        result = self.evaluate([[0, 0, 0]], [[0, 0, 0]], frame_ids=[])
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_unsorted_frames_and_multiple_people(self):
        gt = [[9, 0, 0], [7, 100, 0], [7, 0, 0]]
        pred = [[7, 1, 0], [9, 0, 0], [7, 101, 0]]
        np.testing.assert_allclose(self.evaluate(gt, pred),
                                   [100, 100, 100, (95 + 95 + 100) / 3])

    def test_distance_threshold_and_strict_boundary(self):
        for threshold in [20, 32]:
            with self.subTest(threshold=threshold):
                result = self.evaluate([[0, 0, 0]], [[0, threshold - 1, 0]],
                                       distance_threshold=threshold)
                np.testing.assert_allclose(result, [100, 100, 100, 100 / threshold])
                result = self.evaluate([[0, 0, 0]], [[0, threshold, 0]],
                                       distance_threshold=threshold)
                np.testing.assert_array_equal(result, [0, 0, 0, 0])
        np.testing.assert_array_equal(self.evaluate([[0, 0, 0]], [[0, 24, 0]]),
                                      [0, 0, 0, 0])

    def test_core_evaluator_accepts_empty_inputs(self):
        empty = np.empty((0, 4))
        one = np.array([[0, 0, 0, 0]])
        for gt, pred in [(empty, empty), (one, empty), (empty, one)]:
            with self.subTest(gt_count=len(gt), pred_count=len(pred)):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    np.testing.assert_array_equal(CLEAR_MOD_HUN(gt, pred), [0, 0, 0, 0])

    def make_loader(self, datasets):
        names, configs = [], {}
        for name, frame_ids in datasets:
            dataset_root = self.root / name
            dataset_root.mkdir()
            gt_path = dataset_root / "gt.txt"
            np.savetxt(gt_path, [[0, 0, 0], [10, 0, 0], [11, 0, 0]])
            base = SimpleNamespace(gt_fname=str(gt_path), metric_grid_cell=0.0625)
            dataset = SimpleNamespace(base=base, world_gt=dict.fromkeys(frame_ids))
            names.append(str(dataset_root))
            configs[str(dataset_root)] = {"base": dataset}
        return SimpleNamespace(dataset=SimpleNamespace(dataset_list=names, dicts=configs))

    def test_pipeline_passes_test_frames_and_mvft_two_metre_threshold(self):
        loader = self.make_loader([("MVFT", [10, 11])])
        # 24 MVFT grid cells = 1.5 m; one other test frame is completely missed.
        predictions = torch.tensor([[10, 24, 0, 0.9, 0]])
        result = evaluate_detection_results(str(self.root), 1, loader, predictions)
        np.testing.assert_allclose(result, [50, 25, 100, 50])
        self.assertEqual(list(self.root.glob("_mod_eval_*.txt")), [])

    def test_pipeline_keeps_other_dataset_thresholds(self):
        loader = self.make_loader([("CityTrack", [10, 11]), ("MVCrowdTrack", [10, 11]),
                                   ("Wildtrack", [10, 11]), ("MultiviewX", [10, 11])])
        predictions = torch.tensor([[10, 10, 0, 0.9, index] for index in range(4)])
        result = evaluate_detection_results(str(self.root), 1, loader, predictions)
        np.testing.assert_allclose(result, [50, 50, 100, 50])

    def test_pipeline_counts_sequence_with_no_predictions(self):
        loader = self.make_loader([("CityTrack", [10, 11]), ("MVCrowdTrack", [10, 11])])
        predictions = torch.tensor([[10, 0, 0, 0.9, 0], [11, 0, 0, 0.9, 0]])
        result = evaluate_detection_results(str(self.root), 1, loader, predictions)
        np.testing.assert_allclose(result, [50, 50, 50, 50])

    def test_pipeline_accepts_completely_empty_predictions(self):
        loader = self.make_loader([("MVFT", [10, 11])])
        result = evaluate_detection_results(str(self.root), 1, loader, torch.empty((0, 5)))
        np.testing.assert_array_equal(result, [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
