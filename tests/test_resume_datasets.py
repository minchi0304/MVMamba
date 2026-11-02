"""Resume can reach model loading with no training data installed."""

import contextlib
import io
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class ResumeDatasetTests(unittest.TestCase):
    def test_resume_constructs_only_evaluation_datasets_and_one_sequential_loader(self):
        class ReachedModel(Exception):
            pass

        class Frames:
            bev_gaussian_sigma = 2.

            def __init__(self, base, **kwargs):
                if kwargs["train"] or kwargs.get("build_motion_samples", False) or kwargs["data_aug"]:
                    raise AssertionError("Resume must never construct training data")
                self.root = base.root
                self.dataset_name = Path(base.root).name

            def __len__(self):
                return 2

        class Combined:
            def __init__(self, *datasets):
                self.datasets = list(datasets)
                self.dataset_list = [dataset.root for dataset in datasets]

            def __len__(self):
                return sum(len(dataset) for dataset in self.datasets)

        def base_dataset(path):
            if "missing_train" in path:
                raise AssertionError("Resume opened a training dataset directory")
            return SimpleNamespace(root=path)

        real_open = open
        opened_splits = []

        def read_split(path, *args, **kwargs):
            if str(path) in ("train_datapath.csv", "test_datapath.csv"):
                opened_splits.append(str(path))
                return io.StringIO("train,./missing_train,1,100\n"
                                   "test,./eval_b,0,100\n"
                                   "train,./missing_train_2,1,100\n"
                                   "test,./eval_a,0,100\n")
            return real_open(path, *args, **kwargs)

        for name in ("gmvd_train", "gmvd_test", "citytrack", "multiviewx", "wildtrack", "mvcrowdtrack"):
            with self.subTest(dataset=name):
                opened_splits.clear()
                command = ["main.py", "-d", name, "--resume", "weights.pth", "--aug",
                           "--motion_train_history", "gt", "--num_workers", "0"]
                with contextlib.redirect_stdout(io.StringIO()) as output, \
                        contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(sys, "argv", command))
                    stack.enter_context(patch("os.makedirs"))
                    stack.enter_context(patch("utils.logger.Logger", return_value=output))
                    stack.enter_context(patch("builtins.open", side_effect=read_split))
                    for module, cls in (("multiviewx", "MultiviewX"), ("wildtrack", "Wildtrack"),
                                        ("citytrack", "CityTrack"), ("mvcrowdtrack", "MVCrowdTrack")):
                        stack.enter_context(patch(f"dataloader.{module}.{cls}", side_effect=base_dataset))
                    datasets = stack.enter_context(patch("dataloader.dataloader.GetDataset", side_effect=Frames))
                    stack.enter_context(patch("dataloader.concat_dataset.ConcatDataset", side_effect=Combined))
                    loaders = stack.enter_context(patch("torch.utils.data.DataLoader", wraps=torch.utils.data.DataLoader))
                    stack.enter_context(patch("detector.encoder.resnet18", side_effect=ReachedModel))
                    with self.assertRaises(ReachedModel):
                        runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
                self.assertEqual(loaders.call_count, 1)
                self.assertFalse(loaders.call_args.kwargs["shuffle"])
                self.assertEqual(loaders.call_args.kwargs["batch_size"], 1)
                self.assertNotIn("collate_fn", loaders.call_args.kwargs)
                self.assertNotIn("Training datasets", output.getvalue())
                self.assertNotIn("Detector history cache", output.getvalue())
                self.assertIn("Resume evaluation", output.getvalue())
                if name.startswith("gmvd"):
                    self.assertEqual(datasets.call_count, 2)
                    self.assertEqual(loaders.call_args.args[0].dataset_list, ["/dataset/eval_b", "/dataset/eval_a"])
                    expected = "train_datapath.csv" if name == "gmvd_train" else "test_datapath.csv"
                    self.assertEqual(opened_splits, [expected])
                else:
                    self.assertEqual(datasets.call_count, 1)
                    self.assertEqual(opened_splits, [])


if __name__ == "__main__":
    unittest.main()
