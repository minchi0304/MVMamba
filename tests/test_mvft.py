"""Regression tests for MVFT coordinates and shared preprocessing."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dataloader.concat_dataset import ConcatDataset
from dataloader.dataloader import GetDataset, diffmot_joint_collate
from dataloader.mvft import MVFT


class MVFTTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(Dataset="MVFT", num_cam=8, num_frames=10,
                           img_shape=[9, 16], grid_shape=[1088, 1680], grid_cell=0.0625,
                           origin=[-52.5, 34.0], region_size=[68.0, 105.0])
        self.write_config()
        (self.root / "annotations_positions").mkdir()
        for directory in ["intrinsic", "extrinsic"]:
            (self.root / "calibrations" / directory).mkdir(parents=True)
        for camera in range(8):
            name = "C{}".format(camera + 1)
            directory = self.root / "Image_subsets" / name
            directory.mkdir(parents=True)
            for frame in range(10):
                Image.new("RGB", (16, 9)).save(directory / "{:04d}.jpg".format(frame))
            intrinsic = cv2.FileStorage(str(self.root / "calibrations" / "intrinsic" /
                                           ("intr_" + name + ".xml")), cv2.FILE_STORAGE_WRITE)
            intrinsic.write("camera_matrix", np.array([[100., 0., 8.], [0., 100., 4.5], [0., 0., 1.]]))
            intrinsic.release()
            extrinsic = cv2.FileStorage(str(self.root / "calibrations" / "extrinsic" /
                                           ("extr_" + name + ".xml")), cv2.FILE_STORAGE_WRITE)
            extrinsic.write("rvec", np.zeros((3, 1)))
            extrinsic.write("tvec", np.array([[0.], [0.], [10.]]))
            extrinsic.release()
        for frame in range(10):
            people = [self.person(0, 500, 800 + frame, 0), self.person(1, 550, 900, 1),
                      self.person(2, 600, 1000, None)]
            if frame == 0:
                people.append(people[0])
            self.annotation_path(frame).write_text(json.dumps(people), encoding="utf-8")
        self.base = MVFT(self.root, prepare_gt=False)

    def write_config(self):
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")

    def annotation_path(self, frame):
        return self.root / "annotations_positions" / "{:04d}.json".format(frame)

    @staticmethod
    def person(person_id, row, col, visible_camera):
        views = [dict(viewNum=cam, xmin=-1, ymin=-1, xmax=-1, ymax=-1) for cam in range(8)]
        if visible_camera is not None:
            # Single-pixel visibility is retained, matching MVFT annotations.
            views[visible_camera].update(xmin=3, ymin=4, xmax=3, ymax=4)
        return dict(personID=person_id, positionID=row * 1680 + col, views=views)

    def test_metadata_and_units(self):
        self.assertEqual(self.base.world_grid_shape, [1088, 1680])
        self.assertEqual(self.base.region_size, [68., 105.])
        self.assertEqual(self.base.metric_grid_cell, 0.0625)
        self.assertEqual(self.base.indexing, "ij")
        self.assertEqual(self.base.extrinsic_matrices[0][2, 3], 10.)
        self.assertFalse((self.root / "gt.txt").exists())

    def test_position_roundtrip(self):
        positions = [0, 1679, 1680, 1088 * 1680 - 1]
        positions.extend(np.random.default_rng(7).integers(0, 1088 * 1680, 1000))
        for position in positions:
            grid = self.base.get_worldgrid_from_pos(position)
            self.assertEqual(self.base.get_pos_from_worldgrid(grid), position)
            world = self.base.get_worldcoord_from_worldgrid(grid)
            np.testing.assert_array_equal(self.base.get_worldgrid_from_worldcoord(world), grid)

    def test_world_axes_and_cell_centres(self):
        world = self.base.get_worldcoord_from_worldgrid([0, 0])
        np.testing.assert_allclose(world, [-52.46875, 33.96875])
        np.testing.assert_allclose(self.base.get_worldcoord_from_worldgrid([0, 1]) - world, [0.0625, 0])
        np.testing.assert_allclose(self.base.get_worldcoord_from_worldgrid([1, 0]) - world, [0, -0.0625])
        np.testing.assert_array_equal(self.base.get_worldgrid_from_worldcoord([52.5, -34]), [1087, 1679])

    def test_invalid_positions(self):
        for position in [-1, 1088 * 1680]:
            with self.assertRaises(ValueError):
                self.base.get_worldgrid_from_pos(position)
        with self.assertRaises(ValueError):
            self.base.get_pos_from_worldgrid([1088, 0])
        with self.assertRaises(ValueError):
            self.base.get_worldgrid_from_worldcoord([53., 0])

    def test_stale_frame_count_uses_annotations(self):
        self.config["num_frames"] = 3
        self.write_config()
        with self.assertWarnsRegex(RuntimeWarning, "using the actual annotation count"):
            base = MVFT(self.root, prepare_gt=False)
        self.assertEqual(base.num_frames, 10)
        self.assertEqual(json.loads((self.root / "config.json").read_text())["num_frames"], 3)

    def test_missing_annotation_rejected(self):
        self.annotation_path(3).unlink()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            MVFT(self.root, prepare_gt=False)

    def test_mismatched_grid_rejected(self):
        self.config["grid_shape"] = [1088, 1600]
        self.write_config()
        with self.assertRaisesRegex(ValueError, "grid_shape"):
            MVFT(self.root, prepare_gt=False)

    def test_missing_config_rejected(self):
        (self.root / "config.json").unlink()
        with self.assertRaises(FileNotFoundError):
            MVFT(self.root, prepare_gt=False)

    def test_missing_setting_rejected(self):
        self.config.pop("origin")
        self.write_config()
        with self.assertRaisesRegex(ValueError, "missing fields"):
            MVFT(self.root, prepare_gt=False)

    def test_config_is_the_only_setting_source(self):
        # Leftover reports must not override config.json or be required.
        (self.root / "bev_grid.json").write_text("not a valid JSON report", encoding="utf-8")
        self.config["origin"] = [100.0, 200.0]
        self.write_config()
        base = MVFT(self.root, prepare_gt=False)
        self.assertEqual(base.origin, [100.0, 200.0])
        self.assertEqual([base.min_x, base.max_x, base.min_y, base.max_y], [100.0, 205.0, 132.0, 200.0])
        np.testing.assert_allclose(base.get_worldcoord_from_worldgrid([0, 0]), [100.03125, 199.96875])

    def test_camera_frame_completeness(self):
        paths = self.base.get_image_paths(range(10))
        self.assertTrue(all(len(camera) == 10 for camera in paths.values()))
        (self.root / "Image_subsets" / "C8" / "0004.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "C8"):
            self.base.get_image_paths(range(10))

    def test_temporal_split(self):
        self.assertEqual(list(self.base.get_frame_range(True)), list(range(9)))
        self.assertEqual(list(self.base.get_frame_range(False)), [9])
        self.assertEqual(list(self.base.get_frame_range(True, train_ratio=0.8)), list(range(8)))
        self.assertEqual(list(self.base.get_frame_range(False, train_ratio=0.8)), [8, 9])

    def test_detection_gt_visibility_duplicates_and_single_pixel(self):
        self.base.prepare_gt()
        gt = np.loadtxt(self.base.gt_fname, dtype=int)
        self.assertEqual(gt.shape, (20, 3))
        np.testing.assert_array_equal(gt[0], [0, 500, 800])
        np.testing.assert_array_equal(gt[-1], [9, 550, 900])

    def test_projection_in_both_shared_paths(self):
        dataset = GetDataset(self.base, train=True, data_aug=False)
        concat = ConcatDataset(dataset)
        world = self.base.get_worldcoord_from_worldgrid([500, 800])
        camera_plane = self.base.intrinsic_matrices[0] @ np.delete(self.base.extrinsic_matrices[0], 2, 1)
        uv = camera_plane @ np.r_[world, 1.]
        uv /= uv[2]
        projected = dataset.get_imgcoord2worldgrid_matrix(0) @ uv
        projected /= projected[2]
        np.testing.assert_allclose(projected[:2], [800, 500], atol=1e-5)
        feature_uv = uv.copy()
        feature_uv[:2] *= 1280. / 16. / dataset.img_reduce
        dynamic = dataset.projection_matrix(0, (1280, 720), (0, 0, 1280, 720), (16, 9), np.eye(3))
        static = concat.dicts[self.base.root]["proj_mats"][0].numpy()
        for matrix in [dynamic, static]:
            point = matrix @ feature_uv
            point /= point[2]
            np.testing.assert_allclose(point[:2], [200, 125], atol=1e-5)

    def test_heatmaps_offsets_ids_and_diffusion_motion(self):
        dataset = GetDataset(self.base, train=True, data_aug=False, build_diffmot_motion=True)
        self.assertEqual(dataset.reducedgrid_shape, [272, 420])
        self.assertEqual(dataset.max_objects, 2)
        centres, mask, ids, offsets = dataset.get_bev_gt(np.array([[803., 501.]]), [0])
        self.assertEqual(tuple(centres.shape), (1, 272, 420))
        self.assertTrue(mask[0, 125, 200])
        self.assertEqual(ids[0, 125, 200], 0)
        torch.testing.assert_close(offsets[:, 125, 200], torch.tensor([0.75, 0.25]))
        condition, target = dataset.diffmot_motion_samples[6][0]
        self.assertEqual(condition.shape, (5, 4))
        np.testing.assert_array_equal(target, [1., 0.])
        self.assertFalse(any(frame >= 9 for frame in dataset.diffmot_motion_samples))

    def test_full_sample_and_training_collation(self):
        dataset = GetDataset(self.base, train=True, data_aug=False, build_diffmot_motion=True)
        item, target = dataset[6]
        self.assertEqual(tuple(item["img"].shape), (8, 3, 720, 1280))
        self.assertEqual(tuple(item["proj_mats"].shape), (8, 3, 3))
        self.assertEqual(int(item["grid_gt_count"]), 2)
        np.testing.assert_array_equal(item["grid_gt"][0].numpy(), [806, 500, 0])
        self.assertEqual(tuple(target["pid_img"].shape), (8, 1, 180, 320))
        self.assertEqual(int(target["valid_img"].sum()), 2)
        self.assertEqual(tuple(item["motion_condition"].shape), (2, 5, 4))
        collated, _ = diffmot_joint_collate([(item, target)])
        self.assertEqual(tuple(collated["motion_bev_grid_shape"].shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
