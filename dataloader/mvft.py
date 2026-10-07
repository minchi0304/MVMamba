"""MVFT adapter for the existing BEV detection and tracking pipeline."""

import json
import os
import warnings
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from torchvision.datasets import VisionDataset

from dataloader.realworld_json import JsonPositionDataset, _xml_vector


class MVFT(JsonPositionDataset):
    """Read MVFT without converting or changing its source annotations.

    Native grid coordinates are [row, col], positionID = row * width + col.
    The shared loader swaps these to [col, row] for heatmaps and projection.
    Calibration world coordinates and grid_cell are both in metres.
    Grid-to-world conversion represents each quantized cell by its centre.
    All dataset settings use the Wildtrack-style config.json fields. Its origin
    is the top-left metric boundary [min_x, max_y], and region_size is [H, W].
    """

    dataset_name = "MVFT"
    num_cameras = 8
    calibration_camera_names = ["C{}".format(i) for i in range(1, 9)]
    default_grid_reduce = 4
    default_img_reduce = 4
    indexing = "ij"
    use_projection_permutation = True

    def __init__(self, root, prepare_gt=True):
        VisionDataset.__init__(self, os.fspath(root))
        self.root = os.fspath(root)
        self.__name__ = self.dataset_name
        self.gt_fname = os.path.join(self.root, "gt.txt")
        config_path = os.path.join(self.root, "config.json")
        config = self._read_json(config_path)
        required_fields = ("Dataset", "num_cam", "num_frames", "img_shape",
                           "grid_shape", "grid_cell", "origin", "region_size")
        missing_fields = [name for name in required_fields if name not in config]
        if missing_fields:
            raise ValueError("config.json is missing fields: {}".format(missing_fields))
        if config["Dataset"] != self.dataset_name:
            raise ValueError("config.json Dataset must be MVFT")
        self.num_cam = int(config["num_cam"])
        if self.num_cam != self.num_cameras:
            raise ValueError("MVFT requires eight cameras, C1 through C8")
        self.img_shape = [int(value) for value in config["img_shape"]]
        if len(self.img_shape) != 2 or min(self.img_shape) <= 0:
            raise ValueError("img_shape must contain positive [height, width]")

        self.grid_cell = float(config["grid_cell"])
        self.metric_grid_cell = self.grid_cell
        if not np.isfinite(self.grid_cell) or self.grid_cell <= 0:
            raise ValueError("MVFT grid_cell must be positive and expressed in metres")
        self.origin = [float(value) for value in config["origin"]]
        self.region_size = [float(value) for value in config["region_size"]]
        if len(self.origin) != 2 or not np.isfinite(self.origin).all():
            raise ValueError("config.json origin must contain finite [min_x, max_y] in metres")
        if len(self.region_size) != 2 or not np.isfinite(self.region_size).all() or min(self.region_size) <= 0:
            raise ValueError("config.json region_size must contain positive [height, width] in metres")
        self.min_x, self.max_y = self.origin
        self.max_x = self.min_x + self.region_size[1]
        self.min_y = self.max_y - self.region_size[0]
        expected_shape = np.asarray(self.region_size) / self.grid_cell
        self.world_grid_shape = [int(value) for value in config["grid_shape"]]
        if (len(self.world_grid_shape) != 2 or min(self.world_grid_shape) <= 0
                or not np.array_equal(config["grid_shape"], self.world_grid_shape)
                or not np.allclose(expected_shape, self.world_grid_shape)):
            raise ValueError("config.json grid_shape, grid_cell and region_size do not agree")
        if any(size % self.default_grid_reduce for size in self.world_grid_shape):
            raise ValueError("MVFT grid dimensions must be divisible by grid_reduce=4")
        self.grid_position_modulus = self.world_grid_shape[1]
        self.worldgrid2worldcoord_mat = np.asarray([
            [0.0, self.grid_cell, self.min_x + 0.5 * self.grid_cell],
            [-self.grid_cell, 0.0, self.max_y - 0.5 * self.grid_cell],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self._frame_files = super()._annotation_files()
        if not self._frame_files:
            raise ValueError("MVFT has no frame JSON annotations")
        self._frame_files.sort(key=lambda item: item[0])
        frames = [frame for frame, _ in self._frame_files]
        self.num_frames = len(frames)
        if frames != list(range(self.num_frames)):
            raise ValueError("MVFT annotation frames must be unique and contiguous from zero")
        if "num_frames" in config and int(config["num_frames"]) != self.num_frames:
            warnings.warn(
                "config.json num_frames={} differs from {} annotations; using the actual annotation count".format(
                    config["num_frames"], self.num_frames), RuntimeWarning,
            )
        matrices = [self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)]
        self.intrinsic_matrices, self.distortion_coeffs, self.extrinsic_matrices = zip(*matrices)
        if prepare_gt:
            self.prepare_gt()
        print("MVFT: {} cameras, {} frames, image {}, BEV {}, cell {} m".format(
            self.num_cam, self.num_frames, self.img_shape, self.world_grid_shape, self.grid_cell))

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)

    def _annotation_files(self):
        return self._frame_files

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        name = self.calibration_camera_names[camera_i]
        intrinsic_path = os.path.join(self.root, "calibrations", "intrinsic", "intr_{}.xml".format(name))
        storage = cv2.FileStorage(intrinsic_path, cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise FileNotFoundError(intrinsic_path)
        try:
            intrinsic = storage.getNode("camera_matrix").mat()
            distortion_node = storage.getNode("distortion_coefficients")
            distortion = np.zeros((5, 1), dtype=np.float64) if distortion_node.empty() else distortion_node.mat()
        finally:
            storage.release()
        extrinsic_path = os.path.join(self.root, "calibrations", "extrinsic", "extr_{}.xml".format(name))
        calibration = ET.parse(extrinsic_path).getroot()
        rvec, tvec = _xml_vector(calibration, "rvec"), _xml_vector(calibration, "tvec")
        rotation, _ = cv2.Rodrigues(rvec.astype(np.float64))
        extrinsic = np.hstack((rotation, tvec.astype(np.float64).reshape(3, 1)))
        if intrinsic is None or intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("invalid camera_matrix in {}".format(intrinsic_path))
        if not np.isfinite(extrinsic).all():
            raise ValueError("invalid extrinsic matrix in {}".format(extrinsic_path))
        return intrinsic, distortion, extrinsic

    def get_worldgrid_from_pos(self, pos):
        pos = int(pos)
        if not 0 <= pos < self.world_grid_shape[0] * self.world_grid_shape[1]:
            raise ValueError("positionID is outside the MVFT BEV grid: {}".format(pos))
        return np.asarray(divmod(pos, self.grid_position_modulus), dtype=np.int64)

    def get_pos_from_worldgrid(self, worldgrid):
        row, col = np.asarray(worldgrid, dtype=np.int64)[:2]
        if not (0 <= row < self.world_grid_shape[0] and 0 <= col < self.world_grid_shape[1]):
            raise ValueError("worldgrid is outside the MVFT BEV grid")
        return int(row * self.grid_position_modulus + col)

    def get_worldgrid_from_worldcoord(self, worldcoord):
        x, y = np.asarray(worldcoord, dtype=np.float64)[:2]
        if not (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            raise ValueError("world coordinates are outside the MVFT pitch")
        row = int(np.floor((self.max_y - y) / self.grid_cell))
        col = int(np.floor((x - self.min_x) / self.grid_cell))
        return np.asarray([min(row, self.world_grid_shape[0] - 1),
                           min(col, self.world_grid_shape[1] - 1)], dtype=np.int64)

    def get_image_paths(self, frame_range):
        frame_set = set(int(frame) for frame in frame_range)
        paths = {cam: {} for cam in range(self.num_cam)}
        for cam, name in enumerate(self.calibration_camera_names):
            directory = os.path.join(self.root, "Image_subsets", name)
            for filename in sorted(os.listdir(directory)):
                stem, suffix = os.path.splitext(filename)
                if not stem.isdigit() or suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                frame = int(stem)
                if frame in frame_set:
                    if frame in paths[cam]:
                        raise ValueError("duplicate image frame {} in {}".format(frame, name))
                    paths[cam][frame] = os.path.join(directory, filename)
            missing = frame_set.difference(paths[cam])
            if missing:
                raise FileNotFoundError("{} is missing image frames: {}".format(name, sorted(missing)[:10]))
        return paths
