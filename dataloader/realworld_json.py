import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from torchvision.datasets import VisionDataset


def _numeric_stem(filename):
    stem = os.path.splitext(filename)[0]
    if not stem.isdigit():
        return None
    return int(stem)


def _camera_index(folder_name):
    digits = "".join(ch for ch in folder_name if ch.isdigit())
    if not digits:
        return None
    return int(digits) - 1


def _xml_vector(root, name):
    node = root.find(name)
    if node is None:
        raise ValueError("missing {} in extrinsic calibration".format(name))
    data_node = node.find("data")
    text = data_node.text if data_node is not None else node.text
    if text is None:
        raise ValueError("empty {} in extrinsic calibration".format(name))
    return np.asarray([float(value) for value in text.replace("\n", " ").split()], dtype=np.float32)


class JsonPositionDataset(VisionDataset):
    """Base adapter for MVTrackTrans-style real-world datasets.

    The dataset stores annotations as one JSON file per frame in
    annotations_positions/. The positionID value is decoded into x/y world-grid
    coordinates and exposed through the same fields used by GetDataset.
    """

    dataset_name = None
    image_shape = None
    world_grid_shape_value = None
    num_cameras = None
    num_frames_value = None
    worldgrid_to_worldcoord = None
    grid_position_modulus = None
    train_start_frame = 0
    split_frame = None
    test_stop_frame = None
    include_test_stop = False
    default_grid_reduce = 4
    default_img_reduce = 4
    indexing = "xy"
    calibration_camera_names = None

    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.gt_fname = os.path.join(self.root, "gt.txt")
        self.__name__ = self.dataset_name
        self.num_cam = int(self.num_cameras)
        if self.calibration_camera_names is None:
            raise ValueError("calibration_camera_names must be set")
        if len(self.calibration_camera_names) != self.num_cam:
            raise ValueError("calibration_camera_names must match num_cameras")
        self.num_frames = int(self.num_frames_value)
        self.img_shape = list(self.image_shape)
        self.world_grid_shape = list(self.world_grid_shape_value)
        self.grid_position_modulus = int(
            self.grid_position_modulus or self.world_grid_shape[0]
        )
        self.worldgrid2worldcoord_mat = np.asarray(
            self.worldgrid_to_worldcoord, dtype=np.float32
        )
        self.grid_cell = float(
            0.5
            * (
                abs(self.worldgrid2worldcoord_mat[0, 0])
                + abs(self.worldgrid2worldcoord_mat[1, 1])
            )
        )
        self.origin = [
            float(self.worldgrid2worldcoord_mat[0, 2]),
            float(self.worldgrid2worldcoord_mat[1, 2]),
        ]
        self.region_size = [
            self.world_grid_shape[1] * self.grid_cell / 100.0,
            self.world_grid_shape[0] * self.grid_cell / 100.0,
        ]
        matrices = [self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)]
        self.intrinsic_matrices, self.distortion_coeffs, self.extrinsic_matrices = zip(
            *matrices
        )

        print(self.root)
        print("Dataset Name : {}".format(self.dataset_name))
        print("Cameras : {}, Frames : {}".format(self.num_cam, self.num_frames))
        print("Image Shape(H,W) : {}".format(self.img_shape))
        print("Grid Shape(rows,cols) : {}".format(self.world_grid_shape))
        print("Grid Cell(in cm) : {}cm i.e {}m".format(self.grid_cell, self.grid_cell / 100.0))
        print("Grid Origin(x,y) : {}".format(self.origin))
        print("Area/Region size(in m) : {}m x {}m".format(self.region_size[0], self.region_size[1]))
        self.prepare_gt()

    def get_frame_range(self, train, train_ratio=0.9, skip_frame_ratio=1):
        split_frame = self.split_frame
        if split_frame is None:
            split_frame = int(train_ratio * self.num_frames)

        if train:
            return range(
                int(self.train_start_frame),
                int(split_frame),
                int(skip_frame_ratio),
            )

        stop_frame = self.test_stop_frame
        if stop_frame is None:
            stop_frame = self.num_frames
        if self.include_test_stop:
            stop_frame += 1
        return range(int(split_frame), int(stop_frame), int(skip_frame_ratio))

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        camera_name = self.calibration_camera_names[camera_i]
        intrinsic_file = os.path.join(
            self.root, "calibrations", "intrinsic", "intr_{}.xml".format(camera_name)
        )
        fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(intrinsic_file)
        intrinsic_matrix = fs.getNode("camera_matrix").mat()
        distortion_node = fs.getNode("distortion_coefficients")
        if distortion_node.empty():
            distortion_coeffs = np.zeros((5, 1), dtype=np.float32)
        else:
            distortion_coeffs = distortion_node.mat()
        fs.release()

        extrinsic_file = os.path.join(
            self.root, "calibrations", "extrinsic", "extr_{}.xml".format(camera_name)
        )
        if not os.path.exists(extrinsic_file):
            raise FileNotFoundError(extrinsic_file)
        root = ET.parse(extrinsic_file).getroot()
        rvec = _xml_vector(root, "rvec")
        tvec = _xml_vector(root, "tvec")

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = tvec.reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
        return intrinsic_matrix, distortion_coeffs, extrinsic_matrix

    def get_worldgrid_from_pos(self, pos):
        pos = int(pos)
        grid_y = pos % self.grid_position_modulus
        grid_x = pos // self.grid_position_modulus
        return np.asarray([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = np.asarray(worldgrid, dtype=int)[:2]
        return int(grid_x * self.grid_position_modulus + grid_y)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = np.asarray(worldgrid, dtype=np.float32)[:2]
        point = self.worldgrid2worldcoord_mat @ np.asarray([grid_x, grid_y, 1.0], dtype=np.float32)
        return point[:2]

    def get_worldcoord_from_pos(self, pos):
        return self.get_worldcoord_from_worldgrid(self.get_worldgrid_from_pos(pos))

    def get_worldgrid_from_worldcoord(self, worldcoord):
        coord_x, coord_y = np.asarray(worldcoord, dtype=np.float32)[:2]
        worldcoord2worldgrid = np.linalg.inv(self.worldgrid2worldcoord_mat)
        point = worldcoord2worldgrid @ np.asarray([coord_x, coord_y, 1.0], dtype=np.float32)
        return point[:2].astype(int)

    def get_pos_from_worldcoord(self, worldcoord):
        return self.get_pos_from_worldgrid(self.get_worldgrid_from_worldcoord(worldcoord))

    def get_image_paths(self, frame_range):
        frame_set = set(int(frame) for frame in frame_range)
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        image_root = os.path.join(self.root, "Image_subsets")
        for camera_folder in sorted(os.listdir(image_root)):
            folder_path = os.path.join(image_root, camera_folder)
            if not os.path.isdir(folder_path):
                continue
            cam = _camera_index(camera_folder)
            if cam is None or cam < 0 or cam >= self.num_cam:
                continue
            for filename in sorted(os.listdir(folder_path)):
                frame = _numeric_stem(filename)
                if frame is None or frame not in frame_set:
                    continue
                img_fpaths[cam][frame] = os.path.join(folder_path, filename)
        return img_fpaths

    def _annotation_files(self):
        annotation_root = os.path.join(self.root, "annotations_positions")
        files = []
        for filename in sorted(os.listdir(annotation_root)):
            frame = _numeric_stem(filename)
            if frame is None:
                continue
            files.append((frame, os.path.join(annotation_root, filename)))
        return files

    def _visible_in_any_camera(self, pedestrian):
        views = pedestrian.get("views", [])
        for cam in range(min(self.num_cam, len(views))):
            view = views[cam]
            if view is None:
                continue
            bbox = (
                view.get("xmin", -1),
                view.get("ymin", -1),
                view.get("xmax", -1),
                view.get("ymax", -1),
            )
            if bbox != (-1, -1, -1, -1):
                return True
        return False

    def prepare_gt(self):
        gt_rows = []
        for frame, path in self._annotation_files():
            seen_ids = set()
            with open(path, "r") as json_file:
                pedestrians = json.load(json_file)
            for pedestrian in pedestrians:
                if pedestrian is None:
                    continue
                person_id = pedestrian.get("personID")
                if person_id in seen_ids:
                    continue
                seen_ids.add(person_id)
                if not self._visible_in_any_camera(pedestrian):
                    continue
                grid_x, grid_y = self.get_worldgrid_from_pos(pedestrian["positionID"])
                gt_rows.append([frame, grid_x, grid_y])

        gt_rows = np.asarray(gt_rows, dtype=np.int64).reshape(-1, 3)
        os.makedirs(os.path.dirname(self.gt_fname), exist_ok=True)
        np.savetxt(self.gt_fname, gt_rows, fmt="%d")
