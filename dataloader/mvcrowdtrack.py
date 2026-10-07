import numpy as np

from dataloader.realworld_json import JsonPositionDataset


class MVCrowdTrack(JsonPositionDataset):
    dataset_name = "MVCrowdTrack"
    image_shape = [2988, 5312]
    world_grid_shape_value = [800, 1200]
    num_cameras = 7
    calibration_camera_names = ["1", "2", "3", "4", "5", "6", "7"]
    num_frames_value = 4122
    grid_position_modulus = 800
    train_start_frame = 1
    split_frame = int(num_frames_value * 0.8)
    test_stop_frame = num_frames_value
    default_grid_reduce = 4
    default_img_reduce = 4
    use_projection_permutation = False
    bev_gaussian_sigma = 3
    metric_grid_cell = 0.1
    worldgrid_to_worldcoord = np.linalg.inv(
        np.asarray(
            [
                [1 / 10.0, 0.0, 0.0],
                [0.0, 1 / 10.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )


Canghai = MVCrowdTrack
