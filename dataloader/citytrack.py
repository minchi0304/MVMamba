import numpy as np

from dataloader.realworld_json import JsonPositionDataset


class CityTrack(JsonPositionDataset):
    dataset_name = "CityTrack"
    image_shape = [1520, 2704]
    world_grid_shape_value = [768, 640]
    num_cameras = 3
    calibration_camera_names = ["1", "2", "3"]
    num_frames_value = 2558
    grid_position_modulus = 768
    split_frame = 1948
    test_stop_frame = 2558
    default_grid_reduce = 2
    default_img_reduce = 4
    bev_gaussian_sigma = 3.0
    # bev_gaussian_sigma = 2.5
    use_projection_permutation = False
    metric_grid_cell = 0.1
    worldgrid_to_worldcoord = np.linalg.inv(
        np.asarray(
            [
                [1 / 76.25, 0.0, 352 * 0.8],
                [0.0, 1 / 76.25, 522 * 0.8],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )


Citystreet = CityTrack
