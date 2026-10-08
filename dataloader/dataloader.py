import os
import json
from collections import defaultdict
from networkx import center
import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor
from utils import basic
from operator import itemgetter

class GetDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(),grid_reduce=4, img_reduce=4, train_ratio=0.9, sample_require=0, data_aug=True, build_motion_samples=False, motion_history_len=5, bev_gaussian_sigma=2.0):
        # parameters in super can be accessed as (self.param) eg:- self.transform, self.target_transform
        super().__init__(base.root, transform=transform, target_transform=target_transform)
        self.base = base
        self.train = bool(train)
        self.data_aug = bool(data_aug)
        self.build_motion_samples = bool(build_motion_samples)
        self.motion_history_len = int(motion_history_len)
        if self.motion_history_len < 1:
            raise ValueError("motion_history_len must be at least 1")
        self.root, self.num_cam, self.num_frames = base.root, base.num_cam, base.num_frames
        self.dataset_name = base.dataset_name
        self.img_shape, self.world_grid_shape = base.img_shape, base.world_grid_shape # H,W; N_row,N_col
        self.data_aug_conf = {'final_dim': (720, 1280), 'resize_lim': (0.8, 1.2)}
        if sample_require:
           self.skip_frame_ratio = int(self.num_frames//sample_require)
        else:
           self.skip_frame_ratio = 1
        
        # Reduce grid [480,1440] by factor of 4.
        self.grid_reduce = grid_reduce
        #[480,1440]/4 --> [120,360] i.e [1200cm,3600cm] and 10cm is pixel size, therefore [1200/10,3600/10]
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.world_grid_shape)) 

        self.img_reduce = img_reduce
        self.bev_gaussian_sigma = float(bev_gaussian_sigma)
        
        # Map kernel size = 41*41
        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        # normalizing the kernel with max value.
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)
        
        # Split train/test data
        if hasattr(self.base, 'get_frame_range'):
            frame_range = self.base.get_frame_range(
                train,
                train_ratio=train_ratio,
                skip_frame_ratio=self.skip_frame_ratio,
            )
        elif train:
            frame_range = range(0,int(train_ratio*self.num_frames), self.skip_frame_ratio)
        else:
            frame_range = range(int(train_ratio*self.num_frames), self.num_frames, self.skip_frame_ratio)
        
        ############ Cam set Selection ############
        #frame_range = range(0,self.num_frames) 
        ###########################################
        self.img_fpath = self.base.get_image_paths(frame_range)
  
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.download(frame_range)
        max_frame_objects = max((len(points) for points, _ in self.world_gt.values()), default=0)
        self.max_objects = int(max_frame_objects)
        self.motion_samples = (
            self._build_motion_samples()
            if self.train and self.build_motion_samples
            else {}
        )


    def _build_motion_samples(self):
        trajectories = defaultdict(list)
        for frame in sorted(self.world_gt):
            world_pts, world_pids = self.world_gt[frame]
            for point, person_id in zip(world_pts, world_pids):
                trajectories[int(person_id)].append(
                    (int(frame), np.asarray(point[:2], dtype=np.float32))
                )

        samples_by_frame = defaultdict(list)
        history_len = self.motion_history_len
        window_len = history_len + 2
        dataset_frames = np.asarray(sorted(self.world_gt), dtype=np.int64)
        frame_steps = np.diff(dataset_frames)
        frame_steps = frame_steps[frame_steps > 0]
        nominal_frame_step = int(frame_steps.min()) if len(frame_steps) else 1
        for trajectory in trajectories.values():
            for end_index in range(window_len - 1, len(trajectory)):
                window = trajectory[end_index - window_len + 1:end_index + 1]
                window_frames = np.asarray(
                    [entry[0] for entry in window],
                    dtype=np.int64,
                )
                if not np.all(np.diff(window_frames) == nominal_frame_step):
                    continue
                boxes = np.stack([entry[1] for entry in window], axis=0)
                deltas = boxes[1:history_len + 1] - boxes[0:history_len]
                condition = np.concatenate(
                    [boxes[1:history_len + 1], deltas],
                    axis=1,
                )
                delta_position = boxes[history_len + 1] - boxes[history_len]
                target_frame = window[-1][0]
                samples_by_frame[target_frame].append(
                    (condition, delta_position)
                )
        return dict(samples_by_frame)


    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame_stem = os.path.splitext(fname)[0]
            if not frame_stem.isdigit():
                continue
            frame = int(frame_stem)
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)

                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                seen_ids = set()
                for pedestrian in all_pedestrians:
                    if pedestrian is None:
                        continue
                    person_id = pedestrian['personID']
                    if person_id in seen_ids:
                        continue
                    seen_ids.add(person_id)

                    visible = False
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            visible = True
                            break
                    if not visible:
                        continue
                    
                    x, y = self.base.get_worldgrid_from_pos(pedestrian['positionID'])
                    
                    if person_id not in self.pid_dict:
                        self.pid_dict[person_id] = len(self.pid_dict)
                    
                    # Keep dataset GT in original world-grid coordinates. BEV heatmap
                    # targets are reduced only when they are rasterized in get_bev_gt.
                    if self.base.indexing == 'xy':
                        world_pts.append((x, y))
                    else:
                        world_pts.append((y, x))
                    
                    world_pids.append(person_id)
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]))
                            img_pids[cam].append(person_id)

                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))


    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.train and self.data_aug:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)
            crop_offset = int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:
            resize_dims = (fW, fH)
            crop = (0, 0, fW, fH)
        return resize_dims, crop

    def image_aug_matrix(self, resize_dims, crop, image_size):
        orig_w, orig_h = image_size
        sx = resize_dims[0] / float(orig_w)
        sy = resize_dims[1] / float(orig_h)
        return np.array(
            [
                [self.img_reduce / sx, 0.0, crop[0] / sx],
                [0.0, self.img_reduce / sy, crop[1] / sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def sample_bev_augmentation(self):
        bev_aug_mat = np.eye(3, dtype=np.float32)
        if self.train and self.data_aug:
            scene_center = np.zeros(2, dtype=np.float32)
            scene_center[:] = np.random.uniform(-0.25, 0.25, size=2)
            bev_aug_mat[0, 2] = -scene_center[0]
            bev_aug_mat[1, 2] = -scene_center[1]
        return bev_aug_mat

    def apply_bev_augmentation(self, worldgrid_pts, bev_aug_mat):
        if len(worldgrid_pts) == 0:
            return worldgrid_pts
        ones = np.ones((worldgrid_pts.shape[0], 1), dtype=np.float32)
        worldgrid_h = np.concatenate([worldgrid_pts[:, :2], ones], axis=1)
        return (bev_aug_mat @ worldgrid_h.T).T[:, :2]

    def get_imgcoord2worldgrid_matrix(self, cam):
        worldcoord2imgcoord_mat = self.base.intrinsic_matrices[cam] @ np.delete(
            self.base.extrinsic_matrices[cam], 2, 1
        )
        worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ self.base.worldgrid2worldcoord_mat
        imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
        if not getattr(self.base, 'use_projection_permutation', True):
            return imgcoord2worldgrid_mat
        permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        return permutation_mat @ imgcoord2worldgrid_mat

    def projection_matrix(self, cam, resize_dims, crop, image_size, bev_aug_mat):
        map_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))
        return (
            map_zoom_mat
            @ bev_aug_mat
            @ self.get_imgcoord2worldgrid_matrix(cam)
            @ self.image_aug_matrix(resize_dims, crop, image_size)
        )


    def get_bev_gt(self, world_pts, pids):
        Y, X = self.reducedgrid_shape
        center = torch.zeros((1, Y, X), dtype=torch.float32)
        valid_mask = torch.zeros((1, Y, X), dtype=torch.bool)
        offset = torch.zeros((2, Y, X), dtype=torch.float32)
        person_ids = torch.zeros((1, Y, X), dtype=torch.long)
        for pt, pid in zip(world_pts, pids):
            ct = torch.tensor(pt[:2], dtype=torch.float32) / self.grid_reduce
            ct_int = ct.int()
            if ct_int[1] >= Y or ct_int[0] >= X or ct_int[1] < 0 or ct_int[0] < 0:
                continue
            basic.draw_umich_gaussian(center[0], ct_int, sigma=self.bev_gaussian_sigma)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid
        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, H, W, sx=1.0, sy=1.0, crop=(0, 0, 0, 0)):

        orig_H, orig_W = self.img_shape
        scale_x = sx / self.img_reduce
        scale_y = sy / self.img_reduce

        # center = torch.zeros((2, H, W), dtype=torch.float32)
        # offset = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        for pt, pid in zip(img_pts, img_pids):
            x1, y1, x2, y2 = pt
            x1 = x1 * scale_x - crop[0] / self.img_reduce
            x2 = x2 * scale_x - crop[0] / self.img_reduce
            y1 = y1 * scale_y - crop[1] / self.img_reduce
            y2 = y2 * scale_y - crop[1] / self.img_reduce
            
            # ct = torch.tensor([(pt[0]+pt[2])/2, (pt[1]+pt[3])/2], dtype=torch.float32)
            ct = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2], dtype=torch.float32)
            ct_int = ct.int()
            # print(f"bbox: {pt}, center: {ct_int}, H: {H}, W: {W}")
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue
            # basic.draw_umich_gaussian(center[0], ct_int, sigma=1.5)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            # offset[:, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid
        # return center, offset, person_ids, valid_mask
        return person_ids, valid_mask

    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]
        imgs = []
        img_aug_params = []
        proj_mats = []
        bev_aug_mat = self.sample_bev_augmentation()
        for cam in range(self.num_cam):

            img = Image.open(self.img_fpath[cam][frame]).convert('RGB')
            resize_dims, crop = self.sample_augmentation()
            image_size = img.size
            sx = resize_dims[0] / float(image_size[0])
            sy = resize_dims[1] / float(image_size[1])
            img = basic.img_transform(img, resize_dims, crop)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
            img_aug_params.append((sx, sy, crop))
            proj_mats.append(torch.from_numpy(self.projection_matrix(cam, resize_dims, crop, image_size, bev_aug_mat)).float())
        imgs = torch.stack(imgs)
        proj_mats = torch.stack(proj_mats)

        # BEV GT
        worldgrid_pts, world_pids = self.world_gt[frame]
        worldgrid_pts_original = worldgrid_pts.astype(np.float32)
        worldgrid_pts = self.apply_bev_augmentation(worldgrid_pts_original, bev_aug_mat)
        center_bev, valid_bev, pid_bev, offset_bev = self.get_bev_gt(worldgrid_pts, world_pids)

        _, _, H, W = imgs.shape
        H, W = H // self.img_reduce, W // self.img_reduce
        center_imgs, offset_imgs, pid_imgs, valid_imgs = [], [], [], []
        for cam in range(self.num_cam):
            img_pts, img_pids = self.imgs_gt[frame][cam]
            if len(img_pts) == 0:
                # center_img = torch.zeros((2, H, W), dtype=torch.float32)
                # offset_img = torch.zeros((2, H, W), dtype=torch.float32)
                pid_img = torch.zeros((1, H, W), dtype=torch.long)
                valid_img = torch.zeros((1, H, W), dtype=torch.bool)
            else:
                # center_img, offset_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids, H, W)
                sx, sy, crop = img_aug_params[cam]
                pid_img, valid_img = self.get_img_gt(img_pts, img_pids, H, W, sx=sx, sy=sy, crop=crop)
            # center_imgs.append(center_img)
            # offset_imgs.append(offset_img)
            pid_imgs.append(pid_img)
            valid_imgs.append(valid_img)
        # center_imgs = torch.stack(center_imgs)
        # offset_imgs = torch.stack(offset_imgs)
        pid_imgs = torch.stack(pid_imgs)
        valid_imgs = torch.stack(valid_imgs)

        # grid_gt: (max_objects, 3) [original world-grid x, original world-grid y, pid]
        max_objects = self.max_objects
        worldgrid_pts_torch = torch.tensor(worldgrid_pts_original, dtype=torch.long)
        world_pids_torch = torch.tensor(world_pids, dtype=torch.long)
        grid_gt = torch.zeros((max_objects, 3), dtype=torch.long)
        n_obj = len(worldgrid_pts_torch)
        if n_obj > max_objects:
            raise ValueError("grid_gt capacity is smaller than frame GT count")
        if n_obj > 0:
            grid_gt[:n_obj, :2] = worldgrid_pts_torch[:n_obj]
            grid_gt[:n_obj, 2] = world_pids_torch[:n_obj]

        item = {
            'img': imgs,
            'frame': frame,
            'grid_gt': grid_gt,
            'grid_gt_count': torch.tensor(n_obj, dtype=torch.long),
            'root': self.root,
            'proj_mats': proj_mats,
        }
        if self.train and self.build_motion_samples:
            motion_samples = self.motion_samples.get(frame, [])
            if motion_samples:
                item['motion_condition'] = torch.from_numpy(
                    np.stack([sample[0] for sample in motion_samples], axis=0)
                ).float()
                item['motion_target'] = torch.from_numpy(
                    np.stack([sample[1] for sample in motion_samples], axis=0)
                ).float()
            else:
                item['motion_condition'] = torch.empty((0, 5, 4), dtype=torch.float32)
                item['motion_target'] = torch.empty((0, 2), dtype=torch.float32)
            item['motion_bev_grid_shape'] = torch.tensor(
                self.world_grid_shape,
                dtype=torch.float32,
            )

        return item, {
            'center_bev': center_bev, #gaussian heatmap gt
            'valid_bev': valid_bev, # binary center heatmap gt
            'pid_bev': pid_bev, # person ID center heatmap gt
            'offset_bev': offset_bev, # offset gt
            
            'pid_img': pid_imgs,
            'valid_img': valid_imgs,
        }


    def __len__(self):
        # length of dataset
        return len(self.world_gt.keys())


def motion_joint_collate(batch):
    items, targets = zip(*batch)
    motion_conditions = []
    motion_targets = []
    motion_grid_shapes = []
    detector_items = []

    for item in items:
        detector_item = {
            key: value
            for key, value in item.items()
            if key not in ('motion_condition', 'motion_target', 'motion_bev_grid_shape')
        }
        detector_items.append(detector_item)

        condition = item.get('motion_condition')
        if condition is None:
            continue
        if condition.size(0) == 0:
            continue
        motion_conditions.append(condition)
        motion_targets.append(item['motion_target'])
        motion_grid_shapes.append(
            item['motion_bev_grid_shape'].unsqueeze(0).repeat(condition.size(0), 1)
        )

    collated_item = default_collate(detector_items)
    collated_target = default_collate(list(targets))
    if motion_conditions:
        collated_item['motion_condition'] = torch.cat(motion_conditions, dim=0)
        collated_item['motion_target'] = torch.cat(motion_targets, dim=0)
        collated_item['motion_bev_grid_shape'] = torch.cat(motion_grid_shapes, dim=0)
    else:
        collated_item['motion_condition'] = torch.empty((0, 5, 4), dtype=torch.float32)
        collated_item['motion_target'] = torch.empty((0, 2), dtype=torch.float32)
        collated_item['motion_bev_grid_shape'] = torch.empty((0, 2), dtype=torch.float32)
    return collated_item, collated_target
