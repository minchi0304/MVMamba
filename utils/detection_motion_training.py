"""Build detector/motion histories for joint training with GT future positions."""

import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import utils


def match_detections_to_gt(record, metric_scale, match_threshold, score_threshold):
    """One-to-one identities for observations, without substituting GT positions."""
    gt_xy = np.asarray(record['gt_xy'], dtype=np.float64).reshape(-1, 2)
    gt_ids = np.asarray(record['gt_ids'], dtype=np.int64)
    xy = np.asarray(record['xy'], dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(record['scores'])
    uncertainty = np.asarray(record['uncertainty'], dtype=np.float32)
    keep = scores > score_threshold
    xy, uncertainty = xy[keep], uncertainty[keep]
    if not len(gt_xy) or not len(xy):
        return {}
    distances = np.linalg.norm(gt_xy[:, None] - xy[None], axis=-1) * metric_scale
    rows, cols = linear_sum_assignment(np.where(distances <= match_threshold, distances, 1e9))
    return {
        int(gt_ids[i]): (xy[j], uncertainty[j])
        for i, j in zip(rows, cols) if distances[i, j] <= match_threshold
    }


class DetectionMotionDataset(Dataset):
    """Cached observations; optionally roll out the current Mamba over misses."""

    def __init__(self, sequences, history_len, high_cls_thres, match_threshold=None, missing='skip'):
        conditions, targets, shapes, sample_keys = [], [], [], []
        rollout_observations = []
        self.missing = missing
        self.summary = []
        for sequence_index, sequence in enumerate(sequences):
            records = sorted(sequence['records'], key=lambda row: row['frame'])
            frames = np.asarray([row['frame'] for row in records], dtype=np.int64)
            steps = np.diff(frames)
            step = int(steps[steps > 0].min()) if np.any(steps > 0) else 1
            threshold = match_threshold
            if threshold is None:
                threshold = 2.0 if sequence['dataset_name'].lower() in ('citytrack', 'mvcrowdtrack') else 1.0
            # Use the same candidate confidence floor as tracking history updates.
            matched = [match_detections_to_gt(
                row, sequence['metric_scale'], threshold, max(0.0, high_cls_thres - 0.1),
            ) for row in records]
            count_before = len(targets)
            skipped_history = 0
            skipped_time_gap = 0
            motion_tokens = 0
            for end in range(history_len + 1, len(records)):
                start = end - history_len - 1
                if np.any(np.diff(frames[start:end + 1]) != step):
                    skipped_time_gap += 1
                    continue
                # The target frame needs GT, but does not need a detection.
                for gt_id, gt_next in zip(records[end]['gt_ids'], records[end]['gt_xy']):
                    observations = [row.get(int(gt_id)) for row in matched[start:end]]
                    detected = [index for index, value in enumerate(observations) if value is not None]
                    if not detected or (missing == 'skip' and len(detected) != len(observations)):
                        skipped_history += 1
                        continue
                    if missing == 'motion':
                        tokens = np.zeros((history_len + 1, 11), dtype=np.float32)
                        # Before the first observation, use the existing repeated-first-state padding.
                        first = detected[0]
                        for index, observation in enumerate(observations):
                            if index < first:
                                observation = observations[first]
                            if observation is None:
                                tokens[index, 9] = 1.0  # Filled by current model at training time.
                                motion_tokens += 1
                            else:
                                tokens[index, :2], tokens[index, 4] = observation
                                tokens[index, 8] = 1.0
                        rollout_observations.append(tokens)
                        targets.append(np.asarray(gt_next, dtype=np.float32))
                    else:
                        positions = np.stack([observation[0] for observation in observations])
                        condition = np.zeros((history_len, 11), dtype=np.float32)
                        condition[:, :2] = positions[1:]
                        condition[:, 2:4] = np.diff(positions, axis=0)
                        condition[:, 4] = [observation[1] for observation in observations[1:]]
                        condition[:, 8] = 1.0
                        conditions.append(condition)
                        targets.append(np.asarray(gt_next, dtype=np.float32) - positions[-1])
                    shapes.append(sequence['bev_grid_shape'])
                    sample_keys.append((sequence_index, int(frames[end]), int(gt_id)))
            self.summary.append({
                'root': sequence['root'], 'frames': len(records),
                'matched_observations': sum(map(len, matched)),
                'samples': len(targets) - count_before,
                'skipped_missing_history': skipped_history,
                'skipped_time_gap_windows': skipped_time_gap,
                'motion_tokens': motion_tokens,
                'match_threshold_m': threshold,
            })
        self.conditions = torch.from_numpy(np.asarray(conditions, dtype=np.float32).reshape(-1, history_len, 11))
        self.observations = torch.from_numpy(np.asarray(rollout_observations, dtype=np.float32).reshape(-1, history_len + 1, 11))
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32).reshape(-1, 2))
        self.grid_shapes = torch.from_numpy(np.asarray(shapes, dtype=np.float32).reshape(-1, 2))
        self.sample_keys = sample_keys
        self.indices_by_frame = defaultdict(list)
        for index, (sequence_index, frame, _) in enumerate(sample_keys):
            self.indices_by_frame[(sequences[sequence_index]['root'], frame)].append(index)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.missing == 'motion':
            return {
                'motion_observations': self.observations[index],
                'motion_target_position': self.targets[index],
                'motion_bev_grid_shape': self.grid_shapes[index],
            }
        return {
            'motion_condition': self.conditions[index],
            'motion_target': self.targets[index],
            'motion_bev_grid_shape': self.grid_shapes[index],
        }

    def batch_for_frames(self, roots, frames):
        """Attach histories to shuffled image batches using their target frame."""
        indices = [index for root, frame in zip(roots, frames)
                   for index in self.indices_by_frame.get((root, int(frame)), ())]
        return self[torch.tensor(indices, dtype=torch.long)]


@torch.no_grad()
def rollout_detection_history(predictor, observations, grid_shapes):
    """Insert detached model predictions at missing times, in original grid units."""
    was_training = predictor.training
    predictor.eval()
    history = observations[:, :1].clone()
    for index in range(1, observations.size(1)):
        token = observations[:, index].clone()
        missing = token[:, 9].bool()
        if missing.any():
            shape = grid_shapes[missing]
            mean, covariance, kappa, _, _ = predictor.predict_normalized_distribution(history[missing], shape)
            delta = predictor._denormalize_delta(mean, shape)
            token[missing, :2] = history[missing, -1, :2] + delta
            if covariance is not None:
                covariance = predictor._denormalize_covariance(covariance, shape)
                token[missing, 5] = covariance[:, 0, 0]
                token[missing, 6] = covariance[:, 1, 1]
                token[missing, 7] = covariance[:, 0, 1]
                token[missing, 10] = kappa[:, 0]
        token[:, 2:4] = token[:, :2] - history[:, -1, :2]
        history = torch.cat((history, token.unsqueeze(1)), dim=1)
    predictor.train(was_training)
    return history[:, 1:]


def motion_inputs(model, item, device):
    shape = item['motion_bev_grid_shape'].to(device)
    if 'motion_observations' in item:
        history = rollout_detection_history(model.motion_predictor, item['motion_observations'].to(device), shape)
        target = item['motion_target_position'].to(device) - history[:, -1, :2]
    else:
        history = item['motion_condition'].to(device)
        target = item['motion_target'].to(device)
    return history, target, shape


def _cache_signature(dataset, args, model):
    checkpoint_hash = hashlib.sha256()
    with open(args.train_from, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checkpoint_hash.update(chunk)
    detector_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('motion_predictor.') or name in ('instance_weight', 'motion_weight'):
            continue
        detector_hash.update(name.encode('utf-8'))
        detector_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    sequences = []
    for sub in dataset.datasets:
        gt_hash = hashlib.sha256()
        for frame in sorted(sub.world_gt):
            points, ids = sub.world_gt[frame]
            gt_hash.update(np.asarray([frame], dtype=np.int64).tobytes())
            gt_hash.update(np.asarray(points, dtype=np.float32).tobytes())
            gt_hash.update(np.asarray(ids, dtype=np.int64).tobytes())
        # Includes the train frame range; test-cache reuse cannot pass this check.
        sequences.append({
            'root': os.path.abspath(sub.root), 'gt_sha256': gt_hash.hexdigest(),
            'frames': sorted(map(int, sub.world_gt)),
            'grid_shape': list(map(int, sub.world_grid_shape)),
            'grid_reduce': sub.grid_reduce, 'img_reduce': sub.img_reduce,
            'num_cam': sub.num_cam, 'img_shape': list(map(int, sub.img_shape)),
            'projection': dataset.dicts[sub.root]['proj_mats'].tolist(),
        })
    return {
        'version': 3, 'checkpoint_sha256': checkpoint_hash.hexdigest(),
        'detector_sha256': detector_hash.hexdigest(),
        'view_pool': args.view_pool,
        'topk': utils.get_decode_topk(args), 'augmentation': False,
        'sequences': sequences,
    }


def prepare_detection_motion_dataset(model, dataset, args, device, logdir, epoch=1):
    if any(not sub.train for sub in dataset.datasets):
        raise ValueError('Detection-history training requires training splits, not test splits')
    for sub in dataset.datasets:
        sub.data_aug = False
    signature = _cache_signature(dataset, args, model)
    # An external cache may initialize epoch 1 only. Later epochs use the
    # current detector and overwrite the run-local snapshot, not the source cache.
    external_cache = args.motion_detection_cache if epoch == 1 else None
    cache_path = external_cache or os.path.join(logdir, 'motion_detections_train.pt')
    sequences = None
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location='cpu')
        if cache['signature'] == signature:
            sequences = cache['sequences']
            print('Loaded training detections for epoch {}: {}'.format(epoch, cache_path))
        elif external_cache:
            raise ValueError('Detection cache does not match the checkpoint, training split, or decoder settings; use a new cache path')
    if sequences is None:
        sequences = []
        by_root = {}
        for sub in dataset.datasets:
            sequence = {
                'root': sub.root, 'dataset_name': sub.dataset_name,
                'bev_grid_shape': list(sub.world_grid_shape),
                'metric_scale': float(getattr(sub.base, 'metric_grid_cell', float(sub.base.grid_cell) / 100.0)),
                'records': [],
            }
            sequences.append(sequence)
            by_root[sub.root] = (sequence, sub)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model.eval()
        for index, (item, _) in enumerate(tqdm(loader, desc='Epoch {} training detections'.format(epoch))):
            decoded = utils.decode_frame(model, dataset, item, index, device, args)
            root, frame, _ = utils.get_frame_metadata(loader, item)
            sequence, sub = by_root[root]
            gt_xy, gt_ids = sub.world_gt[frame]
            sequence['records'].append({
                'frame': frame,
                # Tracker x/y order, in original grid units, with no evaluator-axis swap.
                'xy': decoded['xy_all'].detach().cpu().numpy() * float(sub.grid_reduce),
                'scores': decoded['scores_all'].detach().cpu().numpy().reshape(-1),
                'uncertainty': decoded['uncertainty_all'].detach().cpu().numpy().reshape(-1),
                'gt_xy': np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2),
                'gt_ids': np.asarray(gt_ids, dtype=np.int64),
            })
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save({'signature': signature, 'sequences': sequences}, cache_path)
        print('Saved current training detections: {}'.format(cache_path))
    motion_dataset = DetectionMotionDataset(
        sequences, args.motion_history_len, args.high_cls_thres, args.motion_detection_match_thresh,
        missing=getattr(args, 'motion_history_missing', 'motion'),
    )
    for summary in motion_dataset.summary:
        print('Detection history: {}'.format(json.dumps(summary)))
    with open(os.path.join(logdir, 'motion_training_samples_epoch_{}.json'.format(epoch)), 'w') as output:
        json.dump({
            'epoch': epoch,
            'history_source': 'detection', 'history_len': args.motion_history_len,
            'missing_history': getattr(args, 'motion_history_missing', 'motion'),
            'cache': os.path.abspath(cache_path), 'checkpoint': os.path.abspath(args.train_from),
            'checkpoint_sha256': signature['checkpoint_sha256'],
            'detector_sha256': signature['detector_sha256'],
            'high_cls_thres': args.high_cls_thres, 'sequences': motion_dataset.summary,
        }, output, indent=2)
    if not len(motion_dataset):
        raise ValueError('No motion training histories: check the checkpoint, detection threshold, GT matching distance, and history length')
    return motion_dataset
