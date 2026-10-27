"""Detached detector histories, optionally cached, for joint-training targets."""

from collections import Counter, deque
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from tracker import matching
from tracker.motion_history import detection_token, motion_token


@contextmanager
def history_inference(model):
    """Do not update BN buffers or retain graphs while replaying the past."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


class DetectionHistory:
    """GT supplies identities only; all input positions come from det/motion.

    This is teacher association, not a full tracker simulation: unmatched false
    positives cannot get a supervised identity. Each instance belongs to one
    target window. Motion state is rebuilt even when detector outputs are cached.
    """

    def __init__(self, history_len, high_threshold, match_threshold, max_gap,
                 uncertainty_mode, uncertainty_gate):
        self.history_len = int(history_len)
        self.high_threshold = float(high_threshold)
        self.low_threshold = max(0., self.high_threshold - .1)
        self.match_threshold = float(match_threshold)
        self.max_gap = int(max_gap)
        self.mode = uncertainty_mode
        self.gate = uncertainty_gate
        self.width = 11 if self.gate != 'none' else (4 if self.mode == 'none' else 10)
        self.histories = {}
        self.sources = {}
        self.missed = {}

    def advance(self, xy, scores, uncertainty, gt_xy, gt_ids, predictor, grid_shape):
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        uncertainty = np.asarray(uncertainty, dtype=np.float32).reshape(-1)
        gt_xy = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)
        gt_ids = np.asarray(gt_ids).reshape(-1)
        keep = (scores > self.low_threshold) & np.isfinite(xy).all(axis=1)
        keep &= np.isfinite(scores) & np.isfinite(uncertainty)
        xy, scores, uncertainty = xy[keep], scores[keep], uncertainty[keep]

        costs = matching.center_distance(gt_xy, xy)
        # Low-confidence observations may continue an existing identity, but
        # must not steal a birth assignment or initialize an unseen identity.
        for row, pid in enumerate(gt_ids):
            if int(pid) not in self.histories:
                costs[row, scores < self.high_threshold] = np.inf
        matches, _, _ = matching.linear_assignment(costs, self.match_threshold)
        observed = set()
        for gt_index, det_index in matches:
            pid = int(gt_ids[gt_index])
            observed.add(pid)
            history = self.histories.setdefault(pid, deque(maxlen=self.history_len))
            delta = xy[det_index] - history[-1][:2] if history else np.zeros(2, dtype=np.float32)
            history.append(detection_token(
                xy[det_index], delta, uncertainty[det_index], self.mode, self.gate,
            ))
            self.sources.setdefault(pid, deque(maxlen=self.history_len)).append(True)
            self.missed[pid] = 0

        missing = []
        for pid in list(self.histories):
            if pid in observed:
                continue
            self.missed[pid] += 1
            if self.missed[pid] > self.max_gap:
                del self.histories[pid], self.sources[pid], self.missed[pid]
            else:
                missing.append(pid)
        if missing:
            histories = [np.asarray(self.histories[pid]) for pid in missing]
            # Evaluation mode/no_grad is owned by build_detection_motion_batch.
            deltas, covariances, kappas = predictor.generate_distribution(
                histories, bev_grid_shape=grid_shape, return_kappa=True,
            )
            for index, pid in enumerate(missing):
                delta = deltas[index]
                xy_next = self.histories[pid][-1][:2] + delta
                token = motion_token(
                    xy_next, delta,
                    None if covariances is None else covariances[index],
                    None if kappas is None else float(np.asarray(kappas[index]).reshape(-1)[0]),
                    self.mode, self.gate,
                )
                if not np.isfinite(token).all():
                    raise FloatingPointError("Non-finite autoregressive motion history")
                self.histories[pid].append(token)
                self.sources[pid].append(False)

    def samples(self, gt_xy, gt_ids):
        conditions, targets = [], []
        stats = Counter()
        for point, pid in zip(gt_xy, gt_ids):
            history = self.histories.get(int(pid))
            if not history:
                continue
            tokens = np.asarray(history, dtype=np.float32)
            # Count real tokens before repeated-first-state padding.
            det_mask = np.asarray(self.sources[int(pid)], dtype=bool)
            motion_mask = ~det_mask
            stats['det_tokens'] += int(det_mask.sum())
            stats['motion_tokens'] += int(motion_mask.sum())
            if self.width >= 10:
                stats['det_u_sum'] += float(tokens[det_mask, 4].sum())
                stats['motion_trace_sum'] += float(tokens[motion_mask, 5:7].sum())
            padding = np.repeat(tokens[:1], self.history_len - len(tokens), axis=0)
            padding[:, 2:4] = 0.
            conditions.append(np.concatenate((padding, tokens), axis=0))
            targets.append(np.asarray(point[:2], dtype=np.float32) - tokens[-1, :2])
        stats['motion_samples'] = len(conditions)
        return conditions, targets, stats


def _decode_history_frame(model, dataset, root, frame, device, args):
    """Return owned CPU detection arrays; caller supplies eval/no_grad context."""
    from utils.utils import decode_frame

    sub_dataset = dataset.dicts[root]['base']
    past = sub_dataset.load_history_frame(frame)
    past['img'] = past['img'].unsqueeze(0)
    past['proj_mats'] = past['proj_mats'].unsqueeze(0)
    past['root'] = [root]
    decoded = decode_frame(model, dataset, past, 0, device, args)
    xy = decoded['xy_all'].detach().cpu().numpy().astype(np.float32, copy=True)
    xy *= sub_dataset.grid_reduce
    return (
        xy,
        decoded['scores_all'].detach().cpu().numpy().astype(np.float32, copy=True),
        decoded['uncertainty_all'].detach().cpu().numpy().astype(np.float32, copy=True),
    )


def precompute_detection_cache(model, dataset, device, args, epoch=0, progress=False):
    """Decode this training dataset once with fixed epoch-start detector weights.

    All frames are loaded one at a time without augmentation. No optimizer or
    motion predictor runs here; cached arrays contain no graphs or GPU storage.
    Returning a new dict keeps the cache local to the current epoch.
    """
    frame_cache = {}
    total = sum(len(config['base'].world_gt) for config in dataset.dicts.values())
    with history_inference(model), tqdm(
        total=total, desc='Detector cache epoch {}'.format(epoch),
        unit='frame', leave=False, disable=not progress,
    ) as bar:
        for root, config in dataset.dicts.items():
            for frame in sorted(config['base'].world_gt):
                frame_cache[(root, int(frame))] = _decode_history_frame(
                    model, dataset, root, frame, device, args,
                )
                bar.update(1)
    return frame_cache


def build_detection_motion_batch(model, dataset, item, device, args, frame_cache=None):
    """Replay past frames, decoding only cache misses before the training forward.

    The caller owns the cache lifetime; train() creates a new dict each epoch.
    Entries contain only detached CPU arrays (world-grid xy, score, uncertainty),
    never images, feature maps, GT assignments, or motion predictions.
    """
    conditions, targets, grid_shapes = [], [], []
    stats = Counter()
    with history_inference(model):
        for root, target_frame in zip(item['root'], item['frame']):
            sub_dataset = dataset.dicts[root]['base']
            replay = DetectionHistory(
                args.motion_history_len, args.high_cls_thres,
                args.motion_train_match_thresh, args.track_buffer,
                args.mamba_uncertainty_mode, args.mamba_uncertainty_gate,
            )
            target_frame = int(target_frame)
            for frame in sub_dataset.history_frames(target_frame):
                # Explicit past-only check also protects future dataset adapters.
                if frame >= target_frame:
                    raise ValueError("Motion history must be strictly before the target frame")
                cache_key = (root, int(frame))
                detections = None if frame_cache is None else frame_cache.get(cache_key)
                if detections is None:
                    if getattr(args, 'motion_train_cache', 'none') == 'precompute':
                        raise KeyError('Missing precomputed detection for root {!r}, frame {}'.format(root, frame))
                    detections = _decode_history_frame(
                        model, dataset, root, frame, device, args,
                    )
                    if frame_cache is not None:
                        frame_cache[cache_key] = detections
                    stats['history_inferences'] += 1
                else:
                    stats['history_cache_hits'] += 1
                gt_xy, gt_ids = sub_dataset.world_gt[frame]
                replay.advance(
                    *detections,
                    gt_xy, gt_ids, model.motion_predictor, sub_dataset.world_grid_shape,
                )
                stats['history_frames'] += 1
            gt_xy, gt_ids = sub_dataset.world_gt[target_frame]
            sample_conditions, sample_targets, sample_stats = replay.samples(gt_xy, gt_ids)
            conditions.extend(sample_conditions)
            targets.extend(sample_targets)
            grid_shapes.extend([sub_dataset.world_grid_shape] * len(sample_conditions))
            stats.update(sample_stats)

    width = 11 if args.mamba_uncertainty_gate != 'none' else (
        4 if args.mamba_uncertainty_mode == 'none' else 10
    )
    return {
        'motion_condition': torch.as_tensor(
            np.asarray(conditions, dtype=np.float32).reshape(-1, args.motion_history_len, width),
            device=device,
        ),
        'motion_target': torch.as_tensor(
            np.asarray(targets, dtype=np.float32).reshape(-1, 2), device=device,
        ),
        'motion_bev_grid_shape': torch.as_tensor(
            np.asarray(grid_shapes, dtype=np.float32).reshape(-1, 2), device=device,
        ),
    }, stats


def format_history_stats(stats):
    return (
        'motion_samples {motion_samples}, det_tokens {det_tokens}, motion_tokens {motion_tokens}, '
        'history_frames {history_frames}, history_inferences {history_inferences}, '
        'cache_hits {history_cache_hits} ({hit_rate:.1%}), '
        'det_u {det_u:.4f}, motion_cov_trace {trace:.4f}'
    ).format(
        **{key: stats[key] for key in (
            'motion_samples', 'det_tokens', 'motion_tokens', 'history_frames',
            'history_inferences', 'history_cache_hits',
        )},
        hit_rate=stats['history_cache_hits'] / max(stats['history_frames'], 1),
        det_u=stats['det_u_sum'] / max(stats['det_tokens'], 1),
        trace=stats['motion_trace_sum'] / max(stats['motion_tokens'], 1),
    )
