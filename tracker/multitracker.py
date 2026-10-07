from collections import deque
from typing import List

import numpy as np
import torch

from . import matching
from .kalman_filter import KalmanFilter

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    shared_kalman = KalmanFilter()
    _count = 0

    def __init__(
        self,
        xy,
        score,
        buffer_size=30,
    ):
        self._xy = np.asarray(xy, dtype=np.float32).reshape(2)
        self.score = float(score)
        self.buffer_size = buffer_size
        self.tracklet_len = 0
        self.track_id = 0
        self.mean = None
        self.covariance = None
        self.kalman_filter = None
        self.state = TrackState.New
        self.is_activated = False
        self.start_frame = 0
        self.frame_id = 0
        self.conds = deque([], maxlen=5)

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    @property
    def xy(self):
        if self.mean is None:
            return self._xy.copy()
        return self.mean[:2].astype(np.float32)

    @property
    def end_frame(self):
        return self.frame_id

    def to_xy(self):
        return self.xy

    def activate(self, kalman_filter, frame_id, motion_model="kalman"):
        self.track_id = self.next_id()
        if motion_model == "kalman":
            self.kalman_filter = kalman_filter
            self.mean, self.covariance = self.kalman_filter.initiate(
                self.to_xy(),
            )
        else:
            self.conds.append(np.concatenate((self._xy.copy(), np.zeros(2, dtype=np.float32))))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, motion_model="kalman"):
        if motion_model == "kalman":
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean,
                self.covariance,
                new_track.to_xy(),
            )
        else:
            self._update_diffusion_history(new_track.to_xy())
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score

        if new_id:
            self.track_id = self.next_id()

    def predict(self):
        if self.mean is not None:
            self.mean, self.covariance = self.kalman_filter.predict(
                self.mean.copy(),
                self.covariance,
            )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) == 0:
            return

        multi_mean = np.asarray([track.mean.copy() for track in stracks], dtype=np.float32)
        multi_covariance = np.asarray([track.covariance for track in stracks], dtype=np.float32)

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean,
            multi_covariance,
        )

        for idx, (mean, covariance) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[idx].mean = mean
            stracks[idx].covariance = covariance

    def update(self, new_track, frame_id, motion_model="kalman"):
        self.frame_id = frame_id
        self.tracklet_len += 1

        if motion_model == "kalman":
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean,
                self.covariance,
                new_track.to_xy(),
            )
        else:
            self._update_diffusion_history(new_track.to_xy())

        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True

    def _update_diffusion_history(self, observed_xy):
        observed_xy = np.asarray(observed_xy, dtype=np.float32).reshape(2)
        if self.conds:
            if len(self.conds) > 1:
                previous_xy = np.asarray(self.conds[-2][:2], dtype=np.float32)
            else:
                previous_xy = np.asarray(self.conds[-1][:2], dtype=np.float32)
            condition = np.concatenate((observed_xy, observed_xy - previous_xy))
            self.conds[-1] = condition
        else:
            self.conds.append(np.concatenate((observed_xy, np.zeros(2, dtype=np.float32))))
        self._xy = observed_xy

    @staticmethod
    def multi_predict_diff(stracks, model, bev_grid_shape):
        if len(stracks) == 0:
            return

        conditions = [track.conds for track in stracks]
        with torch.no_grad():
            motion = model.generate(
                conditions,
                sample=1,
                bestof=True,
                bev_grid_shape=bev_grid_shape,
            ).mean(0)

        for track, delta in zip(stracks, motion):
            previous_xy = track.xy
            predicted_xy = previous_xy + np.asarray(delta, dtype=np.float32)
            track._xy = predicted_xy
            track.conds.append(np.concatenate((predicted_xy, predicted_xy - previous_xy)))

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class Tracker:
    """MVTrackTrans-style BEV association using current detections."""

    def __init__(
        self,
        high_cls_thres=0.4,
        track_buffer=5,
        match_thresh=75.0,
        unconfirmed_match_thresh=100.0,
        duplicate_thresh=6.0,
        motion_model="kalman",
        motion_normalization="bev",
        bev_grid_shape=None,
        device=None,
        diffusion_motion_predictor=None,
    ):
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

        self.det_thresh = float(high_cls_thres)
        self.low_det_thresh = max(0.0, self.det_thresh - 0.1)
        self.max_time_lost = int(track_buffer)

        self.match_thresh = float(match_thresh)
        self.unconfirmed_match_thresh = float(unconfirmed_match_thresh)
        self.duplicate_thresh = float(duplicate_thresh)

        if motion_model not in ("kalman", "diffusion"):
            raise ValueError("motion_model must be 'kalman' or 'diffusion'")
        self.motion_model = motion_model
        self.bev_grid_shape = bev_grid_shape

        self.kalman_filter = KalmanFilter() if motion_model == "kalman" else None
        self.diffusion_motion_predictor = None
        if motion_model == "diffusion":
            if motion_normalization == "bev" and bev_grid_shape is None:
                raise ValueError("bev_grid_shape is required for BEV-normalized diffusion motion")
            if diffusion_motion_predictor is None:
                raise ValueError("a trained diffusion motion predictor is required")
            self.diffusion_motion_predictor = diffusion_motion_predictor
            self.diffusion_motion_predictor.to(device or "cpu")
            self.diffusion_motion_predictor.eval()

    def _make_detections(self, dets, scores):
        return [
            STrack(xy, score, self.max_time_lost)
            for xy, score in zip(dets, scores)
        ]

    @staticmethod
    def _center_distance(tracks, detections):
        return matching.center_distance(
            [track.xy for track in tracks],
            [det.xy for det in detections],
        )

    def _linear_association(self, tracks, detections, thresh):
        dists = self._center_distance(tracks, detections)
        return matching.linear_assignment(dists, thresh=thresh)

    def update(
        self,
        dets,
        scores=None,
        frame_id=None,
        **_,
    ):
        self.frame_id += 1

        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 2)

        if scores is None:
            scores = np.ones((len(dets),), dtype=np.float32)

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if len(dets) != len(scores):
            raise ValueError("dets and scores must have the same length")

        candidate_indices = np.where(scores > self.low_det_thresh)[0]

        detections = self._make_detections(
            dets[candidate_indices],
            scores[candidate_indices],
        )

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        unconfirmed = []
        tracked_stracks = []

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = _joint_stracks(tracked_stracks, self.lost_stracks)
        if self.motion_model == "kalman":
            STrack.multi_predict(strack_pool)
        else:
            STrack.multi_predict_diff(
                strack_pool,
                self.diffusion_motion_predictor,
                self.bev_grid_shape,
            )

        matches_idx, unmatched_tracks_idx, unmatched_detections_idx = self._linear_association(
            strack_pool, detections, self.match_thresh
        )
        matches = [(strack_pool[int(i)], detections[int(j)]) for i, j in matches_idx]
        unmatched_tracks = [strack_pool[int(i)] for i in unmatched_tracks_idx]
        detections = [detections[int(i)] for i in unmatched_detections_idx]

        for track, det in matches:
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, motion_model=self.motion_model)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, motion_model=self.motion_model)
                refind_stracks.append(track)

        for track in unmatched_tracks:
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        matches_idx, unmatched_unconfirmed_idx, unmatched_detections_idx = self._linear_association(
            unconfirmed, detections, self.unconfirmed_match_thresh
        )
        matches = [(unconfirmed[int(i)], detections[int(j)]) for i, j in matches_idx]
        unmatched_unconfirmed = [unconfirmed[int(i)] for i in unmatched_unconfirmed_idx]
        detections = [detections[int(i)] for i in unmatched_detections_idx]

        for track, det in matches:
            track.update(det, self.frame_id, motion_model=self.motion_model)
            activated_stracks.append(track)

        for track in unmatched_unconfirmed:
            track.mark_removed()
            removed_stracks.append(track)

        for track in detections:
            if track.score < self.det_thresh:
                continue
            track.activate(
                self.kalman_filter,
                self.frame_id,
                motion_model=self.motion_model,
            )
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [
            track for track in self.tracked_stracks
            if track.state == TrackState.Tracked
        ]

        self.tracked_stracks = _joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = _sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = _sub_stracks(self.lost_stracks, self.removed_stracks)
        self.lost_stracks = _sub_stracks(self.lost_stracks, removed_stracks)

        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = _remove_duplicate_stracks(
            self.tracked_stracks,
            self.lost_stracks,
            self.duplicate_thresh,
        )

        return [track for track in self.tracked_stracks if track.is_activated]


def _joint_stracks(tlista, tlistb):
    exists = {}
    res = []

    for track in tlista:
        exists[track.track_id] = True
        res.append(track)

    for track in tlistb:
        if not exists.get(track.track_id, False):
            exists[track.track_id] = True
            res.append(track)

    return res


def _sub_stracks(tlista, tlistb):
    tracks = {track.track_id: track for track in tlista}

    for track in tlistb:
        tracks.pop(track.track_id, None)

    return list(tracks.values())


def _remove_duplicate_stracks(stracksa, stracksb, thresh):
    pdist = matching.center_distance(
        [track.xy for track in stracksa],
        [track.xy for track in stracksb],
    )

    pairs = np.where(pdist < thresh)

    dupa, dupb = [], []

    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame

        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [
        track for idx, track in enumerate(stracksa)
        if idx not in dupa
    ]

    resb = [
        track for idx, track in enumerate(stracksb)
        if idx not in dupb
    ]

    return resa, resb
