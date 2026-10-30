from collections import deque
from typing import List

import numpy as np
import torch

from . import matching
from .kalman_filter import KalmanFilter
from .motion_history import detection_token, history_token_width, motion_token

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
        motion_history_len=5,
        detection_uncertainty=0.0,
        mamba_uncertainty_mode="none",
        mamba_uncertainty_gate="none",
        implicit_motion_uncertainty_type="total",
    ):
        self._xy = np.asarray(xy, dtype=np.float32).reshape(2)
        self.score = float(score)
        self.detection_uncertainty = float(detection_uncertainty)
        self.mamba_uncertainty_mode = mamba_uncertainty_mode
        self.mamba_uncertainty_gate = mamba_uncertainty_gate
        self.implicit_motion_uncertainty_type = implicit_motion_uncertainty_type
        self.requires_motion_kappa = history_token_width(
            mamba_uncertainty_mode, mamba_uncertainty_gate, implicit_motion_uncertainty_type,
        ) == 11
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
        self.conds = deque([], maxlen=int(motion_history_len))
        self.mamba_state = None
        self.mamba_next_delta = None
        self.mamba_next_covariance = None
        self.mamba_next_kappa = None
        self.motion_delta = None
        self.motion_covariance = None
        self.motion_kappa = None
        self.motion_prediction_xy = None
        self.motion_prediction_pending = False

    def _detection_condition(self, observed_xy, delta):
        return detection_token(
            observed_xy, delta, self.detection_uncertainty,
            self.mamba_uncertainty_mode, self.mamba_uncertainty_gate,
            self.implicit_motion_uncertainty_type,
        )

    def _motion_condition(self, predicted_xy, delta):
        return motion_token(
            predicted_xy, delta, self.motion_covariance, self.motion_kappa,
            self.mamba_uncertainty_mode, self.mamba_uncertainty_gate,
            self.implicit_motion_uncertainty_type,
        )

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
            self.conds.append(self._detection_condition(
                self._xy.copy(),
                np.zeros(2, dtype=np.float32),
            ))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, motion_model="kalman"):
        self.detection_uncertainty = new_track.detection_uncertainty
        if motion_model == "kalman":
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean,
                self.covariance,
                new_track.to_xy(),
            )
        else:
            self._update_motion_history(new_track.to_xy())
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
        self.detection_uncertainty = new_track.detection_uncertainty

        if motion_model == "kalman":
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean,
                self.covariance,
                new_track.to_xy(),
            )
        else:
            self._update_motion_history(new_track.to_xy())

        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True

    def _update_motion_history(self, observed_xy):
        observed_xy = np.asarray(observed_xy, dtype=np.float32).reshape(2)
        if self.motion_prediction_pending and len(self.conds) > 1:
            previous_xy = np.asarray(self.conds[-2][:2], dtype=np.float32)
        elif self.conds:
            previous_xy = np.asarray(self.conds[-1][:2], dtype=np.float32)
        else:
            previous_xy = observed_xy
        condition = self._detection_condition(observed_xy, observed_xy - previous_xy)
        if self.motion_prediction_pending and self.conds:
            self.conds[-1] = condition
        else:
            self.conds.append(condition)
        self._xy = observed_xy
        self.motion_prediction_pending = False

    @staticmethod
    def multi_predict_mamba(stracks, model, bev_grid_shape):
        if len(stracks) == 0:
            return

        conditions = [track.conds for track in stracks]
        kappa = None
        with torch.no_grad():
            if stracks[0].requires_motion_kappa:
                motion, covariance, kappa = model.generate_distribution(
                    conditions, bev_grid_shape=bev_grid_shape, return_kappa=True,
                )
            elif hasattr(model, "generate_distribution"):
                motion, covariance = model.generate_distribution(
                    conditions,
                    bev_grid_shape=bev_grid_shape,
                )
            else:
                motion = model.generate(
                    conditions,
                    bev_grid_shape=bev_grid_shape,
                )
                covariance = None

        for row_index, (track, delta) in enumerate(zip(stracks, motion)):
            track.motion_kappa = None if kappa is None else float(kappa[row_index, 0])
            previous_xy = track.xy
            predicted_xy = previous_xy + np.asarray(delta, dtype=np.float32)
            track.motion_delta = np.asarray(delta, dtype=np.float32)
            track.motion_prediction_xy = predicted_xy.copy()
            track.motion_covariance = (
                None
                if covariance is None
                else np.asarray(covariance[row_index], dtype=np.float32)
            )
            track._xy = predicted_xy
            track.conds.append(track._motion_condition(
                predicted_xy,
                predicted_xy - previous_xy,
            ))
            track.motion_prediction_pending = True

    @staticmethod
    def commit_mamba_state(stracks, model, bev_grid_shape):
        if len(stracks) == 0:
            return

        tokens = np.asarray([track.conds[-1] for track in stracks], dtype=np.float32)
        states = [track.mamba_state for track in stracks]
        kappa = None
        with torch.no_grad():
            if stracks[0].requires_motion_kappa:
                motion, covariance, next_states, kappa = model.stateful_step_distribution(
                    tokens, states=states, bev_grid_shape=bev_grid_shape,
                    return_kappa=True,
                )
            elif hasattr(model, "stateful_step_distribution"):
                motion, covariance, next_states = model.stateful_step_distribution(
                    tokens,
                    states=states,
                    bev_grid_shape=bev_grid_shape,
                )
            else:
                motion, next_states = model.stateful_step(
                    tokens,
                    states=states,
                    bev_grid_shape=bev_grid_shape,
                )
                covariance = None

        for row_index, (track, delta, state) in enumerate(zip(stracks, motion, next_states)):
            track.mamba_next_kappa = None if kappa is None else float(kappa[row_index, 0])
            track.mamba_state = state
            track.mamba_next_delta = np.asarray(delta, dtype=np.float32)
            track.mamba_next_covariance = (
                None
                if covariance is None
                else np.asarray(covariance[row_index], dtype=np.float32)
            )

    @staticmethod
    def multi_predict_mamba_stateful(stracks, model, bev_grid_shape):
        if len(stracks) == 0:
            return

        missing_state = [
            track for track in stracks
            if track.mamba_state is None or track.mamba_next_delta is None
        ]
        STrack.commit_mamba_state(missing_state, model, bev_grid_shape)

        for track in stracks:
            track.motion_kappa = track.mamba_next_kappa
            previous_xy = track.xy
            predicted_xy = previous_xy + track.mamba_next_delta
            track.motion_delta = track.mamba_next_delta.copy()
            track.motion_prediction_xy = predicted_xy.copy()
            track.motion_covariance = (
                None
                if track.mamba_next_covariance is None
                else track.mamba_next_covariance.copy()
            )
            track._xy = predicted_xy.astype(np.float32)
            track.conds.append(track._motion_condition(
                track._xy,
                track._xy - previous_xy,
            ))
            track.motion_prediction_pending = True

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class Tracker:
    """MVTrackTrans-style BEV association using Euclidean center distance."""

    def __init__(
        self,
        high_cls_thres=0.4,
        track_buffer=5,
        match_thresh=75.0,
        unconfirmed_match_thresh=100.0,
        duplicate_thresh=6.0,
        motion_model="kalman",
        bev_grid_shape=None,
        device=None,
        motion_predictor=None,
        motion_history_len=5,
        mamba_history_mode="window",
        mamba_uncertainty_mode="none",
        mamba_uncertainty_gate="none",
        implicit_motion_uncertainty_type="total",
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

        if motion_model not in ("kalman", "mamba", "bimamba"):
            raise ValueError("motion_model must be 'kalman', 'mamba', or 'bimamba'")
        self.motion_model = motion_model
        self.bev_grid_shape = bev_grid_shape
        self.motion_history_len = int(motion_history_len)
        if mamba_history_mode not in ("window", "stateful"):
            raise ValueError("mamba_history_mode must be 'window' or 'stateful'")
        if motion_model == "bimamba" and mamba_history_mode != "window":
            raise ValueError("BiMamba requires mamba_history_mode='window'")
        self.mamba_history_mode = mamba_history_mode
        self.mamba_uncertainty_mode = mamba_uncertainty_mode
        self.mamba_uncertainty_gate = mamba_uncertainty_gate
        self.implicit_motion_uncertainty_type = implicit_motion_uncertainty_type

        self.kalman_filter = KalmanFilter() if motion_model == "kalman" else None
        self.motion_predictor = None
        self.last_motion_predictions = []
        if motion_model in ("mamba", "bimamba"):
            if bev_grid_shape is None:
                raise ValueError("bev_grid_shape is required for BEV-normalized learned motion")
            if motion_predictor is None:
                raise ValueError("a trained motion predictor is required")
            self.motion_predictor = motion_predictor
            self.motion_predictor.to(device or "cpu")
            self.motion_predictor.eval()

    def _make_detections(self, dets, scores, uncertainties):
        return [
            STrack(
                xy,
                score,
                self.max_time_lost,
                motion_history_len=self.motion_history_len,
                detection_uncertainty=uncertainty,
                mamba_uncertainty_mode=self.mamba_uncertainty_mode,
                mamba_uncertainty_gate=self.mamba_uncertainty_gate,
                implicit_motion_uncertainty_type=self.implicit_motion_uncertainty_type,
            )
            for xy, score, uncertainty in zip(dets, scores, uncertainties)
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
        uncertainties=None,
        frame_id=None,
        **_,
    ):
        self.frame_id += 1

        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 2)

        if scores is None:
            scores = np.ones((len(dets),), dtype=np.float32)

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if uncertainties is None:
            uncertainties = np.zeros((len(dets),), dtype=np.float32)
        uncertainties = np.asarray(uncertainties, dtype=np.float32).reshape(-1)

        if len(dets) != len(scores) or len(dets) != len(uncertainties):
            raise ValueError("dets, scores, and uncertainties must have the same length")

        candidate_indices = np.where(scores > self.low_det_thresh)[0]

        detections = self._make_detections(
            dets[candidate_indices],
            scores[candidate_indices],
            uncertainties[candidate_indices],
        )

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        stateful_commit_tracks = []

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
            if self.mamba_history_mode == "stateful":
                STrack.multi_predict_mamba_stateful(
                    strack_pool,
                    self.motion_predictor,
                    self.bev_grid_shape,
                )
            else:
                STrack.multi_predict_mamba(
                    strack_pool,
                    self.motion_predictor,
                    self.bev_grid_shape,
                )

        predicted_tracks = list(strack_pool)

        matches_idx, unmatched_tracks_idx, unmatched_detections_idx = (
            self._linear_association(strack_pool, detections, self.match_thresh)
        )
        matches = [(strack_pool[int(i)], detections[int(j)]) for i, j in matches_idx]
        matched_prediction_ids = {id(track) for track, _ in matches}
        unmatched_tracks = [strack_pool[int(i)] for i in unmatched_tracks_idx]
        detections = [detections[int(i)] for i in unmatched_detections_idx]

        for track, det in matches:
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, motion_model=self.motion_model)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, motion_model=self.motion_model)
                refind_stracks.append(track)
            stateful_commit_tracks.append(track)

        for track in unmatched_tracks:
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
            stateful_commit_tracks.append(track)

        matches_idx, unmatched_unconfirmed_idx, unmatched_detections_idx = self._linear_association(
            unconfirmed, detections, self.unconfirmed_match_thresh
        )
        matches = [(unconfirmed[int(i)], detections[int(j)]) for i, j in matches_idx]
        unmatched_unconfirmed = [unconfirmed[int(i)] for i in unmatched_unconfirmed_idx]
        detections = [detections[int(i)] for i in unmatched_detections_idx]

        for track, det in matches:
            track.update(det, self.frame_id, motion_model=self.motion_model)
            activated_stracks.append(track)
            stateful_commit_tracks.append(track)

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
            stateful_commit_tracks.append(track)

        if self.motion_model == "mamba" and self.mamba_history_mode == "stateful":
            STrack.commit_mamba_state(
                stateful_commit_tracks,
                self.motion_predictor,
                self.bev_grid_shape,
            )

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

        self.last_motion_predictions = []
        for track in predicted_tracks:
            if track.motion_covariance is None:
                continue
            covariance = track.motion_covariance
            self.last_motion_predictions.append({
                "track_id": int(track.track_id),
                "matched": int(id(track) in matched_prediction_ids),
                "predicted_x": float(track.motion_prediction_xy[0]),
                "predicted_y": float(track.motion_prediction_xy[1]),
                "mean_dx": float(track.motion_delta[0]),
                "mean_dy": float(track.motion_delta[1]),
                "cov_xx": float(covariance[0, 0]),
                "cov_xy": float(covariance[0, 1]),
                "cov_yy": float(covariance[1, 1]),
            })

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
