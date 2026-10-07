import os

import motmetrics as mm
import numpy as np


def _load_mot_rows(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 11), dtype=np.float64)

    rows = np.loadtxt(path, delimiter=",")
    # Match MVTrackTrans's precision when scaling coordinates near the distance gate.
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.size == 0:
        return np.zeros((0, 11), dtype=np.float64)
    return rows.reshape(-1, 11)


def _resolve_scale(seq_scales, seq_idx, default_scale):
    if seq_scales is None:
        return float(default_scale)
    if isinstance(seq_scales, dict):
        return float(seq_scales.get(seq_idx, default_scale))
    if isinstance(seq_scales, (list, tuple)) and seq_idx < len(seq_scales):
        return float(seq_scales[seq_idx])
    return float(default_scale)


def _motmetrics_summary(gt_rows, dt_rows, scale, seq_scales, match_thresh):
    seq_ids = sorted({int(seq) for seq in gt_rows[:, 0]}) if len(gt_rows) else []
    accs = []
    for seq_idx in seq_ids:
        seq_scale = _resolve_scale(seq_scales, seq_idx, scale)
        acc = mm.MOTAccumulator()
        gt_seq = gt_rows[gt_rows[:, 0] == seq_idx]
        dt_seq = dt_rows[dt_rows[:, 0] == seq_idx]
        frames = sorted({int(frame) for frame in gt_seq[:, 1]}) if len(gt_seq) else []
        for frame_id in frames:
            gt_dets = gt_seq[gt_seq[:, 1] == frame_id][:, (2, 8, 9)]
            dt_dets = dt_seq[dt_seq[:, 1] == frame_id][:, (2, 8, 9)]

            dists = mm.distances.norm2squared_matrix(
                gt_dets[:, 1:3] * seq_scale,
                dt_dets[:, 1:3] * seq_scale,
                max_d2=match_thresh ** 2,
            )
            dists = np.sqrt(dists)

            acc.update(
                gt_dets[:, 0].astype(int).tolist(),
                dt_dets[:, 0].astype(int).tolist(),
                dists,
                frameid=frame_id,
            )
        accs.append(acc)

    if not accs:
        return {
            "idf1": 0.0,
            "mota": 0.0,
            "motp_raw": 0.0,
            "motp": 0.0,
            "mt": 0.0,
            "ml": 0.0,
        }

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True,
    )
    overall = summary.loc["OVERALL"]

    num_unique_objects = float(overall["num_unique_objects"])
    mt = float(overall["mostly_tracked"])
    ml = float(overall["mostly_lost"])
    if num_unique_objects > 0:
        mt /= num_unique_objects
        ml /= num_unique_objects

    motp_raw = float(overall["motp"])
    if np.isfinite(motp_raw):
        motp = (1.0 - motp_raw) * 100.0
    else:
        motp_raw = 0.0
        motp = 0.0

    return {
        "idf1": float(overall["idf1"]) * 100.0,
        "mota": float(overall["mota"]) * 100.0,
        "motp_raw": motp_raw,
        "motp": motp,
        "mt": mt * 100.0,
        "ml": ml * 100.0,
    }


def format_mot_summary(summary):
    return (
        "####### "
        f"idf1: {summary['idf1']:.1f}%, "
        f"mota: {summary['mota']:.1f}%, "
        f"motp_raw: {summary['motp_raw']:.4f}, "
        f"motp: {summary['motp']:.1f}%, "
        f"mt: {summary['mt']:.1f}%, "
        f"ml: {summary['ml']:.1f}% "
        "#######"
    )


def mot_metrics(tSource, gtSource, scale=0.025, seq_scales=None, match_thresh=1.0, verbose=False):
    gt_rows = _load_mot_rows(gtSource)
    dt_rows = _load_mot_rows(tSource)

    summary = _motmetrics_summary(gt_rows, dt_rows, scale, seq_scales, match_thresh)

    if verbose:
        print(format_mot_summary(summary))
    return summary
