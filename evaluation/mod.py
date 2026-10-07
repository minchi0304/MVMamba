import numpy as np
# from CLEAR_MOD_HUN import CLEAR_MOD_HUN
from evaluation.CLEAR_MOD_HUN import CLEAR_MOD_HUN
import os

def modMetricsCalculator(res_fpath, gt_fpath, distance_threshold=50 / 2.5, frame_ids=None):
    """
    This is simply the python translation of a MATLAB　Evaluation tool
    used to evaluate detection result created by P. Dollar.

    This API allow the project to run purely in Python without using MATLAB Engine.

    Some critical information to notice before you use this API:
    1. This API is only tested and deployed in this project: MVDet https://github.com/hou-yz/MVDet,
    might not be compatible with other projects.
    2. The detection result using this API is a little lower (approximately 0~2% decrease in MODA, MODP)
    than that using MATLAB evaluation tool, the reason might be that the Hungarian Algorithm implemented in
    sklearn.utils.linear_assignment_.linear_assignment is a little different with the one implemented by
    P. Dollar, hence leading to different results. Therefore, please use the official MATLAB API if you
    want to obtain the same result shown in the paper. This Python API is only used for convenience.
    3. The training process would not be affected by this API.

    @param res_fpath: detection result file path
    @param gt_fpath: ground truth result file path
    @param distance_threshold: matching radius in original grid cells
    @param frame_ids: evaluation frames; None evaluates the union of GT and prediction frames

    @return: recall, precision, MODA, MODP
    """
    def _load_rows(path, min_cols=3):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return np.empty((0, min_cols))

        rows = np.loadtxt(path)
        rows = np.asarray(rows)

        if rows.size == 0:
            return np.empty((0, min_cols))

        rows = np.atleast_2d(rows)
        return rows
    

    # gtRaw = np.loadtxt(gt_fpath)
    # detRaw = np.loadtxt(res_fpath)

    gtRaw = _load_rows(gt_fpath)
    detRaw = _load_rows(res_fpath)
    if frame_ids is not None:
        # GT files can contain both training and test annotations.
        frame_ids = np.asarray(list(frame_ids))
        gtRaw = gtRaw[np.isin(gtRaw[:, 0], frame_ids)]
        detRaw = detRaw[np.isin(detRaw[:, 0], frame_ids)]
    frames = np.union1d(gtRaw[:, 0], detRaw[:, 0])
    frame_ctr = 0
    gt_flag = True
    det_flag = True

    gtAllMatrix = 0
    detAllMatrix = 0
    if frames.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    for t in frames:
        idxs = np.where(gtRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in gtRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in gtRaw[idx, 2]])

        if gt_flag:
            gtAllMatrix = tmp_arr
            gt_flag = False
        else:
            gtAllMatrix = np.concatenate((gtAllMatrix, tmp_arr), axis=0)
        idxs = np.where(detRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in detRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in detRaw[idx, 2]])

        if det_flag:
            detAllMatrix = tmp_arr
            det_flag = False
        else:
            detAllMatrix = np.concatenate((detAllMatrix, tmp_arr), axis=0)
        frame_ctr += 1
    recall, precision, MODA, MODP = CLEAR_MOD_HUN(gtAllMatrix, detAllMatrix, distance_threshold=distance_threshold)
    return recall, precision, MODA, MODP
