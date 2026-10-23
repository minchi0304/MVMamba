import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import glob
import os
import time
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation.mod import modMetricsCalculator
from evaluation.mot import format_mot_summary, mot_metrics
from evaluation.plot_tracking import plot_tracking_camera_video, plot_tracking_file, plot_tracking_video
from tracker.multitracker import Tracker

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decoder(center_e, K=60, return_inds=False):
    """
    center_e: B,1,H,W
    Returns BEV cell coordinates in the reduced grid.
    """

    batch, cat, height, width = center_e.size()
    center_e = _nms(center_e)

    scores, inds, ys, xs = _topk(center_e, K=K)  # B,K

    scores = scores.unsqueeze(2)  # B,K,1
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)
    xy = torch.cat((xs, ys), dim=2)  # batch,K,2

    if return_inds:
        return xy.detach(), scores.detach(), inds.detach()
    return xy.detach(), scores.detach()


def _topk(scores, K=40):
    batch, cat, length, width = scores.size()  # cat = 1

    '''
    For each channel, select K positions with high scores, cat * K total positions
    topk_scores / topk_inds: (bs, cat, K)

    From cat * K positions, select K positions with high scores
    topk_score / topk_ind: (bs, K)
    topk_clses: (bs, K)
    '''
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    # topk_inds = topk_inds % (length * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    # feat: (bs, h*w, 2), ind: (bs, max_objs)
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # (bs, max_objs, 2)
    feat = feat.gather(1, ind)  # (bs, max_objs, 2)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (bs, 2, 56, 56) -> (bs, 56, 56, 2)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (bs, 56*56, 2)
    feat = _gather_feat(feat, ind)  # (bs, max_objs, 2)
    return feat


def cleanup(logdir, indices):
    if isinstance(indices, int):
        indices = [indices]
    retain = []
    for idx in indices:
        retain.extend([
            'map_'+str(idx)+'.jpg',
            'test_gta_scene5_'+str(idx)+'.txt',
            'test_gta_scene6_'+str(idx)+'.txt',
        ])
    os.chdir(logdir)
    for img in glob.glob('map_*.jpg'):
        if img not in retain:
            os.remove(img)
    occupancy_dir = 'occupancy_map'
    if os.path.isdir(occupancy_dir):
        for img in glob.glob(os.path.join(occupancy_dir, 'map_[0-9]*.jpg')):
            if os.path.basename(img) not in retain:
                os.remove(img)
    #for outfile in glob.glob('test*.txt'):
    #    if outfile not in retain:
    #        os.remove(outfile)


# Functions Train / Test
def isbest(new_key, prev_key):
    if not isinstance(new_key, tuple):
        new_key = (new_key,)
    return prev_key is None or new_key > prev_key


def detection_score_key(metrics):
    # score layout: [moda, modp, prec, rec, idf1, mota, motp_raw, motp, mt, ml]
    return (metrics[0],)


def tracking_score_key(metrics, criterion='idf1'):
    # score layout: [moda, modp, prec, rec, idf1, mota, motp_raw, motp, mt, ml]
    # Break ties in the selected metric using the other tracking metric.
    indices = {'idf1': (4, 5), 'mota': (5, 4)}[criterion]
    return tuple(metrics[index] for index in indices)


def detection_score_text(score_key):
    return "moda: {:.1f}%".format(score_key[0])


def tracking_score_text(score_key, criterion='idf1'):
    labels = {'idf1': 'idf1/mota', 'mota': 'mota/idf1'}[criterion]
    return "{}: {:.1f}%/{:.1f}%".format(labels, score_key[0], score_key[1])


def split_occupancy_params(bev_occupancy):
    evidence = F.softplus(bev_occupancy) + 1.0
    alpha = evidence[:, 0:1]
    beta = evidence[:, 1:2]
    return alpha, beta, alpha + beta


def get_occupancy_prob(bev_occupancy):
    alpha, _, evidence_sum = split_occupancy_params(bev_occupancy)
    return alpha / evidence_sum


def get_occupancy_uncertainty(bev_occupancy):
    _, _, evidence_sum = split_occupancy_params(bev_occupancy)
    return 2.0 / evidence_sum


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _scatter_detection_group(ax, xy, color, zorder):
    if len(xy) == 0:
        return
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=10,
        c=color,
        edgecolors="black",
        linewidths=0.25,
        zorder=zorder,
    )


def save_detection_visualization(
    save_path,
    bev_center,
    target,
    vis_detection_xy,
):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    target_map = target['center_bev'].detach().cpu().numpy().squeeze()
    confidence_map = bev_center.detach().cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    axes[0].set_title("GT")
    axes[0].imshow(target_map, vmin=0.0, vmax=1.0)
    axes[1].set_title("Confidence")
    axes[1].imshow(confidence_map, vmin=0.0, vmax=1.0)
    axes[2].set_title("Detection")
    axes[2].imshow(np.zeros_like(confidence_map), vmin=0.0, vmax=1.0, cmap="gray")
    _scatter_detection_group(axes[2], vis_detection_xy, "red", 4)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def get_decode_topk(args):
    dataset_name = str(getattr(args, "dataset", "")).lower()
    if dataset_name == "citytrack":
        return 200
    if dataset_name == "mvcrowdtrack":
        return 100
    if dataset_name in ["gmvd_train", "gmvd_test"]:
        return 80
    return 60


def decode_frame(model, dataset, item, batch_idx, device, args):
    data = item['img'].to(device)
    with torch.no_grad():
        bev_occupancy = model(dataset, item['root'], data, item.get('proj_mats'))

        bev_center = get_occupancy_prob(bev_occupancy)
        occupancy_uncertainty = get_occupancy_uncertainty(bev_occupancy)
        xy, scores, inds = decoder(
            bev_center,
            K=get_decode_topk(args),
            return_inds=True,
        )
        point_uncertainty_all = _transpose_and_gather_feat(
            occupancy_uncertainty,
            inds,
        ).view(occupancy_uncertainty.size(0), -1)

    xy_all = xy[0]
    scores_all = scores[0]
    score_values = scores_all.squeeze(-1)
    score_keep = score_values > float(args.high_cls_thres)

    return {
        'bev_center': bev_center,
        'occupancy_uncertainty': occupancy_uncertainty,
        'xy_all': xy_all,
        'scores_all': scores_all,
        'uncertainty_all': point_uncertainty_all[0],
        'uncertainty': point_uncertainty_all[0][score_keep],
        'xy': xy_all[score_keep],
        'scores': scores_all[score_keep],
        'eval_xy': xy_all[score_keep],
        'eval_scores': scores_all[score_keep],
        'vis_detection_xy': xy_all[score_keep].detach().cpu().numpy(),
    }


def get_frame_metadata(data_loader, item):
    root_name = item['root'][0] if isinstance(item['root'], (list, tuple)) else item['root']
    frame_id = int(torch.as_tensor(item['frame']).reshape(-1)[0].item())
    dataset_idx = data_loader.dataset.dataset_list.index(root_name)
    return root_name, frame_id, dataset_idx


def save_frame_visualization(logdir, epoch, target, decoded, frame_id, dataset_idx):
    if epoch == 0:
        return

    visual_dir = os.path.join(logdir, 'occupancy_map')
    os.makedirs(visual_dir, exist_ok=True)
    frame_name = 'map_seq_{}_frame_{}.jpg'.format(dataset_idx, frame_id)
    save_detection_visualization(
        os.path.join(visual_dir, frame_name),
        decoded['bev_center'],
        target,
        decoded['vis_detection_xy'],
    )


def build_tracker(args, dataset_config, tracker_device, motion_predictor=None):
    base_dataset = dataset_config['base'].base
    return Tracker(
        high_cls_thres=float(args.high_cls_thres),
        match_thresh=float(args.track_match_thresh),
        unconfirmed_match_thresh=float(args.track_low_match_thresh),
        motion_model=args.motion_model,
        bev_grid_shape=tuple(base_dataset.world_grid_shape),
        device=tracker_device,
        motion_predictor=motion_predictor,
        motion_history_len=int(args.motion_history_len),
        mamba_history_mode=getattr(args, 'mamba_history_mode', 'window'),
        mamba_uncertainty_mode=getattr(args, 'mamba_uncertainty_mode', 'none'),
        mamba_uncertainty_gate=getattr(args, 'mamba_uncertainty_gate', 'none'),
        motion_cost=getattr(args, 'motion_cost', 'euclidean'),
        mahalanobis_gate=getattr(args, 'mahalanobis_gate', 5.991),
        motion_cov_scale=getattr(args, 'motion_cov_scale', 1.0),
        motion_cov_floor=getattr(args, 'motion_cov_floor', 1e-4),
    )


def get_tracker(trackers_by_root, root_name, args, dataset_config, tracker_device, motion_predictor=None):
    if root_name not in trackers_by_root:
        trackers_by_root[root_name] = build_tracker(
            args,
            dataset_config,
            tracker_device,
            motion_predictor=motion_predictor,
        )
    return trackers_by_root[root_name]


def update_tracking_for_frame(
    data_loader,
    item,
    decoded,
    root_name,
    frame_id,
    dataset_idx,
    trackers_by_root,
    tracking_results,
    tracking_gt_results,
    tracking_uncertainty_results,
    tracking_scales,
    args,
    measure_fps=False,
    realtime_tic=None,
    motion_predictor=None,
):
    dataset_config = data_loader.dataset.dicts[root_name]

    if dataset_config['base'].base.indexing == 'xy':
        xy_save = decoded['eval_xy']
    else:
        xy_save = decoded['eval_xy'][:, [1, 0]]
    grid_reduce = float(dataset_config['base'].grid_reduce)
    xy_world = xy_save * grid_reduce

    base_dataset = dataset_config['base'].base
    tracking_scales[dataset_idx] = float(
        getattr(base_dataset, 'metric_grid_cell', float(base_dataset.grid_cell) / 100.0)
    )
    xy_track_world = decoded['xy_all'] * grid_reduce
    tracker = get_tracker(
        trackers_by_root,
        root_name,
        args,
        dataset_config,
        decoded['xy_all'].device,
        motion_predictor=motion_predictor,
    )
    tracks = tracker.update(
        xy_track_world.detach().cpu().numpy(),
        decoded['scores_all'].squeeze(-1).detach().cpu().numpy(),
        uncertainties=decoded['uncertainty_all'].detach().cpu().numpy(),
        frame_id=frame_id,
    )

    for prediction in tracker.last_motion_predictions:
        tracking_uncertainty_results[root_name].append({
            "frame": int(frame_id),
            **prediction,
        })

    realtime_elapsed = None
    if measure_fps and realtime_tic is not None:
        synchronize_cuda()
        realtime_elapsed = time.perf_counter() - realtime_tic

    for track in tracks:
        tracking_results[root_name].append([
            frame_id,
            track.track_id,
            float(track.xy[0]),
            float(track.xy[1]),
            float(track.score),
        ])

    grid_gt = torch.as_tensor(item['grid_gt']).reshape(-1, 3).cpu().numpy()
    if 'grid_gt_count' in item:
        gt_count = int(torch.as_tensor(item['grid_gt_count']).reshape(-1)[0].item())
        valid_grid_gt = grid_gt[:gt_count]
    else:
        valid_grid_gt = grid_gt[np.any(grid_gt != 0, axis=1)]
    for gt_x, gt_y, gt_pid in valid_grid_gt:
        tracking_gt_results[root_name].append([
            frame_id,
            int(gt_pid),
            float(gt_x),
            float(gt_y),
        ])

    result_rows = torch.cat([
        torch.ones((decoded['eval_xy'].shape[0], 1), device=decoded['eval_xy'].device, dtype=decoded['eval_xy'].dtype) * frame_id,
        xy_world,
        decoded['eval_scores'],
        torch.ones((decoded['eval_xy'].shape[0], 1), device=decoded['eval_xy'].device, dtype=decoded['eval_xy'].dtype) * dataset_idx,
    ], dim=1)
    return result_rows, realtime_elapsed


def stack_detection_results(all_res_rows, device):
    if len(all_res_rows) != 0:
        return torch.cat(all_res_rows, dim=0)
    return torch.zeros((0, 5), device=device)


def write_tracking_result_files(res_dir, data_loader, all_res, tracking_results, tracking_gt_results):
    np.savetxt(res_dir + '/all_res.txt', all_res.cpu().numpy(), '%.8f')

    mota_pred_rows = []
    mota_gt_rows = []
    for seq_idx, seq_name in enumerate(data_loader.dataset.dataset_list):
        for frame_id, track_id, x, y, score_val in tracking_results.get(seq_name, []):
            mota_pred_rows.append([seq_idx, frame_id, track_id, -1, -1, -1, -1, score_val, x, y, -1])
        for frame_id, gt_id, x, y in tracking_gt_results.get(seq_name, []):
            mota_gt_rows.append([seq_idx, frame_id, gt_id, -1, -1, -1, -1, 1, x, y, -1])
    mota_pred = np.asarray(mota_pred_rows, dtype=np.float32).reshape(-1, 11)
    mota_gt = np.asarray(mota_gt_rows, dtype=np.float32).reshape(-1, 11)
    np.savetxt(res_dir + '/mota_pred.txt', mota_pred, fmt='%.6f', delimiter=',')
    np.savetxt(res_dir + '/mota_gt.txt', mota_gt, fmt='%.6f', delimiter=',')


def write_motion_uncertainty_file(
    res_dir,
    epoch,
    data_loader,
    tracking_uncertainty_results,
):
    output_path = os.path.join(
        res_dir,
        'motion_uncertainty_epoch_{}.csv'.format(epoch),
    )
    fieldnames = [
        'sequence',
        'frame',
        'track_id',
        'matched',
        'predicted_x',
        'predicted_y',
        'mean_dx',
        'mean_dy',
        'cov_xx',
        'cov_xy',
        'cov_yy',
    ]
    with open(output_path, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for sequence in data_loader.dataset.dataset_list:
            for row in tracking_uncertainty_results.get(sequence, []):
                writer.writerow({'sequence': sequence, **row})
    return output_path


def normalize_plot_sequence_indices(seq_names, seq_ids=None):
    if seq_ids is None:
        return None
    normalized = []
    for seq_id in seq_ids:
        seq_id = int(seq_id)
        if seq_id < 0:
            seq_id += len(seq_names)
        if 0 <= seq_id < len(seq_names):
            normalized.append(seq_id)
    return normalized


def plot_tracking_results(res_dir, epoch, data_loader, plot_processing_fps, plot_videos=True, seq_ids=None):
    seq_names = data_loader.dataset.dataset_list
    seq_ids = normalize_plot_sequence_indices(seq_names, seq_ids)
    trajectory_path = os.path.join(res_dir, 'tracking_trajectory_' + str(epoch) + '.png')
    plot_tracking_file(
        res_dir + '/mota_pred.txt',
        trajectory_path,
        seq_names=seq_names,
        title='tracking trajectory epoch ' + str(epoch),
        seq_ids=seq_ids,
    )
    if not plot_videos:
        return

    video_path = os.path.join(res_dir, 'tracking_video_' + str(epoch) + '.avi')
    plot_tracking_video(
        res_dir + '/mota_pred.txt',
        video_path,
        seq_names=seq_names,
        title='tracking trajectory epoch ' + str(epoch),
        seq_ids=seq_ids,
    )
    camera_video_path = os.path.join(res_dir, 'tracking_camera_video_' + str(epoch) + '.avi')
    plot_tracking_camera_video(
        res_dir + '/mota_pred.txt',
        data_loader.dataset,
        camera_video_path,
        seq_names=data_loader.dataset.dataset_list,
        processing_fps=plot_processing_fps,
    )


def evaluate_detection_results(res_dir, epoch, data_loader, all_res):
    recall, precision, moda, modp = [], [], [], []
    for dataset_idx, dataset_name in enumerate(data_loader.dataset.dataset_list):
        res_list = []
        all_res_list_temp = all_res[all_res[:, 4] == dataset_idx, :]
        for frame in np.unique(all_res_list_temp[:, 0].cpu().numpy()):
            res = all_res_list_temp[all_res_list_temp[:, 0] == frame, :]
            positions = res[:, 1:3]
            count = positions.shape[0]
            res_list.append(torch.cat([torch.ones([count, 1], device=positions.device) * frame, positions], dim=1))
        res_list = torch.cat(res_list, dim=0).cpu().numpy() if res_list else np.empty([0, 3])
        mod_eval_path = os.path.join(res_dir, '_mod_eval_{}_{}.txt'.format(epoch, dataset_idx))
        np.savetxt(mod_eval_path, res_list[:, :3], '%d')

        evaluation_dataset = data_loader.dataset.dicts[dataset_name]["base"]
        base_dataset = evaluation_dataset.base
        try:
            lrecall, lprecision, lmoda, lmodp = modMetricsCalculator(
                mod_eval_path,
                os.path.abspath(base_dataset.gt_fname),
                distance_threshold=20.0,
                frame_ids=evaluation_dataset.world_gt.keys(),
            )
        finally:
            if os.path.exists(mod_eval_path):
                os.remove(mod_eval_path)
        moda.append(lmoda)
        modp.append(lmodp)
        precision.append(lprecision)
        recall.append(lrecall)

    return np.average(moda), np.average(modp), np.average(precision), np.average(recall)


def finalize_results(
    epoch,
    data_loader,
    res_fpath,
    plot_tracking,
    measure_fps,
    all_res_rows,
    tracking_results,
    tracking_gt_results,
    tracking_uncertainty_results,
    tracking_scales,
    realtime_update_time,
    realtime_update_count,
    realtime_fps_by_frame,
    score,
    device,
    plot_tracking_videos=True,
    plot_tracking_seq_ids=None,
):
    all_res = stack_detection_results(all_res_rows, device)
    res_dir = os.path.abspath(os.path.dirname(res_fpath))
    write_tracking_result_files(res_dir, data_loader, all_res, tracking_results, tracking_gt_results)
    if tracking_uncertainty_results:
        uncertainty_path = write_motion_uncertainty_file(
            res_dir,
            epoch,
            data_loader,
            tracking_uncertainty_results,
        )
        print('Motion uncertainty: {}'.format(uncertainty_path))

    realtime_fps = None
    if measure_fps and realtime_update_time > 0.0:
        realtime_fps = realtime_update_count / realtime_update_time
    plot_processing_fps = realtime_fps_by_frame if realtime_fps_by_frame else realtime_fps
    if plot_tracking:
        plot_tracking_results(
            res_dir,
            epoch,
            data_loader,
            plot_processing_fps,
            plot_videos=plot_tracking_videos,
            seq_ids=plot_tracking_seq_ids,
        )

    moda_avg, modp_avg, prec_avg, rec_avg = evaluate_detection_results(res_dir, epoch, data_loader, all_res)
    dataset_name = os.path.basename(data_loader.dataset.dataset_list[0]).lower()
    mot_eval_match_thresh = 2.0 if dataset_name in ["citytrack", "mvcrowdtrack"] else 1.0
    tracking_summary = mot_metrics(
        res_dir + '/mota_pred.txt',
        res_dir + '/mota_gt.txt',
        seq_scales=tracking_scales,
        match_thresh=mot_eval_match_thresh,
        verbose=False,
    )
    tracking_values = [
        tracking_summary["idf1"],
        tracking_summary["mota"],
        tracking_summary["motp_raw"],
        tracking_summary["motp"],
        tracking_summary["mt"],
        tracking_summary["ml"],
    ]
    score.append([
        moda_avg,
        modp_avg,
        prec_avg,
        rec_avg,
        *tracking_values,
    ])

    print(f'####### moda: {moda_avg:.1f}%, modp: {modp_avg:.1f}%, prec: {prec_avg:.1f}%, recall: {rec_avg:.1f}% #######')
    print(format_mot_summary(tracking_summary))

    return moda_avg
