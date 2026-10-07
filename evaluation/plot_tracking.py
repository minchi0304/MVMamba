import argparse
import hashlib
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_mot_rows(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 11), dtype=np.float32)

    rows = np.loadtxt(path, delimiter=",")
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.size == 0:
        return np.zeros((0, 11), dtype=np.float32)
    return rows.reshape(-1, 11)


def _track_color(index):
    return plt.get_cmap("hsv")((index * 0.61803398875) % 1.0)


def _id_to_bgr(track_id):
    digest = hashlib.sha256(str(int(track_id)).encode("utf-8")).hexdigest()
    return tuple(int(digest[i:i + 2], 16) for i in (4, 2, 0))


def _interpolate_color(color1, color2, ratio):
    return tuple(int(color1[i] * (1.0 - ratio) + color2[i] * ratio) for i in range(3))


def _draw_interpolated_points(img, points, start_color, end_color=(123, 123, 123)):
    if len(points) == 0:
        return
    if len(points) == 1:
        cv2 = _import_cv2()
        cv2.circle(img, points[0], 2, start_color, -1)
        return

    cv2 = _import_cv2()
    denom = max(1, len(points) - 1)
    for idx in range(len(points) - 1):
        color = _interpolate_color(start_color, end_color, idx / denom)
        cv2.line(img, points[idx], points[idx + 1], color, 2)
        cv2.circle(img, points[idx + 1], 2, color, -1)


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("tracking video rendering requires opencv-python/cv2") from exc
    return cv2


def _resolve_processing_fps(processing_fps, frame_id, seq_id=None):
    if processing_fps is None:
        return None
    if not isinstance(processing_fps, dict):
        return float(processing_fps)

    frame_id = int(frame_id)
    if seq_id is not None:
        seq_id = int(seq_id)
        for key in ((seq_id, frame_id), "{}:{}".format(seq_id, frame_id)):
            if key in processing_fps:
                return float(processing_fps[key])

    for key in (frame_id, str(frame_id)):
        if key in processing_fps:
            return float(processing_fps[key])

    frame_values = []
    for key, value in processing_fps.items():
        if isinstance(key, tuple) and len(key) >= 2 and int(key[1]) == frame_id:
            frame_values.append(float(value))
    if frame_values:
        return float(np.mean(frame_values))
    return None


def _filter_sequence_rows(rows, seq_ids=None):
    if seq_ids is None:
        return rows
    seq_ids = {int(seq_id) for seq_id in seq_ids}
    if not seq_ids or len(rows) == 0:
        return np.zeros((0, 11), dtype=rows.dtype)
    keep = np.isin(rows[:, 0].astype(np.int32), list(seq_ids))
    return rows[keep]


def _tracking_context(rows):
    seq_ids = sorted({int(seq) for seq in rows[:, 0]}) if len(rows) else [0]
    contexts = []
    for seq_id in seq_ids:
        seq_rows = rows[rows[:, 0] == seq_id] if len(rows) else rows
        track_ids = sorted({int(track_id) for track_id in seq_rows[:, 2]}) if len(seq_rows) else []
        if len(seq_rows):
            pad = 10.0
            xlim = (float(np.min(seq_rows[:, 8])) - pad, float(np.max(seq_rows[:, 8])) + pad)
            ylim = (float(np.min(seq_rows[:, 9])) - pad, float(np.max(seq_rows[:, 9])) + pad)
        else:
            xlim = None
            ylim = None
        contexts.append({
            "seq_id": seq_id,
            "rows": seq_rows,
            "track_ids": track_ids,
            "xlim": xlim,
            "ylim": ylim,
        })
    return contexts


def _draw_tracking_axes(
    ax,
    context,
    color_indices,
    seq_names=None,
    frame_id=None,
    trail_length=None,
    show_text=True,
    line_width=2.0,
    marker_size=8,
):
    ax.set_facecolor("black")
    if show_text:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")
    ax.axis("off")

    seq_rows = context["rows"]
    if frame_id is not None and len(seq_rows):
        seq_rows = seq_rows[seq_rows[:, 1] <= frame_id]
    if trail_length is not None and len(seq_rows):
        min_frame = frame_id - trail_length + 1
        seq_rows = seq_rows[seq_rows[:, 1] >= min_frame]

    for track_id in context["track_ids"]:
        track = seq_rows[seq_rows[:, 2] == track_id] if len(seq_rows) else seq_rows
        track = track[np.argsort(track[:, 1])]
        if len(track) == 0:
            continue
        color = _track_color(color_indices[track_id])
        ax.plot(track[:, 8], track[:, 9], linewidth=line_width, color=color)

    if context["xlim"] is not None:
        ax.set_xlim(*context["xlim"])
        ax.set_ylim(*context["ylim"])
    elif show_text:
        ax.text(0.5, 0.5, "No tracks", color="white", ha="center", va="center", transform=ax.transAxes)

    if not show_text:
        return

    seq_id = context["seq_id"]
    if seq_names is not None and seq_id < len(seq_names):
        ax.set_title(str(seq_names[seq_id]), color="white")
    else:
        ax.set_title("seq {}".format(seq_id), color="white")


def _make_tracking_figure(
    rows,
    seq_names=None,
    title=None,
    frame_id=None,
    trail_length=None,
    show_text=True,
    line_width=2.0,
    marker_size=8,
    square=False,
):
    contexts = _tracking_context(rows)
    fig_width = max(7, 6 * len(contexts))
    if square:
        fig_height = fig_width
    else:
        fig_height = fig_width * 9.0 / 16.0 if not show_text else 6
    fig, axes = plt.subplots(1, len(contexts), figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.reshape(-1)
    fig.patch.set_facecolor("black")

    for ax, context in zip(axes, contexts):
        color_indices = {track_id: color_idx for color_idx, track_id in enumerate(context["track_ids"])}
        _draw_tracking_axes(
            ax,
            context,
            color_indices,
            seq_names=seq_names,
            frame_id=frame_id,
            trail_length=trail_length,
            show_text=show_text,
            line_width=line_width,
            marker_size=marker_size,
        )

    if show_text and title:
        video_title = "{} | frame {}".format(title, int(frame_id)) if frame_id is not None else title
        fig.suptitle(video_title, color="white")
    elif show_text and frame_id is not None:
        fig.suptitle("frame {}".format(int(frame_id)), color="white")

    if show_text:
        plt.tight_layout()
    else:
        plt.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
    return fig


def _figure_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def plot_tracking_file(input_path, output_path=None, seq_names=None, title=None, seq_ids=None):
    rows = _load_mot_rows(input_path)
    rows = _filter_sequence_rows(rows, seq_ids=seq_ids)
    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + "_trajectory.png"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig = _make_tracking_figure(rows, seq_names=seq_names, title=title, show_text=False, square=True)
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_tracking_video(
    input_path,
    output_path=None,
    seq_names=None,
    title=None,
    fps=2.0,
    trail_length=None,
    seq_ids=None,
):
    rows = _load_mot_rows(input_path)
    rows = _filter_sequence_rows(rows, seq_ids=seq_ids)
    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + "_tracking.avi"

    cv2 = _import_cv2()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    frame_ids = sorted({int(frame) for frame in rows[:, 1]}) if len(rows) else [0]
    writer = None
    fourcc_name = "mp4v" if os.path.splitext(output_path)[1].lower() == ".mp4" else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)

    try:
        for frame_id in frame_ids:
            fig = _make_tracking_figure(
                rows,
                seq_names=seq_names,
                title=title,
                frame_id=frame_id,
                trail_length=trail_length,
                show_text=False,
                line_width=2.0,
                marker_size=8,
            )
            rgb = _figure_to_rgb(fig)
            plt.close(fig)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError("failed to open video writer for {}".format(output_path))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    return output_path


def _sequence_datasets(dataset):
    if hasattr(dataset, "datasets"):
        return list(dataset.datasets)
    return [dataset]


def _sequence_names(dataset, seq_names=None):
    if seq_names is not None:
        return list(seq_names)
    if hasattr(dataset, "dataset_list"):
        return list(dataset.dataset_list)
    return [getattr(dataset, "root", "seq0")]


def _base_dataset(sequence_dataset):
    return getattr(sequence_dataset, "base", sequence_dataset)


def _read_pom_bboxes(sequence_dataset):
    base = _base_dataset(sequence_dataset)
    root = getattr(base, "root", getattr(sequence_dataset, "root", None))
    if root is None:
        raise ValueError("dataset does not expose a root path for rectangles.pom")

    pom_path = os.path.join(root, "rectangles.pom")
    if not os.path.exists(pom_path):
        raise FileNotFoundError(pom_path)

    img_shape = getattr(base, "img_shape", getattr(sequence_dataset, "img_shape", (1080, 1920)))
    img_h, img_w = int(img_shape[0]), int(img_shape[1])
    num_cam = int(getattr(sequence_dataset, "num_cam", getattr(base, "num_cam", 0)))
    bbox_by_pos_cam = {}
    cam_pos_pattern = re.compile(r"(\d+) (\d+)")
    cam_pos_bbox_pattern = re.compile(r"(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)")

    with open(pom_path, "r") as fp:
        for line in fp:
            if "RECTANGLE" not in line:
                continue
            cam_pos_match = cam_pos_pattern.search(line)
            if cam_pos_match is None:
                continue
            cam, pos = map(int, cam_pos_match.groups())
            if num_cam and cam >= num_cam:
                continue
            if pos not in bbox_by_pos_cam:
                bbox_by_pos_cam[pos] = {}
            if "notvisible" in line:
                bbox_by_pos_cam[pos][cam] = None
                continue

            bbox_match = cam_pos_bbox_pattern.search(line)
            if bbox_match is None:
                bbox_by_pos_cam[pos][cam] = None
                continue
            cam, pos, left, top, right, bottom = map(int, bbox_match.groups())
            bbox_by_pos_cam[pos][cam] = [
                max(left, 0),
                max(top, 0),
                min(right, img_w - 1),
                min(bottom, img_h - 1),
            ]

    return bbox_by_pos_cam


def _prediction_to_pos(base, xy):
    grid = np.asarray(xy, dtype=np.float32)
    if getattr(base, "indexing", "xy") != "xy":
        grid = grid[[1, 0]]
    grid = np.rint(grid).astype(np.int64)
    return int(base.get_pos_from_worldgrid(grid))


def _bottom_center(bbox):
    return (int((bbox[2] - bbox[0]) / 2) + int(bbox[0]), int(bbox[3]))


def _put_tile(canvas, tile, tile_index, tile_width, tile_height, gap, cols):
    cv2 = _import_cv2()
    row = tile_index // cols
    col = tile_index % cols
    x0 = col * (tile_width + gap)
    y0 = row * (tile_height + gap)
    canvas[y0:y0 + tile_height, x0:x0 + tile_width] = cv2.resize(tile, (tile_width, tile_height))


def _blank_tile(tile_width, tile_height, text=None):
    cv2 = _import_cv2()
    img = np.ones((tile_height, tile_width, 3), dtype=np.uint8) * 255
    if text:
        cv2.putText(img, text, (20, max(35, tile_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def _draw_camera_frame(img, seq_rows, frame_id, cam, base, bbox_by_pos_cam, trail_length):
    cv2 = _import_cv2()
    current_rows = seq_rows[seq_rows[:, 1] == frame_id] if len(seq_rows) else seq_rows
    image_h, image_w = img.shape[:2]

    for row in current_rows:
        track_id = int(row[2])
        try:
            pos_id = _prediction_to_pos(base, row[[8, 9]])
        except Exception:
            continue
        bbox = bbox_by_pos_cam.get(pos_id, {}).get(cam)
        if bbox is None:
            continue

        color = _id_to_bgr(track_id)
        track = seq_rows[(seq_rows[:, 2] == row[2]) & (seq_rows[:, 1] <= frame_id)]
        track = track[np.argsort(track[:, 1])]
        if trail_length is not None:
            track = track[-int(trail_length):]

        points = []
        for track_row in track:
            try:
                track_pos_id = _prediction_to_pos(base, track_row[[8, 9]])
            except Exception:
                continue
            track_bbox = bbox_by_pos_cam.get(track_pos_id, {}).get(cam)
            if track_bbox is not None:
                points.append(_bottom_center(track_bbox))
        _draw_interpolated_points(img, points, color)

        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        text_scale = max(1.0, image_h / 720.0)
        text_thickness = max(3, image_h // 240)
        text_origin = (int(bbox[0]), max(24, int(bbox[1]) - 6))
        cv2.putText(
            img,
            str(track_id),
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 0),
            text_thickness + 2,
        )
        cv2.putText(
            img,
            str(track_id),
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            text_thickness,
        )

    cv2.putText(
        img,
        "C{}".format(cam + 1),
        (max(20, image_w // 40), max(40, image_h - image_h // 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(1.0, image_h / 360.0),
        (32, 230, 248),
        max(2, image_h // 180),
    )
    return img


def _frame_order(sequence_dataset):
    if hasattr(sequence_dataset, "world_gt"):
        return sorted(int(frame) for frame in sequence_dataset.world_gt.keys())
    img_fpath = getattr(sequence_dataset, "img_fpath", {})
    frames = set()
    for cam_paths in img_fpath.values():
        frames.update(int(frame) for frame in cam_paths.keys())
    return sorted(frames)


def _first_image_size(sequence_datasets):
    cv2 = _import_cv2()
    for sequence_dataset in sequence_datasets:
        img_fpath = getattr(sequence_dataset, "img_fpath", {})
        for cam_paths in img_fpath.values():
            for image_path in cam_paths.values():
                img = cv2.imread(image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    return w, h
    for sequence_dataset in sequence_datasets:
        img_shape = getattr(sequence_dataset, "img_shape", None)
        if img_shape is not None:
            return int(img_shape[1]), int(img_shape[0])
    return 1920, 1080


def _bev_tile(rows, seq_name, frame_id, tile_width, tile_height):
    cv2 = _import_cv2()
    if len(rows):
        bev_rows = rows.copy()
        bev_rows[:, 0] = 0
    else:
        bev_rows = rows
    fig = _make_tracking_figure(
        bev_rows,
        seq_names=[seq_name],
        frame_id=frame_id,
        show_text=False,
        line_width=2.0,
        marker_size=8,
    )
    rgb = _figure_to_rgb(fig)
    plt.close(fig)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (tile_width, tile_height))


def plot_tracking_camera_video(
    input_path,
    dataset,
    output_path=None,
    seq_names=None,
    fps=2.0,
    trail_length=15,
    include_bev=True,
    processing_fps=None,
):
    rows = _load_mot_rows(input_path)
    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + "_camera_tracking.avi"

    cv2 = _import_cv2()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sequences = _sequence_datasets(dataset)
    names = _sequence_names(dataset, seq_names)
    image_w, image_h = _first_image_size(sequences)
    tile_width = max(1, int(image_w / 4))
    tile_height = max(1, int(image_h / 4))
    gap = 15
    cols = 3
    num_tiles = max(int(getattr(seq, "num_cam", 0)) for seq in sequences)
    if include_bev:
        num_tiles += 1
    rows_count = int(math.ceil(num_tiles / float(cols)))
    canvas_width = (tile_width + gap) * cols
    canvas_height = (tile_height + gap) * rows_count + 50

    fourcc_name = "mp4v" if os.path.splitext(output_path)[1].lower() == ".mp4" else "XVID"
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*fourcc_name),
        float(fps),
        (canvas_width, canvas_height),
    )
    if not writer.isOpened():
        raise RuntimeError("failed to open video writer for {}".format(output_path))

    try:
        for seq_idx, sequence_dataset in enumerate(sequences):
            seq_name = names[seq_idx] if seq_idx < len(names) else getattr(sequence_dataset, "root", "seq{}".format(seq_idx))
            seq_label = os.path.basename(os.path.normpath(str(seq_name))) or str(seq_name)
            base = _base_dataset(sequence_dataset)
            bbox_by_pos_cam = _read_pom_bboxes(sequence_dataset)
            seq_rows = rows[rows[:, 0] == seq_idx] if len(rows) else rows
            img_fpath = getattr(sequence_dataset, "img_fpath", {})
            num_cam = int(getattr(sequence_dataset, "num_cam", getattr(base, "num_cam", 0)))

            for frame_id in _frame_order(sequence_dataset):
                canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
                for cam in range(num_cam):
                    image_path = img_fpath.get(cam, {}).get(frame_id)
                    img = cv2.imread(image_path) if image_path is not None else None
                    if img is None:
                        img = _blank_tile(image_w, image_h, "missing C{}".format(cam + 1))
                    img = _draw_camera_frame(img, seq_rows, frame_id, cam, base, bbox_by_pos_cam, trail_length)
                    _put_tile(canvas, img, cam, tile_width, tile_height, gap, cols)

                if include_bev:
                    bev = _bev_tile(seq_rows, seq_label, frame_id, tile_width, tile_height)
                    _put_tile(canvas, bev, num_cam, tile_width, tile_height, gap, cols)

                frame_processing_fps = _resolve_processing_fps(processing_fps, frame_id, seq_idx)
                cv2.putText(
                    canvas,
                    "seq: {}  frame: {}".format(seq_label, frame_id)
                    if frame_processing_fps is None
                    else "seq: {}  frame: {}  FPS: {:.2f}".format(
                        seq_label,
                        frame_id,
                        frame_processing_fps,
                    ),
                    (30, canvas_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                )
                writer.write(canvas)
    finally:
        writer.release()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot BEV tracking trajectories from GMVD MOT output.")
    parser.add_argument("input", help="Path to mota_pred.txt")
    parser.add_argument("-o", "--output", default=None, help="Output image path")
    parser.add_argument("--video", action="store_true", help="Also write a frame-by-frame tracking video")
    parser.add_argument("--video-output", default=None, help="Output video path")
    parser.add_argument("--fps", type=float, default=2.0, help="Video frames per second")
    parser.add_argument("--trail-length", type=int, default=None, help="Only draw this many recent frames in the video")
    args = parser.parse_args()
    print(plot_tracking_file(args.input, args.output))
    if args.video:
        print(plot_tracking_video(args.input, args.video_output, fps=args.fps, trail_length=args.trail_length))


if __name__ == "__main__":
    main()
