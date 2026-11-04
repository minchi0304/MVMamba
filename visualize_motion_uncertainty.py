r"""Plot GT/detection/motion trajectories and motion covariance ellipses.

Run a trained checkpoint and plot its fresh inference results:

    python visualize_motion_uncertainty.py \
        --checkpoint logs/RUN/Best_Multiview_Tracking_IDF1_gmvd_train.pth \
        --data_root /dataset --sequence-id 0 --gmvd-eval-all-frames \
        --consecutive-frames 5 --max-objects 3

Checkpoint mode needs the project's CUDA/Mamba environment. Model settings are
read from log.txt beside the checkpoint, or --model-config LOG_OR_JSON, with
explicit CLI options taking precedence. Loading is strict. Only the selected
evaluation sequence is loaded, and every evaluation frame from its start through
the last requested frame is processed to preserve tracking history. GMVD IDs
are mapped automatically from the split CSV; --sequence OR --sequence-id is
sufficient. --list-sequences lists the actual ID/path mapping without inference.
--consecutive-frames N selects existing original IDs spaced by exactly one;
--start-frame sets a minimum start ID. --gmvd-eval-all-frames disables GMVD
subsampling for this checkpoint run without changing the split CSV. The default
automatic start leaves one earlier evaluation frame for tracking history.
Fresh CSVs (including NIW kappa) and inference settings are saved below
OUTPUT_DIR/inference/RUN_ID, then used directly to generate the figures.
main.py --resume does not automatically generate these covariance plots.

Alternatively, CSV-only plotting needs only NumPy and Matplotlib (no model/GPU
imports). Use files from the SAME evaluation:

    python visualize_motion_uncertainty.py \
        --motion-csv logs/RUN/motion_uncertainty_epoch_100.csv \
        --gt logs/RUN/mota_gt.txt --detections logs/RUN/all_res.txt \
        --frames 100 105 --max-objects 3 --output-dir outputs/motion_plots --show-ids

For GMVD/multiple sequences in CSV mode, specify --sequence NAME --sequence-id N.
NAME is the CSV sequence path (or its unique basename). N is its zero-based
index in the evaluation dataset_list, also stored in mota_gt/all_res. Use
--list-sequences to inspect names and numeric IDs; their order is NOT a mapping.

Input formats produced by this repository:
  motion CSV: sequence,frame,track_id,matched,predicted_x,predicted_y,
              mean_dx,mean_dy,cov_xx,cov_xy,cov_yy
  mota_gt:    comma-separated 11 columns; sequence=0, frame=1, ID=2, x/y=8/9
  all_res:    whitespace-separated [frame, x, y, score, sequence_id]

Coordinates/covariances are already in original BEV grid units. Wildtrack's
all_res uses reversed axes: --detection-order auto handles the Wildtrack
sequence basename; use --detection-order yx for a renamed Wildtrack directory.
GT and motion axes are already aligned. --meters-per-cell scales all positions
and covariance consistently, relative to the BEV grid origin.

Existing motion CSVs contain TOTAL covariance, regardless of the implicit
input option. --component aleatoric/epistemic requires an extra 'kappa' column.
This script cannot recover missing kappa from total covariance alone.
Ellipses use fixed semiaxes sqrt(eigenvalue), oriented by the covariance
eigenvectors. There is no adjustable display multiplier. This unit covariance
ellipse describes one standard deviation along each principal axis; no
particular probability coverage is claimed for the NIW predictive distribution.

all_res contains saved high-confidence detections, not all tracker candidates.
It is overwritten each evaluation: do not pair an old/best motion CSV with a
different epoch's all_res. GT IDs and tracker IDs are separate namespaces;
the plot shows contemporaneous points without inventing identity matches.
By default, all selected frames form one trajectory figure per sequence and
format. Object colors follow motion track IDs; GT/detection context uses that
track's color, with different markers/line styles for each source. Context is
still proximity-based, not verified identity or tracker assignment. Lines stop
at missing selected frames and GT identity changes. Unpaired GT uses its own
GT-ID color, and unpaired detections remain gray points. Use --layout frames
for the previous separate-frame figures. Exported plotted_points_COMPONENT.csv
retains frames/IDs, context mapping and trajectory object colors.

--max-objects N selects the first N available motion track IDs in ascending
order, keeping those same IDs across the requested frames (no replacements
when a track is absent). --track-ids can restrict the candidate IDs first.
Use --object-selection largest/smallest to rank objects by their mean covariance
trace over the selected frames, or both to show the largest and smallest together.
The covariance component follows --component. Missing predictions are excluded
from the mean. With these modes, --max-objects defaults to 1 (both: 2) and counts
the total objects shown; both splits the count between the two ends, giving the
larger end the extra object for odd counts. Ties prefer smaller track IDs, and
no ID is selected twice. Selection stays fixed across the entire plot window.
Selected IDs, candidate mean traces, and frame counts are saved in plot settings.
--object-selection random selects one object by default, drawing again on each
invocation independently of the model's --seed. The same object is kept across
the selected frames within that invocation. --max-objects N samples N distinct
IDs. Random draws can repeat an ID on subsequent runs. --object-selection-seed
reproduces a selection; otherwise a fresh seed is generated and saved in settings.
Each frame then shows at most one nearby GT and one nearby detection per
selected motion point, hence at most 3*N points and N ellipses. Context points
are chosen by greedy shortest-distance, one-to-one selection among the frame's
available motion points, within --context-radius (default 50 original grid
cells). A context point is never reused. This is visualization-only spatial
context, NOT the actual tracker assignment or a verified GT identity mapping.
Absent context is left absent; GT IDs can change between frames. CSV columns
context_track_id/context_distance record this display selection, in plotted
coordinate units. Without --max-objects, existing independent filters apply.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import secrets

from motion_uncertainty_inference import add_inference_arguments, run_checkpoint, select_plot_frames

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
import numpy as np


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value, name):
    number = _finite(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def load_motion(path):
    required = {"sequence", "frame", "track_id", "predicted_x", "predicted_y",
                "cov_xx", "cov_xy", "cov_yy"}
    records, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, 2):
            try:
                record = {key: _finite(row[key], key) for key in
                          ("predicted_x", "predicted_y", "cov_xx", "cov_xy", "cov_yy")}
                record.update(sequence=row["sequence"], frame=_integer(row["frame"], "frame"),
                              track_id=_integer(row["track_id"], "track_id"))
                record["kappa"] = _finite(row["kappa"], "kappa") if row.get("kappa", "") else None
                record["matched"] = _integer(row["matched"], "matched") if row.get("matched", "") else None
                if record["matched"] not in (None, 0, 1):
                    raise ValueError("matched must be 0 or 1")
                key = (record["sequence"], record["frame"], record["track_id"])
                if key in seen:
                    raise ValueError(f"duplicate motion prediction {key}")
                seen.add(key)
                records.append(record)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: row {row_number}: {exc}") from exc
    return records


def load_numeric(path, columns, delimiter=None):
    with Path(path).open(encoding="utf-8-sig") as stream:
        lines = [line for line in stream if line.strip() and not line.lstrip().startswith('#')]
    if not lines:
        return np.empty((0, columns), dtype=float)
    rows = np.loadtxt(lines, delimiter=delimiter, ndmin=2)
    if rows.shape[1] != columns or not np.isfinite(rows).all():
        raise ValueError(f"{path}: expected {columns} finite numeric columns")
    integer_columns = (0, 1, 2) if columns == 11 else (0, 4)
    if not np.equal(rows[:, integer_columns], np.floor(rows[:, integer_columns])).all():
        raise ValueError(f"{path}: frame, sequence and object IDs must be integers")
    return rows


def _basename(name):
    return name.replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]


def select_sequence(motion, gt, detections, name=None, sequence_id=None):
    names = sorted({row["sequence"] for row in motion})
    numeric_ids = sorted({int(v) for v in gt[:, 0]} | {int(v) for v in detections[:, 4]})
    if name is not None:
        candidates = [name] if name in names else [s for s in names if _basename(s) == name]
        if names and len(candidates) != 1:
            raise ValueError("--sequence must match one CSV sequence (full path or unique basename)")
        name = candidates[0] if candidates else name
    elif len(names) > 1:
        raise ValueError("Multiple motion sequences: specify --sequence and --sequence-id; see --list-sequences")
    else:
        name = names[0] if names else f"sequence_{sequence_id if sequence_id is not None else 0}"
    if sequence_id is None:
        if len(names) > 1 or len(numeric_ids) > 1:
            raise ValueError("Specify --sequence-id from evaluation dataset_list; CSV name order is not a mapping")
        sequence_id = numeric_ids[0] if numeric_ids else 0
    if numeric_ids and sequence_id not in numeric_ids:
        raise ValueError(f"Sequence ID {sequence_id} is absent from both GT and detection files")
    return name, sequence_id, [r for r in motion if r["sequence"] == name], \
        gt[gt[:, 0] == sequence_id], detections[detections[:, 4] == sequence_id]


def selected_covariance(record, component="total", coordinate_scale=1.0):
    covariance = np.array([[record["cov_xx"], record["cov_xy"]],
                           [record["cov_xy"], record["cov_yy"]]], dtype=float)
    if component != "total":
        kappa = record["kappa"]
        if kappa is None or kappa <= 0:
            raise ValueError(f"{component} needs positive NIW kappa for frame {record['frame']}, "
                             f"track {record['track_id']}. Existing CSVs store only total covariance; "
                             "use --component total or provide a CSV with a kappa column.")
        covariance *= kappa / (kappa + 1.) if component == "aleatoric" else 1. / (kappa + 1.)
    return covariance * coordinate_scale ** 2


def ellipse_geometry(covariance):
    """Return unit covariance ellipse diameters and angle; tolerate roundoff only."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-10 * max(float(np.abs(covariance).max()), 1e-12)
    if eigenvalues[0] < -tolerance:
        raise ValueError(f"Covariance is not positive semidefinite: {covariance.tolist()}")
    # Matplotlib expects diameters; each semiaxis is sqrt(eigenvalue).
    diameters = 2. * np.sqrt(np.maximum(eigenvalues, 0.))
    major = eigenvectors[:, 1]
    return float(diameters[1]), float(diameters[0]), math.degrees(math.atan2(major[1], major[0]))


def prepare_points(motion, gt, detections, component, coordinate_scale, detection_order,
                   frames=None, track_ids=None, gt_ids=None, min_score=0.0):
    points = []
    for row in motion:
        if (frames is not None and row["frame"] not in frames) or (track_ids is not None and row["track_id"] not in track_ids):
            continue
        covariance = selected_covariance(row, component, coordinate_scale)
        try:
            geometry = ellipse_geometry(covariance)
        except ValueError as exc:
            raise ValueError(f"Frame {row['frame']}, track {row['track_id']}: {exc}") from exc
        points.append(dict(frame=row["frame"], source="motion", id=row["track_id"],
                           xy=np.array([row["predicted_x"], row["predicted_y"]]) * coordinate_scale,
                           covariance=covariance, ellipse=geometry, matched=row["matched"], score=None))
    for row in gt:
        frame, pid = int(row[1]), int(row[2])
        if (frames is None or frame in frames) and (gt_ids is None or pid in gt_ids):
            points.append(dict(frame=frame, source="gt", id=pid, xy=row[8:10] * coordinate_scale))
    for row in detections:
        frame = int(row[0])
        if (frames is None or frame in frames) and row[3] >= min_score:
            xy = row[1:3] if detection_order == "xy" else row[[2, 1]]
            points.append(dict(frame=frame, source="detection", id=None, xy=xy * coordinate_scale, score=row[3]))
    return points


def assign_context(points, context_radius):
    """Annotate unique nearby context without discarding unpaired points."""
    annotated = [dict(point) for point in points]
    by_frame = {}
    for point in annotated:
        by_frame.setdefault(point["frame"], []).append(point)
    for frame_points in by_frame.values():
        motion = sorted((p for p in frame_points if p["source"] == "motion"), key=lambda p: p["id"])
        for point in motion:
            point.update(context_track_id=point["id"], context_distance=0.)
        for source in ("gt", "detection"):
            candidates = [p for p in frame_points if p["source"] == source]
            if not motion or not candidates:
                continue
            # Unselected motion points also compete for nearby context.
            distances = np.linalg.norm(
                np.array([p["xy"] for p in motion])[:, None, :]
                - np.array([p["xy"] for p in candidates])[None, :, :], axis=-1,
            )
            pairs = sorted((float(distances[i, j]), int(i), int(j))
                           for i, j in np.argwhere(distances <= context_radius))
            used_motion, used_context = set(), set()
            for distance, motion_index, context_index in pairs:
                if motion_index in used_motion or context_index in used_context:
                    continue
                used_motion.add(motion_index)
                used_context.add(context_index)
                track_id = motion[motion_index]["id"]
                candidates[context_index].update(context_track_id=track_id, context_distance=distance)
    return annotated


def covariance_scores(points):
    """Mean trace of the displayed covariance, using only available predictions."""
    traces = {}
    for point in points:
        if point["source"] == "motion":
            traces.setdefault(point["id"], []).append(float(np.trace(point["covariance"])))
    return {identity: dict(mean_trace=float(np.mean(values)), frame_count=len(values))
            for identity, values in sorted(traces.items())}


def limit_objects(points, max_objects, context_radius, selection="id", rng=None):
    """Keep stable motion IDs and unique nearby context, never infer true IDs."""
    if selection == "id":
        track_ids = sorted({p["id"] for p in points if p["source"] == "motion"})[:max_objects]
    elif selection == "random":
        candidates = sorted({p["id"] for p in points if p["source"] == "motion"})
        generator = np.random.default_rng() if rng is None else rng
        track_ids = [int(identity) for identity in generator.choice(
            candidates, size=min(max_objects, len(candidates)), replace=False)]
    else:
        scores = covariance_scores(points)
        smallest = sorted(scores, key=lambda identity: (scores[identity]["mean_trace"], identity))
        largest = sorted(scores, key=lambda identity: (-scores[identity]["mean_trace"], identity))
        if selection == "largest":
            track_ids = largest[:max_objects]
        elif selection == "smallest":
            track_ids = smallest[:max_objects]
        elif selection == "both":
            count = min(max_objects, len(scores))
            track_ids = largest[:(count + 1) // 2]
            chosen = set(track_ids)
            track_ids += [identity for identity in smallest if identity not in chosen][:count - len(track_ids)]
        else:
            raise ValueError(f"Unknown object selection: {selection}")
    if not track_ids:
        raise ValueError("--max-objects requires motion predictions after frame/ID filtering")
    chosen = set(track_ids)
    return [p for p in assign_context(points, context_radius) if p.get("context_track_id") in chosen], track_ids


def plot_limits(points):
    bounds = []
    for point in points:
        radius = np.zeros(2)
        if point["source"] == "motion":
            width, height, angle = point["ellipse"]
            radians = math.radians(angle)
            radius = .5 * np.sqrt([
                (width * math.cos(radians)) ** 2 + (height * math.sin(radians)) ** 2,
                (width * math.sin(radians)) ** 2 + (height * math.cos(radians)) ** 2,
            ])
        bounds.extend((point["xy"] - radius, point["xy"] + radius))
    lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
    padding = np.maximum((upper - lower) * .06, 1e-3)
    return (lower[0] - padding[0], upper[0] + padding[0]), (lower[1] - padding[1], upper[1] + padding[1])


def draw_frame(points, frame, sequence, component, xlim, ylim,
               unit="grid cells", show_ids=False, invert_y=False, nearby_context=False):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"gt": "#238443", "detection": "#2166ac", "motion": "#d73027"}
    styles = {"gt": ("x", "GT", 65), "detection": ("s", "Detection", 48),
              "motion": ("o", "Motion prediction", 35)}
    for point in points:
        if point["source"] == "motion":
            width, height, angle = point["ellipse"]
            ax.add_patch(Ellipse(point["xy"], width=width, height=height, angle=angle,
                                 facecolor=colors["motion"], edgecolor="none", alpha=.12, zorder=1))
            ax.add_patch(Ellipse(point["xy"], width=width, height=height, angle=angle,
                                 facecolor="none", edgecolor=colors["motion"], linewidth=1.3, zorder=2))
        if show_ids and point["id"] is not None:
            prefix = "M" if point["source"] == "motion" else "G"
            label_offset = (5, -13) if point["source"] == "motion" else (5, 7)
            ax.annotate(f"{prefix}{point['id']}", point["xy"], xytext=label_offset,
                        textcoords="offset points", fontsize=8, color=colors[point["source"]])
    for source, (marker, label, size) in styles.items():
        if nearby_context and source != "motion":
            label += " (nearby)"
        xy = np.array([p["xy"] for p in points if p["source"] == source]).reshape(-1, 2)
        kwargs = dict(facecolors="none", edgecolors=colors[source]) if source == "detection" else dict(color=colors[source])
        ax.scatter(xy[:, 0], xy[:, 1], marker=marker, s=size, label=label, zorder=3, **kwargs)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=colors["motion"], edgecolor=colors["motion"], alpha=.25))
    labels.append(f"{component.capitalize()} covariance")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=2, frameon=False)
    ax.set(xlim=xlim, ylim=ylim, xlabel=f"BEV x ({unit})", ylabel=f"BEV y ({unit})",
           title=f"{_basename(sequence)} | frame {frame} | {component}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(False, which="both")
    if invert_y:
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


def color_trajectory_points(points):
    """Assign each displayed object one color for the complete selected window."""
    annotated = []
    for point in points:
        if point["source"] == "motion":
            label = f"M{point['id']}"
        elif point.get("context_track_id") is not None:
            label = f"M{point['context_track_id']}"
        elif point["source"] == "gt":
            label = f"G{point['id']}"
        else:
            label = None
        annotated.append(dict(point, object_label=label))
    labels = sorted({p["object_label"] for p in annotated if p["object_label"] is not None},
                    key=lambda label: (label[0] != "M", int(label[1:])))
    if len(labels) <= 10:
        palette = plt.get_cmap("tab10").colors[:len(labels)]
    else:
        palette = plt.get_cmap("turbo")(np.linspace(.05, .95, len(labels)))
    colors = {label: to_hex(color) for label, color in zip(labels, palette)}
    for point in annotated:
        point["object_color"] = colors.get(point["object_label"], "#909090")
    return annotated, colors


def trajectory_segments(points, frames):
    """Connect only the same source/identity at adjacent selected time steps."""
    order = {frame: index for index, frame in enumerate(sorted(set(frames)))}
    groups = {}
    for point in points:
        # A detection without a nearby track has no temporal identity.
        if point["object_label"] is None:
            continue
        identity = point["id"] if point["source"] == "gt" else None
        groups.setdefault((point["object_label"], point["source"], identity), []).append(point)
    segments = []
    for group in groups.values():
        segment = []
        for point in sorted(group, key=lambda p: p["frame"]):
            if segment and order[point["frame"]] != order[segment[-1]["frame"]] + 1:
                segments.append(segment)
                segment = []
            segment.append(point)
        if segment:
            segments.append(segment)
    return segments


def draw_trajectory(points, frames, sequence, component, xlim, ylim,
                    unit="grid cells", show_ids=False, invert_y=False):
    fig, ax = plt.subplots(figsize=(10, 7))
    styles = {"motion": ("o", "-", "Motion prediction"),
              "gt": ("x", "--", "GT (nearby)"),
              "detection": ("s", ":", "Detection (nearby)")}
    for point in points:
        if point["source"] == "motion":
            width, height, angle = point["ellipse"]
            ax.add_patch(Ellipse(point["xy"], width=width, height=height, angle=angle,
                                 facecolor=point["object_color"], edgecolor="none", alpha=.07, zorder=1))
            ax.add_patch(Ellipse(point["xy"], width=width, height=height, angle=angle,
                                 facecolor="none", edgecolor=point["object_color"], linewidth=1., alpha=.8, zorder=2))
    for segment in trajectory_segments(points, frames):
        first = segment[0]
        marker, linestyle, _ = styles[first["source"]]
        xy = np.array([point["xy"] for point in segment])
        ax.plot(xy[:, 0], xy[:, 1], color=first["object_color"], marker=marker, linestyle=linestyle,
                markersize=4.5, markerfacecolor="none" if first["source"] == "detection" else first["object_color"],
                linewidth=1.4, zorder=4 if first["source"] == "motion" else 3)
    unpaired = [p for p in points if p["object_label"] is None]
    if unpaired:
        xy = np.array([p["xy"] for p in unpaired])
        ax.scatter(xy[:, 0], xy[:, 1], marker="s", s=22, facecolors="none", edgecolors="#909090", zorder=3)
    objects = {p["object_label"]: p["object_color"] for p in points if p["object_label"] is not None}
    if show_ids:
        endpoints = {}
        for point in sorted(points, key=lambda p: p["frame"]):
            if point["source"] == "motion" or (point["object_label"] or "").startswith("G"):
                endpoints[point["object_label"]] = point
        for label, point in endpoints.items():
            ax.annotate(label, point["xy"], xytext=(5, 6), textcoords="offset points",
                        fontsize=8, color=point["object_color"])
    has_context = any(p["source"] != "motion" and p.get("context_track_id") is not None for p in points)
    handles = [Line2D([], [], color="#444444", marker=marker, linestyle=line,
                      markerfacecolor="none" if source == "detection" else "#444444",
                      label=label if has_context else label.replace(" (nearby)", ""))
               for source, (marker, line, label) in styles.items()]
    handles.append(Patch(facecolor="none", edgecolor="#444444", label=f"{component.capitalize()} covariance"))
    # Keep the source legend at figure level so adding the object legend cannot
    # replace it or exclude it from tight-bounding-box exports.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .01), ncol=2, frameon=False)
    if objects:
        object_handles = [Line2D([], [], color=color, linewidth=2.5, label=label)
                          for label, color in sorted(objects.items(), key=lambda pair: (pair[0][0] != "M", int(pair[0][1:])))]
        ax.legend(handles=object_handles, title="Object", loc="upper left", bbox_to_anchor=(1.01, 1.), frameon=False)
    ax.set(xlim=xlim, ylim=ylim, xlabel=f"BEV x ({unit})", ylabel=f"BEV y ({unit})",
           title=f"{_basename(sequence)} | frames {min(frames)}–{max(frames)} ({len(frames)} frames) | {component}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False, which="both")
    if invert_y:
        ax.invert_yaxis()
    fig.tight_layout(rect=(0, .12, 1, 1))
    return fig


def write_points(path, points, sequence, sequence_id, component):
    fields = ["sequence", "sequence_id", "frame", "source", "id", "x", "y", "score", "matched",
              "component", "cov_xx", "cov_xy", "cov_yy", "context_track_id", "context_distance", "object_label", "object_color"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in sorted(points, key=lambda p: (p["frame"], p["source"], p["id"] or 0)):
            row = dict(sequence=sequence, sequence_id=sequence_id, frame=point["frame"],
                       source=point["source"], id=point["id"], x=point["xy"][0], y=point["xy"][1],
                       score=point.get("score"), matched=point.get("matched"),
                       context_track_id=point.get("context_track_id"), context_distance=point.get("context_distance"),
                       object_label=point.get("object_label"), object_color=point.get("object_color"))
            if point["source"] == "motion":
                covariance = point["covariance"]
                row.update(component=component, cov_xx=covariance[0, 0],
                           cov_xy=covariance[0, 1], cov_yy=covariance[1, 1])
            writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", "--resume", type=Path, help="Run trained detector + motion weights and plot fresh results")
    source.add_argument("--motion-csv", type=Path, help="Plot an existing motion prediction CSV instead of running inference")
    parser.add_argument("--gt", type=Path, help="mota_gt.txt from the same evaluation; required with --motion-csv")
    parser.add_argument("--detections", type=Path, help="all_res.txt from the same evaluation; required with --motion-csv")
    parser.add_argument("--sequence", help="Sequence path or unique name; checkpoint mode also accepts scene/config/seq")
    parser.add_argument("--sequence-id", type=int, help="Evaluation sequence index; checkpoint mode resolves the matching path automatically")
    parser.add_argument("--list-sequences", action="store_true", help="Print available names/IDs and exit")
    frames = parser.add_mutually_exclusive_group()
    frames.add_argument("--frames", type=int, nargs="+", help="Exact frames to draw; default: all frames in the selected sequence")
    frames.add_argument("--consecutive-frames", type=int, metavar="N", help="Choose N existing original frame IDs spaced by 1")
    parser.add_argument("--start-frame", type=int, help="Minimum frame ID for --consecutive-frames")
    parser.add_argument("--layout", choices=("trajectory", "frames"), default="trajectory",
                        help="Combine selected frames in one trajectory figure (default), or save separate frames")
    parser.add_argument("--track-ids", type=int, nargs="+", help="Filter motion predictions only")
    parser.add_argument("--gt-ids", type=int, nargs="+", help="Filter GT only; GT IDs are not tracker IDs")
    parser.add_argument("--max-objects", type=int, help="Limit to N stable motion track IDs, plus at most one nearby GT/detection each")
    parser.add_argument("--object-selection", choices=("id", "largest", "smallest", "both", "random"), default="id",
                        help="Select by ID (default), mean covariance trace, or a fresh random draw; largest/smallest/random default to 1 object, both to 2")
    parser.add_argument("--object-selection-seed", type=int,
                        help="Reproduce random object selection (default: new seed each run, independent of model --seed)")
    parser.add_argument("--context-radius", type=float, default=50., help="Nearby point selection radius in original grid cells (default: 50); for object limits or trajectory colors")
    parser.add_argument("--component", choices=("total", "aleatoric", "epistemic"), default="total")
    parser.add_argument("--detection-order", choices=("auto", "xy", "yx"), default="auto")
    parser.add_argument("--min-score", type=float, default=0., help="Further filter the saved detections")
    parser.add_argument("--meters-per-cell", type=float, help="Convert grid positions/covariance to relative BEV meters")
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--show-ids", action="store_true", help="Annotate tracker IDs as M and GT IDs as G")
    parser.add_argument("--formats", choices=("png", "pdf", "svg"), nargs="+", default=["png"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/motion_uncertainty"))
    add_inference_arguments(parser)
    return parser


def run(args):
    if args.consecutive_frames is not None and args.consecutive_frames <= 0:
        raise ValueError("--consecutive-frames must be positive")
    if args.start_frame is not None and args.consecutive_frames is None:
        raise ValueError("--start-frame requires --consecutive-frames")
    scale = 1. if args.meters_per_cell is None else args.meters_per_cell
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("--meters-per-cell must be finite and positive")
    if args.dpi <= 0 or not math.isfinite(args.min_score) or not 0 <= args.min_score <= 1:
        raise ValueError("--dpi must be positive and --min-score must be in [0, 1]")
    if args.max_objects is not None and args.max_objects <= 0:
        raise ValueError("--max-objects must be positive")
    if args.object_selection == "both" and args.max_objects == 1:
        raise ValueError("--object-selection both requires --max-objects of at least 2 (or omit it)")
    if args.object_selection_seed is not None and args.object_selection_seed < 0:
        raise ValueError("--object-selection-seed must be nonnegative")
    max_objects = args.max_objects
    if max_objects is None and args.object_selection != "id":
        max_objects = 2 if args.object_selection == "both" else 1
    if not math.isfinite(args.context_radius) or args.context_radius <= 0:
        raise ValueError("--context-radius must be finite and positive")
    for limits in (args.xlim, args.ylim):
        if limits is not None and (not np.isfinite(limits).all() or limits[0] >= limits[1]):
            raise ValueError("Axis limits must be finite, with MIN < MAX")
    if args.checkpoint is not None:
        if args.gt is not None or args.detections is not None:
            raise ValueError("--checkpoint generates fresh GT/detections; omit --gt and --detections")
        if not run_checkpoint(args):
            return []
    elif args.gt is None or args.detections is None:
        raise ValueError("--motion-csv requires both --gt and --detections from the same evaluation")
    elif args.gmvd_eval_all_frames:
        raise ValueError("--gmvd-eval-all-frames requires --checkpoint; existing CSVs cannot supply unsaved frames")
    motion = load_motion(args.motion_csv)
    gt = load_numeric(args.gt, 11, delimiter=",")
    detections = load_numeric(args.detections, 5)
    if args.list_sequences:
        print("Motion CSV sequence names (not mapped by this order):")
        print('\n'.join(sorted({r["sequence"] for r in motion})) or "(no predictions)")
        print("GT numeric sequence IDs:", sorted({int(v) for v in gt[:, 0]}))
        print("Detection numeric sequence IDs:", sorted({int(v) for v in detections[:, 4]}))
        return []
    sequence, sequence_id, motion, gt, detections = select_sequence(
        motion, gt, detections, args.sequence, args.sequence_id)
    order = args.detection_order
    if order == "auto":
        order = "yx" if _basename(sequence).lower() == "wildtrack" else "xy"
    available_frames = {r["frame"] for r in motion} | {int(v) for v in gt[:, 1]} | {int(v) for v in detections[:, 0]}
    frames = select_plot_frames(available_frames, args.frames,
                                None if args.checkpoint else args.consecutive_frames,
                                None if args.checkpoint else args.start_frame)
    points = prepare_points(motion, gt, detections, args.component, scale, order,
                            set(frames), args.track_ids, args.gt_ids, args.min_score)
    selected_track_ids = None
    selection_scores = covariance_scores(points) if args.object_selection in ("largest", "smallest", "both") else None
    selection_seed, selection_rng = None, None
    if args.object_selection == "random":
        selection_seed = args.object_selection_seed if args.object_selection_seed is not None else secrets.randbits(64)
        selection_rng = np.random.default_rng(selection_seed)
    if max_objects is not None:
        points, selected_track_ids = limit_objects(points, max_objects, args.context_radius * scale, args.object_selection, selection_rng)
        print(f"Selected motion track IDs: {selected_track_ids}; GT/detection context uses proximity, not tracker assignment")
        if selection_seed is not None:
            print(f"Random object selection seed: {selection_seed}; IDs stay fixed within this plot window")
        if selection_scores is not None:
            print(f"Object selection: {args.object_selection}, mean trace of {args.component} covariance")
            for identity in selected_track_ids:
                score = selection_scores[identity]
                print(f"  M{identity}: mean trace={score['mean_trace']:.6g}, {score['frame_count']} motion frame(s)")
    elif args.layout == "trajectory":
        points = assign_context(points, args.context_radius * scale)
    if not points:
        raise ValueError("No points remain after applying filters")
    xlim, ylim = plot_limits(points)
    xlim, ylim = args.xlim if args.xlim is not None else xlim, args.ylim if args.ylim is not None else ylim
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", _basename(sequence)) or "sequence"
    output = args.output_dir / f"seq_{sequence_id}_{slug}"
    output.mkdir(parents=True, exist_ok=True)
    object_colors = None
    if args.layout == "trajectory":
        points, object_colors = color_trajectory_points(points)
        plots = [(frames, points, f"trajectory_{frames[0]:06d}_{frames[-1]:06d}_{args.component}")]
    else:
        by_frame = {frame: [] for frame in frames}
        for point in points:
            by_frame[point["frame"]].append(point)
        plots = [(frame, by_frame[frame], f"frame_{frame:06d}_{args.component}") for frame in frames]
    files = []
    for time_selection, plot_points, name in plots:
        draw = draw_trajectory if args.layout == "trajectory" else draw_frame
        options = {} if args.layout == "trajectory" else {"nearby_context": max_objects is not None}
        fig = draw(plot_points, time_selection, sequence, args.component,
                   xlim, ylim, "m" if args.meters_per_cell is not None else "grid cells",
                   args.show_ids, args.invert_y, **options)
        try:
            for extension in dict.fromkeys(args.formats):
                destination = output / f"{name}.{extension}"
                fig.savefig(destination, dpi=args.dpi, bbox_inches="tight")
                files.append(destination)
        finally:
            plt.close(fig)
    write_points(output / f"plotted_points_{args.component}.csv", points, sequence, sequence_id, args.component)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(sequence=sequence, sequence_id=sequence_id, detection_order=order,
                    frames=frames, xlim=list(xlim), ylim=list(ylim), selected_track_ids=selected_track_ids,
                    effective_max_objects=max_objects, object_covariance_scores=selection_scores,
                    effective_object_selection_seed=selection_seed,
                    object_selection_metric="mean_trace" if selection_scores is not None else None,
                    ellipse_definition="semiaxes=sqrt(eigenvalues)",
                    object_colors=object_colors,
                    context_selection="greedy_nearest_one_to_one" if max_objects is not None or args.layout == "trajectory" else None)
    (output / f"plot_settings_{args.component}.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Saved {len(files)} figure(s) and point coordinates to {output}")
    return files


def add_resume_plot_arguments(parser):
    """Options used by main.py; rendering still lives in this standalone file."""
    group = parser.add_argument_group("Motion covariance plots during --resume evaluation")
    group.add_argument("--plot_motion_uncertainty", action=argparse.BooleanOptionalAction, default=False,
                       help="Explicitly enable covariance plots during resume (default: disabled; use the standalone script)")
    group.add_argument("--motion_plot_seq_ids", type=int, nargs="+", help="Evaluation sequence IDs to plot; default: all")
    frames = group.add_mutually_exclusive_group()
    frames.add_argument("--motion_plot_frames", type=int, nargs="+", help="Exact frame IDs to plot; evaluation still processes every frame")
    frames.add_argument("--motion_plot_consecutive", type=int, metavar="N",
                        help="Automatically choose N existing frame IDs spaced by exactly 1 (after the first evaluation frame)")
    group.add_argument("--motion_plot_start_frame", type=int,
                       help="Minimum frame ID for automatic consecutive selection; only with --motion_plot_consecutive")
    group.add_argument("--motion_plot_max_objects", type=int, default=3, help="Maximum stable motion IDs per plotted sequence (default: 3)")
    group.add_argument("--motion_plot_object_selection", choices=("id", "largest", "smallest", "both", "random"), default="id",
                       help="Select motion objects by ID, mean covariance trace, or random draw")
    group.add_argument("--motion_plot_object_selection_seed", type=int,
                       help="Reproduce random object selection; default: fresh seed, independent of model --seed")
    group.add_argument("--motion_plot_component", choices=("total", "aleatoric", "epistemic"), default="total")
    group.add_argument("--motion_plot_layout", choices=("trajectory", "frames"), default="trajectory",
                       help="Combine selected frames in one figure (default), or save separate frames")
    group.add_argument("--motion_plot_track_ids", type=int, nargs="+")
    group.add_argument("--motion_plot_gt_ids", type=int, nargs="+")
    group.add_argument("--motion_plot_context_radius", type=float, default=50., help="Nearby GT/detection selection radius in original grid cells")
    group.add_argument("--motion_plot_xlim", type=float, nargs=2, metavar=("MIN", "MAX"))
    group.add_argument("--motion_plot_ylim", type=float, nargs=2, metavar=("MIN", "MAX"))
    group.add_argument("--motion_plot_show_ids", action="store_true")
    group.add_argument("--motion_plot_formats", choices=("png", "pdf", "svg"), nargs="+", default=["png", "pdf"])


def resume_plot_plan(args, dataset):
    """Validate plot filters before inference; never filter the evaluation loader."""
    if args.motion_plot_max_objects <= 0:
        raise ValueError("--motion_plot_max_objects must be positive")
    if args.motion_plot_object_selection == "both" and args.motion_plot_max_objects == 1:
        raise ValueError("--motion_plot_object_selection both requires --motion_plot_max_objects of at least 2")
    if args.motion_plot_object_selection_seed is not None and args.motion_plot_object_selection_seed < 0:
        raise ValueError("--motion_plot_object_selection_seed must be nonnegative")
    if not np.isfinite(args.motion_plot_context_radius) or args.motion_plot_context_radius <= 0:
        raise ValueError("--motion_plot_context_radius must be finite and positive")
    count = args.motion_plot_consecutive
    if count is not None and count <= 0:
        raise ValueError("--motion_plot_consecutive must be positive")
    if args.motion_plot_start_frame is not None and count is None:
        raise ValueError("--motion_plot_start_frame requires --motion_plot_consecutive")
    for limits in (args.motion_plot_xlim, args.motion_plot_ylim):
        if limits is not None and (not np.isfinite(limits).all() or limits[0] >= limits[1]):
            raise ValueError("Motion plot axis limits must be finite, with MIN < MAX")
    sequence_ids = sorted(set(range(len(dataset.dataset_list)) if args.motion_plot_seq_ids is None else args.motion_plot_seq_ids))
    if any(index < 0 or index >= len(dataset.dataset_list) for index in sequence_ids):
        raise ValueError("--motion_plot_seq_ids must be valid evaluation dataset_list indices")
    plan = {}
    for index in sequence_ids:
        available = dataset.datasets[index].frames
        try:
            frames = select_plot_frames(available, args.motion_plot_frames, count, args.motion_plot_start_frame)
        except ValueError as exc:
            message = str(exc).replace("No run", "no run").replace("--consecutive-frames", "--motion_plot_consecutive")
            message = message.replace("--start-frame", "--motion_plot_start_frame").replace("--gmvd-eval-all-frames", "--gmvd_eval_all_frames")
            raise ValueError(f"Motion plot sequence {index}: {message}") from exc
        if frames:
            plan[index] = frames
            print(f"Motion plot sequence {index} ({dataset.dataset_list[index]}): "
                  f"{frames if len(frames) <= 12 else str(len(frames)) + ' frames'}")
    return plan


def plot_resume_results(args, dataset, epoch, res_dir, plan):
    """Plot the just-completed evaluation files without loading/running a model."""
    res_dir = Path(res_dir)
    motion_csv = res_dir / f"motion_uncertainty_epoch_{epoch}.csv"
    if not motion_csv.is_file():
        print("Motion covariance plots skipped: this evaluation produced no motion predictions")
        return []
    motion = load_motion(motion_csv)
    gt_path, detections_path = res_dir / "mota_gt.txt", res_dir / "all_res.txt"
    gt, detections = load_numeric(gt_path, 11, ","), load_numeric(detections_path, 5)
    files = []
    for index, requested in plan.items():
        root = dataset.dataset_list[index]
        predictions = [row for row in motion if row["sequence"] == root]
        eligible = [row for row in predictions if row["frame"] in requested
                    and (args.motion_plot_track_ids is None or row["track_id"] in args.motion_plot_track_ids)]
        if not eligible:
            print(f"Motion covariance plots skipped for sequence {index}: no predictions for selected frames/IDs")
            continue
        # Entirely empty frames have no points to draw. This does not remove
        # any frames from tracking or evaluation metrics.
        available = ({row["frame"] for row in predictions}
                     | {int(row[1]) for row in gt if int(row[0]) == index}
                     | {int(row[0]) for row in detections if int(row[4]) == index})
        frames = sorted(set(requested) & available)
        if len(frames) != len(requested):
            print(f"Motion plots sequence {index}: skipped empty frames {sorted(set(requested) - available)}")
        plot_args = build_parser().parse_args([
            "--motion-csv", str(motion_csv), "--gt", str(gt_path), "--detections", str(detections_path),
        ])
        plot_args.sequence, plot_args.sequence_id, plot_args.frames = root, index, frames
        plot_args.output_dir = res_dir / "motion_uncertainty_plots"
        for name in ("layout", "max_objects", "object_selection", "object_selection_seed", "component", "track_ids", "gt_ids", "context_radius", "xlim", "ylim", "show_ids", "formats"):
            setattr(plot_args, name, getattr(args, "motion_plot_" + name))
        plot_args.detection_order = "xy" if dataset.datasets[index].base.indexing == "xy" else "yx"
        plot_args.evaluation_checkpoint = args.resume
        plot_args.evaluation_epoch = epoch
        plot_args.evaluation_settings = vars(args).copy()
        files.extend(run(plot_args))
    return files


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
