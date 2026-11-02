r"""Plot per-frame GT/detection/motion points and motion covariance ellipses.

Standalone: only NumPy and Matplotlib are required; no model/GPU imports.
Run from the repository root, using files from the SAME evaluation:

    python visualize_motion_uncertainty.py \
        --motion-csv logs/RUN/motion_uncertainty_epoch_100.csv \
        --gt logs/RUN/mota_gt.txt --detections logs/RUN/all_res.txt \
        --frames 100 105 --output-dir outputs/motion_plots --show-ids

For GMVD/multiple sequences, also specify --sequence NAME --sequence-id N.
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
Ellipse semiaxes are ellipse_scale * sqrt(eigenvalue); no probability coverage
is claimed (in particular, 2 is not automatically a 95% NIW region).

all_res contains saved high-confidence detections, not all tracker candidates.
It is overwritten each evaluation: do not pair an old/best motion CSV with a
different epoch's all_res. GT IDs and tracker IDs are separate namespaces;
the plot shows contemporaneous points without inventing identity matches.
No trajectory lines are drawn. Exported plotted_points_COMPONENT.csv retains
frames/IDs so lines can be drawn separately later.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
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


def ellipse_geometry(covariance, ellipse_scale=2.0):
    """Return full width, full height, angle in degrees; tolerate roundoff only."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-10 * max(float(np.abs(covariance).max()), 1e-12)
    if eigenvalues[0] < -tolerance:
        raise ValueError(f"Covariance is not positive semidefinite: {covariance.tolist()}")
    diameters = 2. * ellipse_scale * np.sqrt(np.maximum(eigenvalues, 0.))
    major = eigenvectors[:, 1]
    return float(diameters[1]), float(diameters[0]), math.degrees(math.atan2(major[1], major[0]))


def prepare_points(motion, gt, detections, component, coordinate_scale, detection_order,
                   frames=None, track_ids=None, gt_ids=None, min_score=0.0, ellipse_scale=2.0):
    points = []
    for row in motion:
        if (frames is not None and row["frame"] not in frames) or (track_ids is not None and row["track_id"] not in track_ids):
            continue
        covariance = selected_covariance(row, component, coordinate_scale)
        try:
            geometry = ellipse_geometry(covariance, ellipse_scale)
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


def draw_frame(points, frame, sequence, component, ellipse_scale, xlim, ylim,
               unit="grid cells", show_ids=False, invert_y=False):
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
            ax.annotate(f"{prefix}{point['id']}", point["xy"], xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color=colors[point["source"]])
    for source, (marker, label, size) in styles.items():
        xy = np.array([p["xy"] for p in points if p["source"] == source]).reshape(-1, 2)
        kwargs = dict(facecolors="none", edgecolors=colors[source]) if source == "detection" else dict(color=colors[source])
        ax.scatter(xy[:, 0], xy[:, 1], marker=marker, s=size, label=label, zorder=3, **kwargs)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=colors["motion"], edgecolor=colors["motion"], alpha=.25))
    labels.append(f"{component.capitalize()} covariance (scale {ellipse_scale:g})")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=2, frameon=False)
    ax.set(xlim=xlim, ylim=ylim, xlabel=f"BEV x ({unit})", ylabel=f"BEV y ({unit})",
           title=f"{_basename(sequence)} | frame {frame} | {component}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(alpha=.25)
    if invert_y:
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


def write_points(path, points, sequence, sequence_id, component):
    fields = ["sequence", "sequence_id", "frame", "source", "id", "x", "y", "score", "matched",
              "component", "cov_xx", "cov_xy", "cov_yy"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in sorted(points, key=lambda p: (p["frame"], p["source"], p["id"] or 0)):
            row = dict(sequence=sequence, sequence_id=sequence_id, frame=point["frame"],
                       source=point["source"], id=point["id"], x=point["xy"][0], y=point["xy"][1],
                       score=point.get("score"), matched=point.get("matched"))
            if point["source"] == "motion":
                covariance = point["covariance"]
                row.update(component=component, cov_xx=covariance[0, 0],
                           cov_xy=covariance[0, 1], cov_yy=covariance[1, 1])
            writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion-csv", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True, help="mota_gt.txt from the same evaluation")
    parser.add_argument("--detections", type=Path, required=True, help="all_res.txt from the same evaluation")
    parser.add_argument("--sequence", help="CSV sequence name/path or unique basename")
    parser.add_argument("--sequence-id", type=int, help="Numeric evaluation dataset_list index for the selected sequence")
    parser.add_argument("--list-sequences", action="store_true", help="Print available names/IDs and exit")
    parser.add_argument("--frames", type=int, nargs="+", help="Frames to draw; default: all frames in the selected sequence")
    parser.add_argument("--track-ids", type=int, nargs="+", help="Filter motion predictions only")
    parser.add_argument("--gt-ids", type=int, nargs="+", help="Filter GT only; GT IDs are not tracker IDs")
    parser.add_argument("--component", choices=("total", "aleatoric", "epistemic"), default="total")
    parser.add_argument("--ellipse-scale", type=float, default=2., help="Ellipse semiaxis multiplier, not a coverage probability")
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
    return parser


def run(args):
    scale = 1. if args.meters_per_cell is None else args.meters_per_cell
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(args.ellipse_scale) or args.ellipse_scale <= 0:
        raise ValueError("--meters-per-cell and --ellipse-scale must be finite and positive")
    if args.dpi <= 0 or not math.isfinite(args.min_score) or not 0 <= args.min_score <= 1:
        raise ValueError("--dpi must be positive and --min-score must be in [0, 1]")
    for limits in (args.xlim, args.ylim):
        if limits is not None and (not np.isfinite(limits).all() or limits[0] >= limits[1]):
            raise ValueError("Axis limits must be finite, with MIN < MAX")
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
    frames = sorted(available_frames if args.frames is None else set(args.frames))
    if not frames or set(frames) - available_frames:
        raise ValueError(f"No data for requested frame(s): {sorted(set(frames) - available_frames)}")
    points = prepare_points(motion, gt, detections, args.component, scale, order,
                            set(frames), args.track_ids, args.gt_ids, args.min_score, args.ellipse_scale)
    if not points:
        raise ValueError("No points remain after applying filters")
    xlim, ylim = plot_limits(points)
    xlim, ylim = args.xlim if args.xlim is not None else xlim, args.ylim if args.ylim is not None else ylim
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", _basename(sequence)) or "sequence"
    output = args.output_dir / f"seq_{sequence_id}_{slug}"
    output.mkdir(parents=True, exist_ok=True)
    by_frame = {frame: [] for frame in frames}
    for point in points:
        by_frame[point["frame"]].append(point)
    files = []
    for frame in frames:
        fig = draw_frame(by_frame[frame], frame, sequence, args.component, args.ellipse_scale,
                         xlim, ylim, "m" if args.meters_per_cell is not None else "grid cells",
                         args.show_ids, args.invert_y)
        try:
            for extension in dict.fromkeys(args.formats):
                destination = output / f"frame_{frame:06d}_{args.component}.{extension}"
                fig.savefig(destination, dpi=args.dpi, bbox_inches="tight")
                files.append(destination)
        finally:
            plt.close(fig)
    write_points(output / f"plotted_points_{args.component}.csv", points, sequence, sequence_id, args.component)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(sequence=sequence, sequence_id=sequence_id, detection_order=order,
                    frames=frames, xlim=list(xlim), ylim=list(ylim))
    (output / f"plot_settings_{args.component}.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Saved {len(files)} figure(s) and point coordinates to {output}")
    return files


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
