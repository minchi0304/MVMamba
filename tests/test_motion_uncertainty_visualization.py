import csv
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

import visualize_motion_uncertainty as viz


class MotionUncertaintyVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.record = dict(sequence="/dataset/scene", frame=2, track_id=42, matched=1,
                           predicted_x=10., predicted_y=20., cov_xx=4., cov_xy=-1., cov_yy=9., kappa=3.)

    def write_inputs(self, records=None, gt=None, detections=None):
        records = [self.record] if records is None else records
        motion_path, gt_path, det_path = (self.root / name for name in ("motion.csv", "mota_gt.txt", "all_res.txt"))
        with motion_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.record))
            writer.writeheader()
            writer.writerows(records)
        np.savetxt(gt_path, np.asarray([[0, 2, 7, -1, -1, -1, -1, 1, 10, 20, -1]] if gt is None else gt).reshape(-1, 11), delimiter=",")
        np.savetxt(det_path, np.asarray([[2, 10, 20, .9, 0]] if detections is None else detections).reshape(-1, 5))
        return ["--motion-csv", str(motion_path), "--gt", str(gt_path), "--detections", str(det_path),
                "--output-dir", str(self.root / "plots")]

    def test_rotated_ellipse_uses_covariance_eigenvectors_and_full_diameters(self):
        angle = np.deg2rad(30.)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        covariance = rotation @ np.diag([9., 1.]) @ rotation.T
        width, height, degrees = viz.ellipse_geometry(covariance)
        self.assertAlmostEqual(width, 6.)
        self.assertAlmostEqual(height, 2.)
        self.assertAlmostEqual(degrees % 180., 30.)
        # The displayed semiaxes reconstruct the original covariance, without
        # a user-selected multiplier or a probability-contour conversion.
        radians = np.deg2rad(degrees)
        axes = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
        np.testing.assert_allclose(axes @ np.diag([(width / 2.) ** 2, (height / 2.) ** 2]) @ axes.T, covariance)
        point = dict(source="motion", xy=np.array([10., 20.]), ellipse=(width, height, degrees))
        xlim, ylim = viz.plot_limits([point])
        self.assertLess(xlim[0], 10. - np.sqrt(covariance[0, 0]))
        self.assertGreater(ylim[1], 20. + np.sqrt(covariance[1, 1]))

    def test_component_decomposition_and_metric_scaling_preserve_total(self):
        total = viz.selected_covariance(self.record)
        aleatoric = viz.selected_covariance(self.record, "aleatoric", .1)
        epistemic = viz.selected_covariance(self.record, "epistemic", .1)
        np.testing.assert_allclose(aleatoric + epistemic, total * .01)
        np.testing.assert_allclose(aleatoric, total * .0075)
        self.assertLess(aleatoric[0, 1], 0.)
        self.assertEqual(self.record["cov_xx"], 4.)

    def test_legacy_total_is_supported_but_missing_kappa_cannot_be_guessed(self):
        record = dict(self.record, kappa=None)
        arguments = self.write_inputs(records=[record])
        loaded = viz.load_motion(self.root / "motion.csv")[0]
        np.testing.assert_array_equal(viz.selected_covariance(loaded), [[4., -1.], [-1., 9.]])
        for component in ("aleatoric", "epistemic"):
            with self.subTest(component=component), self.assertRaisesRegex(ValueError, "kappa"):
                viz.run(viz.build_parser().parse_args(arguments + ["--component", component]))
        self.assertFalse((self.root / "plots").exists())

    def test_sequence_mapping_requires_explicit_numeric_id_and_uses_it(self):
        motion = [self.record, dict(self.record, sequence="/dataset/second")]
        gt = np.array([[0, 2, 7, -1, -1, -1, -1, 1, 10, 20, -1],
                       [1, 2, 8, -1, -1, -1, -1, 1, 30, 40, -1]], dtype=float)
        detections = np.array([[2, 11, 21, .9, 0], [2, 31, 41, .9, 1]])
        with self.assertRaisesRegex(ValueError, "Multiple"):
            viz.select_sequence(motion, gt, detections)
        with self.assertRaisesRegex(ValueError, "sequence-id"):
            viz.select_sequence(motion, gt, detections, "second")
        name, idx, selected_motion, selected_gt, selected_dets = viz.select_sequence(motion, gt, detections, "second", 0)
        self.assertEqual((name, idx), ("/dataset/second", 0))
        self.assertEqual(selected_motion[0]["sequence"], name)
        np.testing.assert_array_equal(selected_gt[:, 2], [7])
        np.testing.assert_array_equal(selected_dets[:, 1:3], [[11., 21.]])

    def test_empty_and_single_row_files_and_bad_covariance(self):
        self.write_inputs(gt=[], detections=[])
        self.assertEqual(viz.load_numeric(self.root / "mota_gt.txt", 11, ",").shape, (0, 11))
        self.assertEqual(viz.load_numeric(self.root / "all_res.txt", 5).shape, (0, 5))
        self.assertEqual(len(viz.load_motion(self.root / "motion.csv")), 1)
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            viz.ellipse_geometry(np.array([[1., 2.], [2., 1.]]))
        self.assertEqual(viz.ellipse_geometry(np.zeros((2, 2)))[:2], (0., 0.))

    def test_wildtrack_axis_order_units_exports_and_gt_only_frame(self):
        args = self.write_inputs(
            records=[dict(self.record, sequence="/dataset/Wildtrack")],
            gt=[[0, 1, 7, -1, -1, -1, -1, 1, 8, 18, -1],
                [0, 2, 7, -1, -1, -1, -1, 1, 10, 20, -1]],
            detections=[[2, 20, 10, .9, 0]],
        )
        files = viz.run(viz.build_parser().parse_args(args + ["--frames", "1", "2", "--meters-per-cell", ".1",
                                                             "--layout", "frames", "--component", "aleatoric", "--formats", "png", "svg", "pdf"]))
        self.assertEqual(len(files), 6)
        self.assertTrue(all(path.stat().st_size > 100 for path in files))
        output = files[0].parent
        with (output / "plotted_points_aleatoric.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        det = next(row for row in rows if row["source"] == "detection")
        self.assertEqual((float(det["x"]), float(det["y"])), (1., 2.))
        motion = next(row for row in rows if row["source"] == "motion")
        self.assertAlmostEqual(float(motion["cov_xx"]), .03)
        self.assertEqual(motion["id"], "42")
        self.assertEqual({r["id"] for r in rows if r["source"] == "gt"}, {"7"})
        settings = json.loads((output / "plot_settings_aleatoric.json").read_text())
        self.assertEqual(settings["frames"], [1, 2])
        self.assertEqual(settings["detection_order"], "yx")
        self.assertFalse(plt.get_fignums())

    def test_filters_keep_gt_ids_separate_from_track_ids_and_do_not_draw_lines(self):
        gt = np.array([[0, 2, 7, -1, -1, -1, -1, 1, 10, 20, -1]], dtype=float)
        detections = np.array([[2, 10, 20, .9, 0], [2, 40, 50, .5, 0]])
        points = viz.prepare_points([self.record], gt, detections, "total", 1., "xy",
                                    {2}, [42], [7], min_score=.8)
        self.assertEqual(len(points), 3)
        fig = viz.draw_frame(points, 2, "scene", "total", (0, 50), (0, 50), show_ids=True)
        try:
            ax = fig.axes[0]
            self.assertEqual(len(ax.lines), 0)
            self.assertEqual({text.get_text() for text in ax.texts}, {"M42", "G7"})
            for patch in ax.patches:
                np.testing.assert_array_equal(patch.center, [10., 20.])
            self.assertEqual(ax.get_aspect(), 1.)
        finally:
            plt.close(fig)

    def test_object_limit_keeps_same_ids_across_frames_and_limits_all_sources(self):
        points = []
        for frame, ids in ((1, (30, 10, 20)), (2, (30, 20))):
            for identity in ids:
                for source, offset in (("motion", 0.), ("gt", .2), ("detection", .4)):
                    points.append(dict(frame=frame, source=source,
                                       id=None if source == "detection" else identity + (100 if source == "gt" else 0),
                                       xy=np.array([identity + offset, 0.])))
        selected, ids = viz.limit_objects(points, 2, 1.)
        self.assertEqual(ids, [10, 20])
        self.assertEqual(len([p for p in selected if p["frame"] == 1]), 6)
        self.assertEqual(len([p for p in selected if p["frame"] == 2]), 3)
        self.assertEqual({p["id"] for p in selected if p["frame"] == 2 and p["source"] == "motion"}, {20})
        self.assertEqual({p["id"] for p in selected if p["source"] == "gt"}, {110, 120})
        self.assertEqual({p["context_track_id"] for p in selected}, {10, 20})
        self.assertTrue(all("context_track_id" not in p for p in points))

    def test_context_is_unique_radius_gated_and_competes_with_unselected_tracks(self):
        points = [dict(frame=1, source="motion", id=1, xy=np.array([0., 0.])),
                  dict(frame=1, source="motion", id=2, xy=np.array([1., 0.])),
                  dict(frame=1, source="gt", id=100, xy=np.array([.1, 0.])),
                  dict(frame=1, source="gt", id=101, xy=np.array([100., 0.])),
                  dict(frame=1, source="detection", id=None, xy=np.array([1., 0.]))]
        selected, _ = viz.limit_objects(points, 1, 2.)
        self.assertEqual([p["source"] for p in selected], ["motion", "gt"])
        self.assertEqual(selected[-1]["id"], 100)
        selected, _ = viz.limit_objects(points, 2, 2.)
        self.assertEqual(len([p for p in selected if p["source"] == "gt"]), 1)
        self.assertEqual(len([p for p in selected if p["source"] == "detection"]), 1)
        self.assertEqual(next(p for p in selected if p["source"] == "detection")["context_track_id"], 2)

    def test_limited_cli_exports_only_selected_context_with_metric_radius_conversion(self):
        arguments = self.write_inputs(
            records=[self.record, dict(self.record, track_id=99, predicted_x=100.)],
            gt=[[0, 2, 7, -1, -1, -1, -1, 1, 11, 20, -1],
                [0, 2, 8, -1, -1, -1, -1, 1, 101, 20, -1]],
            detections=[[2, 12, 20, .9, 0], [2, 102, 20, .9, 0]],
        )
        args = viz.build_parser().parse_args(arguments + ["--max-objects", "1", "--context-radius", "3",
                                                         "--meters-per-cell", ".1"])
        files = viz.run(args)
        with (files[0].parent / "plotted_points_total.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["context_track_id"] for r in rows}, {"42"})
        detection = next(r for r in rows if r["source"] == "detection")
        self.assertAlmostEqual(float(detection["context_distance"]), .2)
        settings = json.loads((files[0].parent / "plot_settings_total.json").read_text())
        self.assertEqual(settings["selected_track_ids"], [42])
        self.assertEqual(settings["context_selection"], "greedy_nearest_one_to_one")
        self.assertFalse(plt.get_fignums())

    def test_object_limit_rejects_invalid_counts_and_missing_motion(self):
        arguments = self.write_inputs(records=[])
        for count in ("0", "-1"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive"):
                viz.run(viz.build_parser().parse_args(arguments + ["--max-objects", count]))
        with self.assertRaisesRegex(ValueError, "requires motion"):
            viz.run(viz.build_parser().parse_args(arguments + ["--max-objects", "3"]))

    def test_five_frames_export_one_trajectory_per_format_with_consistent_object_colors(self):
        records, gt, detections = [], [], []
        for frame in (5, 1, 4, 2, 3):
            for track, identity, offset in ((42, 7, 0.), (99, 8, 30.)):
                x, y = offset + frame * 2., offset + frame
                records.append(dict(self.record, frame=frame, track_id=track, predicted_x=x, predicted_y=y))
                gt.append([0, frame, identity, -1, -1, -1, -1, 1, x + .1, y + .2, -1])
                detections.append([frame, x + .2, y + .3, .9, 0])
        args = viz.build_parser().parse_args(self.write_inputs(records, gt, detections) + [
            "--max-objects", "2", "--frames", "1", "2", "3", "4", "5", "--formats", "png", "pdf", "svg"])
        files = viz.run(args)
        self.assertEqual(len(files), 3)
        self.assertTrue(all(path.name.startswith("trajectory_000001_000005_total") for path in files))
        self.assertTrue(all(path.stat().st_size > 100 for path in files))
        output = files[0].parent
        self.assertEqual(list(output.glob("frame_*")), [])
        with (output / "plotted_points_total.csv").open() as stream:
            points = list(csv.DictReader(stream))
        self.assertEqual(len(points), 30)
        self.assertEqual({point["frame"] for point in points}, {str(i) for i in range(1, 6)})
        colors = {}
        for label in ("M42", "M99"):
            obj = [point for point in points if point["object_label"] == label]
            self.assertEqual({point["source"] for point in obj}, {"motion", "gt", "detection"})
            self.assertEqual(len({point["object_color"] for point in obj}), 1)
            colors[label] = obj[0]["object_color"]
            self.assertEqual({point["cov_xx"] for point in obj if point["source"] == "motion"}, {"4.0"})
        self.assertNotEqual(colors["M42"], colors["M99"])
        settings = json.loads((output / "plot_settings_total.json").read_text())
        self.assertEqual(settings["layout"], "trajectory")
        self.assertEqual(settings["object_colors"], colors)
        self.assertFalse(plt.get_fignums())

    def test_trajectories_break_at_missing_frames_and_gt_identity_changes(self):
        raw = []
        for source, frames, identities in (("motion", [3, 1], [1, 1]),
                                           ("detection", [3, 1], [None, None]),
                                           ("gt", [3, 1, 2], [11, 10, 11])):
            for frame, identity in zip(frames, identities):
                raw.append(dict(source=source, id=identity, frame=frame, context_track_id=1,
                                xy=np.array([float(frame), 2.]), ellipse=(4., 2., 30.)))
        raw.append(dict(source="motion", id=2, frame=2, xy=np.array([20., 30.]), ellipse=(6., 4., 40.)))
        points, colors = viz.color_trajectory_points(raw)
        self.assertTrue(all("object_color" not in point for point in raw))
        fig = viz.draw_trajectory(points, [1, 2, 3], "scene", "total", (0, 30), (0, 40), show_ids=True)
        try:
            ax = fig.axes[0]
            connected = [line for line in ax.lines if len(line.get_xdata()) > 1]
            self.assertEqual(len(connected), 1)
            self.assertEqual(connected[0].get_marker(), "x")
            np.testing.assert_array_equal(connected[0].get_xdata(), [2., 3.])
            self.assertEqual({text.get_text() for text in ax.texts}, {"M1", "M2"})
            self.assertEqual(ax.get_aspect(), 1.)
            self.assertFalse(any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines()))
            self.assertEqual(len(ax.patches), 6)
            self.assertEqual({text.get_text() for text in fig.legends[0].get_texts()},
                             {"Motion prediction", "GT (nearby)", "Detection (nearby)", "Total covariance"})
            motion = [p for p in points if p["source"] == "motion"]
            for outline, point in zip(ax.patches[1::2], motion):
                self.assertEqual((outline.width, outline.height, outline.angle), point["ellipse"])
                self.assertEqual(viz.to_hex(outline.get_edgecolor()), colors[point["object_label"]])
                np.testing.assert_array_equal(outline.center, point["xy"])
        finally:
            plt.close(fig)

    def test_unlimited_context_keeps_unpaired_points_and_separate_id_namespaces(self):
        points = [dict(source="motion", frame=1, id=7, xy=np.array([0., 0.])),
                  dict(source="gt", frame=1, id=7, xy=np.array([100., 100.])),
                  dict(source="detection", frame=1, id=None, xy=np.array([200., 200.])),
                  dict(source="detection", frame=1, id=None, xy=np.array([.1, .1]))]
        annotated, colors = viz.color_trajectory_points(viz.assign_context(points, 1.))
        self.assertEqual(len(annotated), 4)
        self.assertEqual([p["object_label"] for p in annotated], ["M7", "G7", None, "M7"])
        self.assertNotEqual(colors["M7"], colors["G7"])
        self.assertEqual(annotated[2]["object_color"], "#909090")
        self.assertEqual(annotated[3]["object_color"], colors["M7"])
        self.assertTrue(all("context_track_id" not in point for point in points))


if __name__ == "__main__":
    unittest.main()
