"""Regression checks for perception, PID measurements and installed plot entrypoints."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pid_lib.response import step_metrics  # noqa: E402
from trajectory_lib import recorder  # noqa: E402
from vision_lib.calibration import validate_camera_mount  # noqa: E402
from vision_lib.geometry import (  # noqa: E402
    BoxObservation, TRAY_CENTERS, free_tray_slots, persistent_observations,
    sampled_world_points,
)


class ProjectRegressions(unittest.TestCase):
    """Check failures that can otherwise look like successful simulation runs."""

    def test_one_frame_duplicates_cannot_trigger_pick(self):
        box = BoxObservation((.4, -.3, .48), .95, (1, 1, 10, 10))
        self.assertEqual(persistent_observations([[box, box], [], []]), [])
        self.assertEqual(len(persistent_observations([[box, box], [box], []])), 1)

    def test_invalid_intrinsics_cannot_report_empty_table(self):
        for focal_length in (0., -1., np.nan, np.inf):
            matrix = np.eye(3)
            matrix[0, 0] = focal_length
            with self.subTest(fx=focal_length), self.assertRaises(ValueError):
                sampled_world_points(np.ones((16, 16)), matrix, np.eye(4))

    def test_full_tray_has_no_free_slot(self):
        self.assertEqual(free_tray_slots([], TRAY_CENTERS), [])
        self.assertEqual(len(free_tray_slots([], TRAY_CENTERS[:-1])), 1)

    def test_missing_depth_cannot_report_empty_table(self):
        for value in (0., np.nan, np.inf):
            with self.subTest(depth=value), self.assertRaises(ValueError):
                sampled_world_points(np.full((16, 16), value), np.eye(3), np.eye(4))

    def test_camera_mount_matches_mjcf_and_urdf(self):
        package = Path(__file__).resolve().parents[1]
        transform = validate_camera_mount(
            package / 'models/ur5e_robotiq_vision_scene.xml',
            package / 'config/ur5e_vision.urdf.xacro')
        np.testing.assert_allclose(transform[:3, 3], (.068, .0325, -.144))

    def test_pid_rise_and_settling_match_known_first_order_response(self):
        time_s = np.arange(-.2, 3, .001)
        reference = np.where(time_s >= 0, 1., 0.)
        response = np.where(time_s >= 0, 1 - np.exp(-np.maximum(time_s, 0) / .2), 0.)
        for direction in (1., -1.):
            _, metrics = step_metrics(time_s, direction * reference, direction * response,
                                      np.zeros_like(time_s), 0., direction)
            self.assertAlmostEqual(metrics['rise_time_10_90_s'], .2 * np.log(9), delta=.002)
            self.assertAlmostEqual(metrics['settling_time_2_percent_s'],
                                   -.2 * np.log(.02), delta=.002)

    def test_unsettled_response_is_not_given_settling_time(self):
        time_s = np.arange(-.2, 3, .02)
        reference = np.where(time_s >= 0, 1., 0.)
        actual = np.where(time_s >= 0, .8, 0.)
        _, metrics = step_metrics(time_s, reference, actual, np.zeros_like(time_s), 0., 1.)
        self.assertIsNone(metrics['settling_time_2_percent_s'])
        self.assertIsNone(metrics['rise_time_10_90_s'])

    def test_live_plot_uses_renamed_cmake_installed_program(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / 'trajectory_lib/recorder.py'
            module.parent.mkdir()
            module.touch()
            installed_entry = root / 'plot_pid_response_live'
            installed_entry.touch()
            record = recorder.TrajectoryRecorder.__new__(recorder.TrajectoryRecorder)
            record.node = SimpleNamespace(get_logger=lambda: Mock())
            record.output_dir = root
            with patch.object(recorder, '__file__', str(module)), \
                    patch.dict('os.environ', {'DISPLAY': ':test'}), \
                    patch.object(recorder.subprocess, 'Popen') as spawn:
                record._start_live_plot()
            self.assertEqual(spawn.call_args.args[0][1], str(installed_entry))


if __name__ == '__main__':
    unittest.main()
