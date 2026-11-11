import sys
import unittest
from unittest.mock import patch

import smart_parking_simple
import smart_parking_auto
import smart_parking_lines
import smart_parking_detection
import smart_parking_lines_video

class TestSimpleParkingCLI(unittest.TestCase):

    def test_default_output_filename(self):
        with (
            patch.object(sys, "argv", ["smart_parking_simple.py", "parking_lot.jpg"]),
            patch("smart_parking_simple.SimpleParkingDetector") as mock_detector,
            patch("builtins.print"),
        ):
            smart_parking_simple.main()
            mock_detector.return_value.process_image.assert_called_once_with(
                "parking_lot.jpg",
                output_path="result.jpg",
                show=True
            )

    def test_custom_output_filename(self):
        with (
            patch.object(sys, "argv", ["smart_parking_simple.py", "parking_lot.jpg", "--output", "custom.jpg"]),
            patch("smart_parking_simple.SimpleParkingDetector") as mock_detector,
            patch("builtins.print"),
        ):
            smart_parking_simple.main()
            mock_detector.return_value.process_image.assert_called_once_with(
                "parking_lot.jpg",
                output_path="custom.jpg",
                show=True
            )

class TestAutoParkingCLI(unittest.TestCase):

    def test_default_output_filename(self):
        with (
            patch.object(sys, "argv", ["smart_parking_auto.py", "parking_lot.jpg"]),
            patch("smart_parking_auto.AutoParkingDetector") as mock_detector,
            patch("builtins.print"),
        ):
            mock_detector.return_value.process_image.return_value = (None, [])
            smart_parking_auto.main()
            mock_detector.return_value.process_image.assert_called_once_with(
                "parking_lot.jpg",
                output_path="detection_result.jpg",
                show=True
            )

    def test_custom_output_filename(self):
        with (
            patch.object(sys, "argv", ["smart_parking_auto.py", "parking_lot.jpg", "--output", "custom.jpg"]),
            patch("smart_parking_auto.AutoParkingDetector") as mock_detector,
            patch("builtins.print"),
        ):
            mock_detector.return_value.process_image.return_value = (None, [])
            smart_parking_auto.main()
            mock_detector.return_value.process_image.assert_called_once_with(
                "parking_lot.jpg",
                output_path="custom.jpg",
                show=True
            )

class TestLineParkingCLI(unittest.TestCase):

    def test_default_output_filename(self):
        with (
            patch.object(sys, "argv", ["smart_parking_lines.py", "parking_lot.jpg"]),
            patch("smart_parking_lines.LineParkingDetector") as mock_detector,
            patch("builtins.print"),
        ):
            mock_detector.return_value.process_image.return_value = (None, [])
            smart_parking_lines.main()
            mock_detector.return_value.process_image.assert_called_once_with(
                "parking_lot.jpg",
                output_path="detection_result.jpg",
                show=True,
                debug=False
            )

    def test_custom_output_filename(self):
        with (
            patch.object(sys, "argv", ["smart_parking_lines.py", "parking_lot.jpg", "--output", "custom.jpg"]),
            patch("smart_parking_lines.LineParkingDetector") as mock_detector,
            patch("builtins.print"),
        ):
            mock_detector.return_value.process_image.return_value = (None, [])
            smart_parking_lines.main()
            mock_detector.return_value.process_image.assert_called_once_with(
                "parking_lot.jpg",
                output_path="custom.jpg",
                show=True,
                debug=False
            )

    def test_debug_flag_enabled(self):
        with (
            patch.object(sys, "argv", ["smart_parking_lines.py", "parking_lot.jpg", "--debug"]),
            patch("smart_parking_lines.LineParkingDetector") as mock_detector,
            patch("builtins.print"),
        ):
            mock_detector.return_value.process_image.return_value = (None, [])
            smart_parking_lines.main()
            mock_detector.return_value.process_image.assert_called_once_with(
                "parking_lot.jpg",
                output_path="detection_result.jpg",
                show=True,
                debug=True
            )

class TestDetectionParkingCLI(unittest.TestCase):

    def test_default_output_filename(self):
        with (
            patch.object(sys, "argv", ["smart_parking_detection.py", "parking_lot.jpg"]),
            patch("smart_parking_detection.SmartParkingV2") as mock_detector,
            patch("builtins.print"),
        ):
            smart_parking_detection.main()
            mock_detector.return_value.detect_vehicles.assert_called_once_with(
                "parking_lot.jpg",
                output_path="detection_result.jpg",
                show=True
            )

    def test_custom_output_filename(self):
        with (
            patch.object(sys, "argv", ["smart_parking_detection.py", "parking_lot.jpg", "--output", "custom.jpg"]),
            patch("smart_parking_detection.SmartParkingV2") as mock_detector,
            patch("builtins.print"),
        ):
            smart_parking_detection.main()
            mock_detector.return_value.detect_vehicles.assert_called_once_with(
                "parking_lot.jpg",
                output_path="custom.jpg",
                show=True
            )

    def test_config_flag_enabled(self):
        with (
            patch.object(sys, "argv", [
                "smart_parking_detection.py",
                "parking_lot.jpg",
                "--config",
                "slots.json",
                "--output",
                "configured.jpg"
            ]),
            patch("smart_parking_detection.SmartParkingV2") as mock_detector,
            patch("builtins.print"),
        ):
            smart_parking_detection.main()
            mock_detector.return_value.detect_with_slots.assert_called_once_with(
                "parking_lot.jpg",
                "slots.json",
                output_path="configured.jpg"
            )

    def test_define_flag_enabled(self):
        with (
            patch.object(sys, "argv", [
                "smart_parking_detection.py",
                "parking_lot.jpg",
                "--define"
            ]),
            patch("smart_parking_detection.SmartParkingV2") as mock_detector,
            patch("builtins.print"),
        ):
            smart_parking_detection.main()
            mock_detector.return_value.define_slots_interactive.assert_called_once_with(
                "parking_lot.jpg"
            )
            mock_detector.return_value.detect_vehicles.assert_not_called()
            mock_detector.return_value.detect_with_slots.assert_not_called()

class TestVideoParkingCLI(unittest.TestCase):

    def test_custom_output_prefix(self):
        with (
            patch.object(sys, "argv", [
                "smart_parking_lines_video.py",
                "CarPark.mp4",
                "--max_frames", "1",
                "--output", "experiment"
            ]),
            patch("smart_parking_lines_video.os.path.exists", return_value=True),
            patch("smart_parking_lines_video.cv2.VideoCapture") as mock_capture,
            patch("smart_parking_lines_video.cv2.imwrite") as mock_imwrite,
            patch("smart_parking_lines_video.LineParkingDetector") as mock_detector,
            patch("builtins.print"),
        ):
            mock_capture.return_value.isOpened.return_value = True
            mock_capture.return_value.read.side_effect = [(True, "fake_frame"), (False, None)]
            mock_detector.return_value.process_image.return_value = (None, [])
            smart_parking_lines_video.main()

            # Check that the extracted frame uses our custom prefix
            mock_imwrite.assert_called_once_with(
                "experiment_frame_00000.jpg",
                "fake_frame"
            )

            # Check that the detector receives the correct output filename
            mock_detector.return_value.process_image.assert_called_once_with(
                "experiment_frame_00000.jpg",
                output_path="experiment_detection_00000.jpg",
                show=False,
                debug=False
            )

            # Check that the video capture was released
            mock_capture.return_value.release.assert_called_once()