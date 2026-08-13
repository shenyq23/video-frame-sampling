from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import app.video_reader as video_reader
from app.video_reader import OpenCVVideoSource


class VideoReaderTests(unittest.TestCase):
    def test_opencv_fallback_reads_requested_frames_as_rgb(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24)
            )
            if not writer.isOpened():
                self.skipTest("The local OpenCV build cannot encode mp4v")
            for value in (0, 80, 160):
                writer.write(np.full((24, 32, 3), (0, 0, value), dtype=np.uint8))
            writer.release()

            source = OpenCVVideoSource(path, 1)
            frames = source.get_batch([0, 2])
            self.assertEqual(source.total_frames, 3)
            self.assertAlmostEqual(source.fps, 5.0)
            self.assertEqual(frames.shape, (2, 24, 32, 3))
            self.assertGreater(float(frames[1, :, :, 0].mean()), 120)
            source.close()
            self.assertIsNone(source.capture)

    def test_release_video_closes_registered_native_source(self) -> None:
        class FakeSource:
            closed = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            source = FakeSource()
            key = str(path.resolve())
            with video_reader._ACTIVE_SOURCES_LOCK:
                video_reader._ACTIVE_SOURCES[key] = source  # type: ignore[assignment]
            video_reader.release_video(path)
            self.assertTrue(source.closed)
            self.assertNotIn(key, video_reader._ACTIVE_SOURCES)
