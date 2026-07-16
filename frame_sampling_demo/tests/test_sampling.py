import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling import uniform_frame_indices


class UniformFrameIndicesTests(unittest.TestCase):
    def test_spans_both_video_endpoints(self):
        self.assertEqual(uniform_frame_indices(101, 5), [0, 25, 50, 75, 100])

    def test_short_video_does_not_duplicate_frames(self):
        self.assertEqual(uniform_frame_indices(3, 10), [0, 1, 2])

    def test_single_output_uses_first_frame(self):
        self.assertEqual(uniform_frame_indices(100, 1), [0])

    def test_empty_video(self):
        self.assertEqual(uniform_frame_indices(0, 8), [])

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            uniform_frame_indices(-1, 8)
        with self.assertRaises(ValueError):
            uniform_frame_indices(100, 0)


if __name__ == "__main__":
    unittest.main()

