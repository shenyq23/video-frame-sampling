from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.algorithms.aks_adapter import AKSAdapter
from app.settings import AKS_ROOT


class FakeVideoSource:
    fps = 10.0
    total_frames = 50
    backend = "fake"

    def get_batch(self, indices):
        return np.stack(
            [np.full((12, 16, 3), int(index * 4), dtype=np.uint8) for index in indices]
        )


class FakeScorer:
    metadata = {"backend": "fake", "model": "fake-clip"}

    def prepare_query(self, query):
        self.query = query

    def score_images(self, images):
        return [float(np.asarray(image).mean()) for image in images]


class AKSAdapterTests(unittest.TestCase):
    def test_exports_selected_uniform_and_candidate_frame_sets(self) -> None:
        if str(AKS_ROOT) not in sys.path:
            sys.path.insert(0, str(AKS_ROOT))
        parameters = {
            "aks_mode": "robust",
            "max_num_frames": 2,
            "candidate_sampling": "interval",
            "sample_interval": 1.0,
            "feature_backend": "clip",
            "feature_profile": None,
            "clip_model_id": None,
            "model_name": "fake",
            "device": "cpu",
            "batch_size": 2,
            "decode_threads": 1,
            "threshold": 0.8,
            "std_threshold": -100.0,
            "max_depth": 2,
            "jpeg_quality": 80,
            "save_uniform_baseline": True,
            "save_candidate_frames": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"fake video")
            with patch(
                "app.algorithms.aks_adapter.open_video", return_value=FakeVideoSource()
            ), patch("feature_backends.create_relevance_scorer", return_value=FakeScorer()):
                manifest_path = AKSAdapter().run(
                    job_id="job",
                    video_path=video,
                    original_filename="video.mp4",
                    query="target",
                    parameters=parameters,
                    output_dir=root / "run",
                    progress=lambda _stage, _progress: None,
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame_sets"]["selected"]["count"], 2)
            self.assertEqual(manifest["frame_sets"]["uniform"]["count"], 2)
            self.assertEqual(manifest["frame_sets"]["candidates"]["count"], 5)
            for frame_set in manifest["frame_sets"].values():
                for frame in frame_set["frames"]:
                    self.assertTrue((root / "run" / frame["file"]).is_file())
