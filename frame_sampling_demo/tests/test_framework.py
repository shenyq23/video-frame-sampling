from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT / "src"))
sys.path.insert(0, str(DEMO_ROOT.parent / "AKS"))

from frame_sampling_demo import samplers as _samplers  # noqa: F401
from frame_sampling_demo.cli import _read_queries
from frame_sampling_demo.pipeline import run_sampling
from frame_sampling_demo.registry import available_samplers, get_sampler
from frame_sampling_demo.schemas import SamplingRequest, VideoContext, VideoInfo
from frame_sampling_demo.video import candidate_frame_indices, uniform_frame_indices


class FakeBatch:
    def __init__(self, arrays):
        self.arrays = arrays

    def asnumpy(self):
        return self.arrays


class FakeReader:
    def __init__(self, total=100):
        self.frames = np.zeros((total, 8, 8, 3), dtype=np.uint8)
        for index in range(total):
            self.frames[index, :, :, :] = index % 255

    def __len__(self):
        return len(self.frames)

    def get_batch(self, indices):
        return FakeBatch(self.frames[list(indices)])


def fake_context(video_path: Path, cache_dir: Path) -> VideoContext:
    reader = FakeReader()
    return VideoContext(
        reader=reader,
        info=VideoInfo(str(video_path), 10.0, len(reader), len(reader) / 10.0),
        candidate_indices=[],
        candidate_timestamps=[],
        frame_cache_dir=cache_dir,
    )


class FakeFeatureStore:
    def __init__(self, *_args, **_kwargs):
        pass

    def score_queries(self, context, queries):
        count = len(context.candidate_indices)
        result = {}
        for index, query in enumerate(queries):
            values = np.linspace(0.0, 1.0, count)
            if index % 2:
                values = values[::-1]
            result[query] = values.tolist()
        return result


class FrameworkTests(unittest.TestCase):
    def test_algorithms_are_registered(self):
        self.assertEqual(
            ["aks_original", "aks_robust", "clip_threshold", "clip_topk", "uniform"],
            available_samplers(),
        )

    def test_candidate_and_uniform_indices(self):
        self.assertEqual([0, 29, 58], candidate_frame_indices(100, 29.97, "original", 1.0))
        self.assertEqual([0, 30, 60, 90], candidate_frame_indices(100, 29.97, "interval", 1.0))
        self.assertEqual([0, 25, 50, 75, 100], uniform_frame_indices(101, 5))

    def test_fixed_and_adaptive_samplers(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "video.mp4"
            video.touch()
            context = fake_context(video, Path(temp) / "cache")
            context.candidate_indices = list(range(10))
            scores = [value / 10 for value in range(10)]
            fixed_request = SamplingRequest(video, ["q"], "clip_topk", max_frames=3)
            fixed = get_sampler("clip_topk")().select(context, fixed_request, "q", scores)
            self.assertEqual([7, 8, 9], fixed.selected_indices)

            adaptive_request = SamplingRequest(
                video,
                ["q"],
                "clip_threshold",
                score_threshold=0.65,
                min_frames=2,
            )
            adaptive = get_sampler("clip_threshold")().select(
                context, adaptive_request, "q", scores
            )
            self.assertEqual([7, 8, 9], adaptive.selected_indices)

    def test_aks_adapter_uses_shared_core(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "video.mp4"
            video.touch()
            context = fake_context(video, Path(temp) / "cache")
            context.candidate_indices = list(range(100))
            request = SamplingRequest(video, ["q"], "aks_robust", max_frames=16)
            selection = get_sampler("aks_robust")().select(
                context, request, "q", [float(index % 11) for index in range(100)]
            )
            self.assertEqual(16, len(selection.selected_indices))
            self.assertEqual("robust", selection.trace["mode"])

    def test_query_file_supports_lines_and_json(self):
        with tempfile.TemporaryDirectory() as temp:
            line_file = Path(temp) / "queries.txt"
            line_file.write_text("first\nsecond\n", encoding="utf-8")
            json_file = Path(temp) / "queries.json"
            json_file.write_text('["third", "fourth"]', encoding="utf-8")
            self.assertEqual(["inline", "first", "second"], _read_queries(["inline"], str(line_file)))
            self.assertEqual(["third", "fourth"], _read_queries([], str(json_file)))

    def test_uniform_pipeline_exports_frames_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "video.mp4"
            video.touch()
            request = SamplingRequest(
                video_path=video,
                queries=[],
                algorithm="uniform",
                max_frames=5,
                output_dir=root / "outputs",
                cache_dir=root / "cache",
            )
            with patch(
                "frame_sampling_demo.pipeline.open_video",
                side_effect=lambda path, threads, cache: fake_context(path, cache),
            ):
                result = run_sampling(request, progress=lambda *_args: None)
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(5, manifest["groups"][0]["selected_frame_count"])
            self.assertEqual(5, len(list(Path(result["run_dir"]).glob("global/frames/*.jpg"))))

    def test_multi_query_union_has_one_capped_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "video.mp4"
            video.touch()
            request = SamplingRequest(
                video_path=video,
                queries=["first", "second"],
                algorithm="clip_topk",
                max_frames=3,
                multi_query_mode="union",
                output_dir=root / "outputs",
                cache_dir=root / "cache",
            )
            with patch(
                "frame_sampling_demo.pipeline.open_video",
                side_effect=lambda path, threads, cache: fake_context(path, cache),
            ), patch("frame_sampling_demo.pipeline.CLIPFeatureStore", FakeFeatureStore):
                result = run_sampling(request, progress=lambda *_args: None)
            manifest = result["manifest"]
            self.assertEqual(1, len(manifest["groups"]))
            self.assertEqual("union", manifest["groups"][0]["name"])
            self.assertEqual(3, manifest["groups"][0]["selected_frame_count"])


if __name__ == "__main__":
    unittest.main()
