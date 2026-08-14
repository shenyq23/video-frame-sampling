from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.algorithms.sage_adapter import SAGEAdapter


class SAGEAdapterTests(unittest.TestCase):
    def test_prepare_session_supports_remote_upload_and_none(self) -> None:
        for mode in ("remote", "upload", "none"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                session_dir = root / "session"
                source = session_dir / "source"
                source.mkdir(parents=True)
                video = source / "video.mp4"
                video.write_bytes(b"video")
                if mode == "upload":
                    (source / "asr.json").write_text('{"segments": []}', encoding="utf-8")

                adapter = SAGEAdapter()

                def generate(**kwargs):
                    kwargs["destination"].write_text('{"segments": []}', encoding="utf-8")
                    return "remote-job"

                with (
                    patch.object(adapter, "_ensure_source"),
                    patch.object(adapter, "_video_metadata", return_value=(25.0, 250, 1280, 720)),
                    patch.object(adapter, "_sha256", return_value="digest"),
                    patch.object(adapter, "_load_asr", return_value=[]),
                    patch("app.algorithms.sage_adapter.SageAsrClient") as client_class,
                    patch(
                        "app.algorithms.sage_adapter.sage_asr_settings",
                        return_value={"base_url": "http://asr", "token": "x", "delete_remote": True},
                    ),
                ):
                    client_class.return_value.generate.side_effect = generate
                    metadata_path = adapter.prepare_session(
                        session_id="session-1",
                        video_path=video,
                        original_filename="video.mp4",
                        parameters={"asr_mode": mode, "device": "cpu"},
                        session_dir=session_dir,
                        progress=lambda *_: None,
                    )

                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["asr"]["mode"], mode)
                self.assertTrue((session_dir / "preprocess" / "asr.json").is_file())
                if mode == "remote":
                    self.assertEqual(metadata["asr"]["remote_job_id"], "remote-job")
                    client_class.return_value.delete.assert_called_once_with("remote-job")
                else:
                    client_class.return_value.generate.assert_not_called()

    def test_frame_index_at_clamps_video_boundaries(self) -> None:
        self.assertEqual(SAGEAdapter._frame_index_at(-1.0, 25.0, 100), 0)
        self.assertEqual(SAGEAdapter._frame_index_at(4.0, 25.0, 100), 99)
        self.assertEqual(SAGEAdapter._frame_index_at(1.0, 25.0, 100), 25)

    def test_run_from_session_exports_standard_manifest(self) -> None:
        adapter = SAGEAdapter()
        adapter._ensure_source()
        from sage_frame.models import Candidate, Keyframe, VideoSegment

        candidate = Candidate(
            time=1.0,
            segment_index=0,
            feature=(1.0,),
            relevance=0.8,
            change=0.25,
            base_score=0.6,
        )
        keyframe = Keyframe(
            time=1.0,
            frame_index=10,
            segment_index=0,
            score=1.2,
            relevance=0.8,
            change=0.25,
        )

        class FakeProvider:
            duration = 1.0
            fps = 10.0
            frame_count = 10

            def __init__(self, *_):
                pass

        class FakeSelector:
            def __init__(self, *_args, **_kwargs):
                self.last_candidates = []
                self.last_segments = []

            def select(self, **_kwargs):
                self.last_candidates = [candidate]
                self.last_segments = [VideoSegment(0, 0.0, 1.0)]
                return [keyframe]

        def export(_video, specs, _output, relative_dir, _fps):
            return [
                {**spec, "order": order, "file": f"{relative_dir}/{order:04d}.jpg"}
                for order, spec in enumerate(specs, 1)
            ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_dir = root / "session"
            (session_dir / "source").mkdir(parents=True)
            (session_dir / "preprocess").mkdir()
            video = session_dir / "source" / "video.mp4"
            video.write_bytes(b"video")
            (session_dir / "preprocess" / "asr.json").write_text(
                '{"segments": []}', encoding="utf-8"
            )
            metadata = session_dir / "preprocess" / "metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "session_id": "session-1",
                        "video": {
                            "filename": "video.mp4",
                            "fps": 10.0,
                            "duration_seconds": 1.0,
                            "total_frames": 10,
                        },
                        "asr": {"mode": "none", "segment_count": 0},
                        "parameters": {"asr_mode": "none", "device": "cpu"},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "run"
            with (
                patch("app.algorithms.sage_adapter.require_torch"),
                patch("sage_frame.SAGESelector", FakeSelector),
                patch("sage_frame.model_adapters.OpenCVCLIPFrameProvider", FakeProvider),
                patch.object(adapter, "_usable_file", return_value=True),
                patch.object(adapter, "_clip_backend", return_value=object()),
                patch.object(adapter, "_load_asr", return_value=[]),
                patch.object(adapter, "_export_frames", side_effect=export),
            ):
                manifest_path = adapter.run_from_session(
                    job_id="job-1",
                    session_dir=session_dir,
                    metadata_path=metadata,
                    query="举起奖杯",
                    parameters={
                        "asr_mode": "none",
                        "device": "cpu",
                        "budget": 1,
                        "save_uniform_baseline": True,
                        "save_candidate_frames": True,
                    },
                    output_dir=output,
                    progress=lambda *_: None,
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["algorithm"]["id"], "sage")
            self.assertEqual(manifest["summary"]["selected_keyframes"], 1)
            self.assertEqual(manifest["summary"]["candidate_frames"], 1)
            self.assertEqual(manifest["selected_frames"][0]["original_frame_index"], 9)
            self.assertEqual(manifest["candidates"][0]["original_frame_index"], 9)
            self.assertEqual(manifest["selected_frames"][0]["sage_score"], 1.2)
            self.assertTrue(manifest["frame_sets"]["uniform"]["available"])
            self.assertTrue(manifest["frame_sets"]["candidates"]["available"])


if __name__ == "__main__":
    unittest.main()
