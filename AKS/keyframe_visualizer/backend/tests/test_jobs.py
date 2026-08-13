from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.jobs import JobManager
from app.storage import JobStore


class FailingAdapter:
    def run(self, **_kwargs):
        raise RuntimeError("feature service failed")


class FakeRegistry:
    def get(self, _algorithm):
        return FailingAdapter()


class JobManagerTests(unittest.TestCase):
    def test_failed_job_releases_video_before_becoming_deletable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            store = JobStore(root / "jobs.db")
            store.create(
                job_id="failed-job",
                algorithm="aks",
                query="query",
                config={"parameters": {}},
                video_path=video,
                original_filename="video.mp4",
                output_dir=root / "run",
            )
            manager = JobManager(store, FakeRegistry())  # type: ignore[arg-type]
            released = Mock()
            with patch("app.jobs.release_video", released), patch(
                "app.jobs.traceback.print_exc"
            ):
                manager._execute("failed-job")

            row = store.get_raw("failed-job")
            self.assertEqual(row["status"], "failed")  # type: ignore[index]
            self.assertGreaterEqual(released.call_count, 1)
            released.assert_any_call(video)
