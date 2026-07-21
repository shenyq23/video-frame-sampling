from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.storage import JobStore


class StorageTests(unittest.TestCase):
    def test_job_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            store = JobStore(tmp_path / "jobs.db")
            store.create(
                job_id="job-1",
                algorithm="aks",
                query="query",
                config={"parameters": {}},
                video_path=tmp_path / "video.mp4",
                original_filename="video.mp4",
                output_dir=tmp_path / "run",
            )
            queued = store.get_raw("job-1")
            self.assertIsNotNone(queued)
            self.assertEqual(queued["status"], "queued")  # type: ignore[index]

            store.update("job-1", status="running", stage="scoring", progress=0.4)
            running = store.to_record(store.get_raw("job-1"))  # type: ignore[arg-type]
            self.assertEqual(running.status, "running")
            self.assertEqual(running.progress, 0.4)
            self.assertEqual(store.pending_ids(), ["job-1"])

            manifest = tmp_path / "run" / "manifest.json"
            store.update(
                "job-1",
                status="succeeded",
                stage="done",
                progress=1.0,
                manifest_path=str(manifest),
            )
            complete = store.to_record(store.get_raw("job-1"))  # type: ignore[arg-type]
            self.assertTrue(complete.manifest_available)
            self.assertEqual(store.pending_ids(), [])

