from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.asr.client import SageAsrClient, SageAsrError


class SageAsrClientTests(unittest.TestCase):
    @staticmethod
    def _client() -> SageAsrClient:
        return SageAsrClient(
            {
                "base_url": "http://asr.example",
                "token": "secret",
                "poll_interval": 0,
                "job_timeout": 10,
                "delete_remote": True,
            }
        )

    def test_uploads_polls_downloads_and_marks_result_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            destination = root / "asr.json"
            state = root / "remote_asr_job.json"
            video.write_bytes(b"video")
            client = self._client()

            def download(_: str, path: Path) -> None:
                path.write_text('{"segments": []}', encoding="utf-8")

            with (
                patch.object(client, "_request_job", return_value={"job_id": "job-1"}) as upload,
                patch.object(
                    client,
                    "_read_job",
                    side_effect=[
                        {"status": "running", "stage": "nsp_and_asr"},
                        {"status": "succeeded", "stage": "done"},
                    ],
                ) as read,
                patch.object(client, "_download", side_effect=download) as fetch,
            ):
                job_id = client.generate(
                    video_path=video,
                    destination=destination,
                    state_path=state,
                    progress=lambda *_: None,
                )

            self.assertEqual(job_id, "job-1")
            upload.assert_called_once_with(video)
            self.assertEqual(read.call_count, 2)
            fetch.assert_called_once_with("job-1", destination)
            self.assertTrue(json.loads(state.read_text(encoding="utf-8"))["downloaded"])

    def test_reuses_downloaded_json_without_contacting_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "asr.json"
            state = root / "remote_asr_job.json"
            destination.write_text('{"segments": []}', encoding="utf-8")
            state.write_text(
                json.dumps({"job_id": "job-1", "downloaded": True}), encoding="utf-8"
            )
            client = self._client()
            with (
                patch.object(client, "_request_job") as upload,
                patch.object(client, "_read_job") as read,
            ):
                job_id = client.generate(
                    video_path=root / "missing.mp4",
                    destination=destination,
                    state_path=state,
                    progress=lambda *_: None,
                )
            self.assertEqual(job_id, "job-1")
            upload.assert_not_called()
            read.assert_not_called()

    def test_reports_remote_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            client = self._client()
            with (
                patch.object(client, "_request_job", return_value={"job_id": "job-2"}),
                patch.object(
                    client,
                    "_read_job",
                    return_value={"status": "failed", "error": "nsp_upload_failed"},
                ),
                self.assertRaisesRegex(SageAsrError, "nsp_upload_failed"),
            ):
                client.generate(
                    video_path=video,
                    destination=root / "asr.json",
                    state_path=root / "state.json",
                    progress=lambda *_: None,
                )

    def test_upload_sends_bearer_token_and_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video")
            response = Mock(ok=True, status_code=202)
            response.json.return_value = {"job_id": "job-3"}
            client = self._client()
            with patch("app.asr.client.requests.post", return_value=response) as post:
                payload = client._request_job(video)
            self.assertEqual(payload["job_id"], "job-3")
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer secret")
            self.assertEqual(post.call_args.kwargs["files"]["video"][0], "video.mp4")


if __name__ == "__main__":
    unittest.main()
