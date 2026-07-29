from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.main as main_module
from app.models import ClipModelStore
from app.storage import JobStore


app = main_module.app


class ApiTests(unittest.TestCase):
    @staticmethod
    def _locked_error() -> PermissionError:
        error = PermissionError("file is in use")
        error.winerror = 32  # type: ignore[attr-defined]
        return error

    def test_health_and_algorithm_catalog(self) -> None:
        with TestClient(app) as client:
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json(), {"status": "ok"})

            algorithms = client.get("/api/algorithms")
            self.assertEqual(algorithms.status_code, 200)
            self.assertEqual(algorithms.json()[0]["id"], "aks")

            credentials = client.get("/api/settings/feature-profiles")
            self.assertEqual(credentials.status_code, 200)
            self.assertIn("pangu-default", credentials.json())

            vlm_credentials = client.get("/api/settings/vlm-profiles")
            self.assertEqual(vlm_credentials.status_code, 200)
            self.assertIn("mep-vlm-default", vlm_credentials.json())

            models = client.get("/api/models/clip")
            self.assertEqual(models.status_code, 200)

    def test_rejects_unsupported_upload_type(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs",
                files={"video": ("notes.txt", b"not a video", "text/plain")},
                data={"config": '{"algorithm":"aks","query":"event","parameters":{}}'},
            )
            self.assertEqual(response.status_code, 415)

    def test_uploads_valid_clip_model_archive(self) -> None:
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("model/config.json", json.dumps({"model_type": "clip"}))
            archive.writestr("model/preprocessor_config.json", "{}")
            archive.writestr("model/tokenizer.json", "{}")
            archive.writestr("model/model.safetensors", b"weights")
        with tempfile.TemporaryDirectory() as directory:
            model_store = ClipModelStore(Path(directory))
            with patch.object(main_module, "clip_models", model_store), TestClient(app) as client:
                response = client.post(
                    "/api/models/clip",
                    files={"archive": ("offline.zip", archive_bytes.getvalue(), "application/zip")},
                    data={"name": "Offline model"},
                )
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.json()["name"], "Offline model")
                self.assertEqual(len(model_store.list()), 1)

    def test_creates_and_reads_vlm_answer_for_succeeded_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            output_dir = root / "run"
            output_dir.mkdir()
            manifest = output_dir / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            store = JobStore(root / "jobs.db")
            store.create(
                job_id="vlm-job",
                algorithm="aks",
                query="原始问题",
                config={"parameters": {}},
                video_path=video,
                original_filename="video.mp4",
                output_dir=output_dir,
            )
            store.update(
                "vlm-job",
                status="succeeded",
                stage="done",
                progress=1.0,
                manifest_path=str(manifest),
            )
            result = {
                "schema_version": "1.0",
                "job_id": "vlm-job",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "profile_id": "mep-vlm-default",
                "profile_name": "MEP VLM",
                "frame_set": "selected",
                "frame_set_name": "抽帧算法选出的帧",
                "query": "发生了什么？",
                "answer": "测试回答",
                "source_frame_count": 2,
                "used_frame_count": 2,
                "frames_limited": False,
                "used_frames": [],
            }
            with patch.object(main_module, "store", store), patch.object(
                main_module.vlm_answers, "answer", return_value=result
            ) as answer, patch.object(
                main_module.vlm_answers, "saved_answer", return_value=result
            ):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/jobs/vlm-job/vlm-answer",
                        json={
                            "frame_set": "selected",
                            "query": "发生了什么？",
                            "vlm_profile": "mep-vlm-default",
                        },
                    )
                    saved = client.get(
                        "/api/jobs/vlm-job/vlm-answer?frame_set=selected"
                    )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "测试回答")
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(answer.call_args.kwargs["frame_set"], "selected")

    def test_video_endpoint_supports_byte_ranges_for_seeking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(bytes(range(100)))
            store = JobStore(root / "jobs.db")
            store.create(
                job_id="seek-job",
                algorithm="aks",
                query="query",
                config={"parameters": {"feature_backend": "clip"}},
                video_path=video,
                original_filename="video.mp4",
                output_dir=root / "run",
            )
            with patch.object(main_module, "store", store), TestClient(app) as client:
                response = client.get(
                    "/api/jobs/seek-job/video", headers={"Range": "bytes=10-19"}
                )
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, bytes(range(10, 20)))
            self.assertEqual(response.headers["accept-ranges"], "bytes")

    def test_delete_terminal_job_removes_all_owned_data_and_database_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uploads = root / "uploads"
            runs = root / "runs"
            uploads.mkdir()
            runs.mkdir()
            video = uploads / "delete-job_video.mp4"
            video.write_bytes(b"video")
            output_dir = runs / "delete-job"
            (output_dir / "frames").mkdir(parents=True)
            (output_dir / "frames" / "001.jpg").write_bytes(b"image")
            (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
            store = JobStore(root / "jobs.db")
            store.create(
                job_id="delete-job",
                algorithm="aks",
                query="query",
                config={"parameters": {}},
                video_path=video,
                original_filename="video.mp4",
                output_dir=output_dir,
            )
            store.update("delete-job", status="succeeded", stage="done", progress=1.0)

            with patch.object(main_module, "store", store), patch.object(
                main_module, "UPLOAD_DIR", uploads
            ), patch.object(main_module, "RUNS_DIR", runs):
                with TestClient(app) as client:
                    response = client.delete("/api/jobs/delete-job")

            self.assertEqual(response.status_code, 204)
            self.assertFalse(video.exists())
            self.assertFalse(output_dir.exists())
            self.assertIsNone(store.get_raw("delete-job"))

    def test_delete_rejects_non_terminal_job_without_removing_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uploads = root / "uploads"
            runs = root / "runs"
            uploads.mkdir()
            runs.mkdir()
            video = uploads / "running-job_video.mp4"
            video.write_bytes(b"video")
            output_dir = runs / "running-job"
            output_dir.mkdir()
            store = JobStore(root / "jobs.db")
            store.create(
                job_id="running-job",
                algorithm="aks",
                query="query",
                config={"parameters": {}},
                video_path=video,
                original_filename="video.mp4",
                output_dir=output_dir,
            )
            store.update("running-job", status="running", stage="running", progress=0.5)

            with patch.object(main_module, "store", store), patch.object(
                main_module, "UPLOAD_DIR", uploads
            ), patch.object(main_module, "RUNS_DIR", runs):
                with TestClient(app) as client:
                    response = client.delete("/api/jobs/running-job")

            self.assertEqual(response.status_code, 409)
            self.assertTrue(video.exists())
            self.assertTrue(output_dir.exists())
            self.assertIsNotNone(store.get_raw("running-job"))

    def test_windows_file_lock_is_retried_before_delete_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"video")
            original_unlink = Path.unlink
            calls = 0

            def flaky_unlink(target, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise self._locked_error()
                return original_unlink(target, *args, **kwargs)

            with patch.object(
                Path, "unlink", autospec=True, side_effect=flaky_unlink
            ), patch.object(main_module.time, "sleep"):
                main_module._delete_with_retries(
                    path, is_directory=False, label="上传视频"
                )

            self.assertEqual(calls, 3)
            self.assertFalse(path.exists())

    def test_persistent_windows_file_lock_returns_423_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"video")
            with patch.object(
                Path, "unlink", autospec=True, side_effect=self._locked_error()
            ), patch.object(main_module.time, "sleep"):
                with self.assertRaises(HTTPException) as raised:
                    main_module._delete_with_retries(
                        path, is_directory=False, label="上传视频"
                    )

            self.assertEqual(raised.exception.status_code, 423)
            self.assertIn("文件仍被浏览器或其他程序占用", raised.exception.detail)
            self.assertTrue(path.exists())
