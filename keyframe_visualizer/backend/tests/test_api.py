from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_module
from app.models import ClipModelStore
from app.storage import JobStore


app = main_module.app


class ApiTests(unittest.TestCase):
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
