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
