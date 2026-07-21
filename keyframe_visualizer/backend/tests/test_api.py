from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app


class ApiTests(unittest.TestCase):
    def test_health_and_algorithm_catalog(self) -> None:
        with TestClient(app) as client:
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json(), {"status": "ok"})

            algorithms = client.get("/api/algorithms")
            self.assertEqual(algorithms.status_code, 200)
            self.assertEqual(algorithms.json()[0]["id"], "aks")

    def test_rejects_unsupported_upload_type(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs",
                files={"video": ("notes.txt", b"not a video", "text/plain")},
                data={"config": '{"algorithm":"aks","query":"event","parameters":{}}'},
            )
            self.assertEqual(response.status_code, 415)

