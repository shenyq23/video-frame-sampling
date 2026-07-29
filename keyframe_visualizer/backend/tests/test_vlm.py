from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.vlm.service import VlmAnswerService


class VlmAnswerServiceTests(unittest.TestCase):
    def test_builds_multiframe_request_and_persists_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            frames_dir = output_dir / "frames"
            frames_dir.mkdir()
            frames = []
            for index in range(5):
                filename = f"{index + 1:04d}.jpg"
                Image.new("RGB", (64, 48), color=(index * 20, 40, 80)).save(
                    frames_dir / filename
                )
                frames.append(
                    {
                        "order": index + 1,
                        "file": f"frames/{filename}",
                        "timestamp_seconds": float(index),
                        "original_frame_index": index * 30,
                        "candidate_order": index + 1,
                    }
                )
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps({"frame_sets": {"selected": {"frames": frames}}}),
                encoding="utf-8",
            )
            profiles = {
                "test-vlm": {
                    "name": "Test VLM",
                    "backend": "mep",
                    "enabled": True,
                    "config": {
                        "elb": "http://example.test/service",
                        "appid_env": "TEST_VLM_APPID",
                        "secret_key_env": "TEST_VLM_SECRET",
                        "b_id": "test",
                        "flow_id": "test",
                        "max_frames": 3,
                        "max_image_dimension": 32,
                    },
                }
            }
            environment = {"TEST_VLM_APPID": "appid", "TEST_VLM_SECRET": "secret"}
            with patch.dict(os.environ, environment, clear=False), patch(
                "app.vlm.service.load_vlm_profiles", return_value=profiles
            ), patch(
                "app.vlm.service.MepVlmClient.execute", return_value="测试回答"
            ) as execute:
                result = VlmAnswerService().answer(
                    job_id="job-1",
                    output_dir=output_dir,
                    manifest_path=manifest_path,
                    frame_set="selected",
                    query="发生了什么？",
                    profile_id="test-vlm",
                )

            self.assertEqual(result["answer"], "测试回答")
            self.assertEqual(result["source_frame_count"], 5)
            self.assertEqual(result["used_frame_count"], 3)
            self.assertTrue(result["frames_limited"])
            self.assertEqual(
                [frame["timestamp_seconds"] for frame in result["used_frames"]],
                [0.0, 2.0, 4.0],
            )
            request_data = execute.call_args.args[0]
            content = request_data["messages"][1]["content"]
            images = [item for item in content if item["type"] == "image_url"]
            self.assertEqual(len(images), 3)
            self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            saved = VlmAnswerService().saved_answer(output_dir, "selected")
            self.assertIsNotNone(saved)
            self.assertEqual(saved["answer"], "测试回答")

