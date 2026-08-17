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
            self.assertGreaterEqual(result["generation_duration_seconds"], 0)
            self.assertEqual(result["source_frame_count"], 5)
            self.assertEqual(result["used_frame_count"], 3)
            self.assertTrue(result["frames_limited"])
            self.assertEqual(
                [frame["timestamp_seconds"] for frame in result["used_frames"]],
                [0.0, 2.0, 4.0],
            )
            request_data = execute.call_args.args[0]
            content = request_data["messages"][1]["content"]
            user_text = "\n".join(
                item["text"] for item in content if item["type"] == "text"
            )
            self.assertNotIn("关键帧", user_text)
            self.assertNotIn("抽帧", user_text)
            self.assertNotIn("候选帧", user_text)
            self.assertIn("直接回答用户问题", user_text)
            system_prompt = request_data["messages"][0]["content"]
            self.assertIn("不要提及", system_prompt)
            self.assertIn("只输出最终答案", system_prompt)
            images = [item for item in content if item["type"] == "image_url"]
            self.assertEqual(len(images), 3)
            self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            saved = VlmAnswerService().saved_answer(output_dir, "selected")
            self.assertIsNotNone(saved)
            self.assertEqual(saved["answer"], "测试回答")

    def test_vlm_can_read_frames_from_session_media_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "run"
            output_dir.mkdir()
            session_dir = root / "session"
            frame_dir = session_dir / "preprocess" / "candidate_frames"
            frame_dir.mkdir(parents=True)
            Image.new("RGB", (64, 48), color=(20, 40, 80)).save(frame_dir / "0001.jpg")
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "frame_sets": {
                            "selected": {
                                "frames": [
                                    {
                                        "order": 1,
                                        "file": "preprocess/candidate_frames/0001.jpg",
                                        "timestamp_seconds": 0.0,
                                    }
                                ]
                            }
                        }
                    }
                ),
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
                    },
                }
            }
            environment = {"TEST_VLM_APPID": "appid", "TEST_VLM_SECRET": "secret"}
            with patch.dict(os.environ, environment, clear=False), patch(
                "app.vlm.service.load_vlm_profiles", return_value=profiles
            ), patch(
                "app.vlm.service.MepVlmClient.execute", return_value="测试回答"
            ):
                result = VlmAnswerService().answer(
                    job_id="job-1",
                    output_dir=output_dir,
                    media_roots=[session_dir],
                    manifest_path=manifest_path,
                    frame_set="selected",
                    query="发生了什么？",
                    profile_id="test-vlm",
                )

            self.assertEqual(result["answer"], "测试回答")
            self.assertEqual(result["used_frame_count"], 1)

    def test_sage_vlm_request_includes_timestamped_asr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "run"
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            Image.new("RGB", (64, 48), color=(20, 40, 80)).save(frames_dir / "0001.jpg")
            session_dir = root / "session"
            preprocess_dir = session_dir / "preprocess"
            preprocess_dir.mkdir(parents=True)
            (preprocess_dir / "asr.json").write_text(
                json.dumps(
                    {
                        "segments": [
                            {"start": 2.5, "end": 4.0, "text": "第二句话"},
                            {"start_time": 0, "end_time": 1.25, "text": "第一句话"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "algorithm": {"id": "sage"},
                        "asr": {"mode": "remote", "segment_count": 2},
                        "frame_sets": {
                            "selected": {
                                "frames": [
                                    {
                                        "order": 1,
                                        "file": "frames/0001.jpg",
                                        "timestamp_seconds": 3.0,
                                    }
                                ]
                            }
                        },
                    }
                ),
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
                    },
                }
            }
            environment = {"TEST_VLM_APPID": "appid", "TEST_VLM_SECRET": "secret"}
            with patch.dict(os.environ, environment, clear=False), patch(
                "app.vlm.service.load_vlm_profiles", return_value=profiles
            ), patch(
                "app.vlm.service.MepVlmClient.execute", return_value="联合回答"
            ) as execute:
                result = VlmAnswerService().answer(
                    job_id="sage-job",
                    output_dir=output_dir,
                    media_roots=[session_dir],
                    manifest_path=manifest_path,
                    frame_set="selected",
                    query="说了什么？",
                    profile_id="test-vlm",
                )

            content = execute.call_args.args[0]["messages"][1]["content"]
            prompt = content[0]["text"]
            self.assertLess(prompt.index("第一句话"), prompt.index("第二句话"))
            self.assertIn("[0.000-1.250 秒] 第一句话", prompt)
            self.assertIn("请综合语音和视觉信息", prompt)
            self.assertEqual(result["asr_segment_count"], 2)
            self.assertEqual(result["asr_segments_used"], 2)
            self.assertTrue(result["asr_included"])
            self.assertFalse(result["asr_truncated"])

    def test_sage_none_and_non_sage_requests_do_not_include_asr(self) -> None:
        segment = [{"start": 0.0, "end": 1.0, "text": "不应读取"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preprocess_dir = root / "preprocess"
            preprocess_dir.mkdir()
            (preprocess_dir / "asr.json").write_text(
                json.dumps({"segments": segment}, ensure_ascii=False), encoding="utf-8"
            )
            service = VlmAnswerService()
            self.assertEqual(
                service._asr_segments(
                    {"algorithm": {"id": "sage"}, "asr": {"mode": "none"}}, [root]
                ),
                [],
            )
            self.assertEqual(
                service._asr_segments(
                    {"algorithm": {"id": "aks"}, "asr": {"mode": "upload"}}, [root]
                ),
                [],
            )

    def test_asr_is_limited_at_segment_boundaries(self) -> None:
        text, used, truncated = VlmAnswerService._format_asr(
            [
                {"start": 0.0, "end": 1.0, "text": "短句"},
                {"start": 1.0, "end": 2.0, "text": "下一句"},
            ],
            22,
        )
        self.assertIn("短句", text)
        self.assertNotIn("下一句", text)
        self.assertEqual(used, 1)
        self.assertTrue(truncated)
