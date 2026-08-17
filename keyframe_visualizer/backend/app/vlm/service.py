from __future__ import annotations

import base64
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

from ..settings import load_vlm_profiles
from .client import MepVlmClient, VlmRequestError


FRAME_SET_NAMES = {
    "selected": "抽帧算法选出的帧",
    "uniform": "同数量均匀抽取帧",
    "candidates": "候选帧",
}

SYSTEM_PROMPT = """你是一名视频问答助手。请综合提供的语音转写（如有）和按照视频时间顺序提供的视觉信息，直接、自然地回答用户问题。
只输出最终答案，不要描述分析过程，不要提及“关键帧”“抽帧”“图片”“画面编号”“输入内容”或你获取信息的方式。
不要把未观察到的内容当作事实；如果现有信息不足以回答，请简洁说明无法确定，不要猜测。
除非用户明确要求提供时间依据，否则不要主动引用画面编号或时间戳。"""


class VlmAnswerService:
    def answer(
        self,
        *,
        job_id: str,
        output_dir: Path,
        manifest_path: Path,
        frame_set: str,
        query: str,
        profile_id: str,
        media_roots: Sequence[Path] | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        profile = self._profile(profile_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = self._frames(manifest, frame_set)
        if not frames:
            raise VlmRequestError("所选帧集合为空或该任务没有保存这组帧")

        config = dict(profile.get("config", {}))
        max_frames = max(1, int(config.get("max_frames", 32)))
        used_frames = self._limit_frames(frames, max_frames)
        asr_segments = self._asr_segments(manifest, media_roots)
        max_asr_characters = max(0, int(config.get("max_asr_characters", 40000)))
        asr_text, asr_segments_used, asr_truncated = self._format_asr(
            asr_segments, max_asr_characters
        )
        content = self._build_content(
            output_dir=output_dir,
            media_roots=media_roots,
            frames=used_frames,
            query=query,
            asr_text=asr_text,
            max_dimension=max(64, int(config.get("max_image_dimension", 1280))),
            jpeg_quality=min(100, max(1, int(config.get("jpeg_quality", 85)))),
        )
        request_data = {
            "param": {
                "temperature": float(config.get("temperature", 0.1)),
                "max_tokens": int(config.get("max_tokens", 4096)),
                "frequency_penalty": float(config.get("frequency_penalty", 0.0)),
                "top_p": float(config.get("top_p", 0.95)),
            },
            "messages": [
                {"role": "system", "content": str(config.get("system_prompt", SYSTEM_PROMPT))},
                {"role": "user", "content": content},
            ],
        }
        appid = self._required_environment(config, "appid_env")
        secret_key = self._required_environment(config, "secret_key_env")
        answer = MepVlmClient(config, appid, secret_key).execute(request_data)
        generation_duration_seconds = time.perf_counter() - started_at

        result = {
            "schema_version": "1.0",
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile_id": profile_id,
            "profile_name": str(profile.get("name", profile_id)),
            "frame_set": frame_set,
            "frame_set_name": FRAME_SET_NAMES[frame_set],
            "query": query,
            "answer": answer,
            "generation_duration_seconds": generation_duration_seconds,
            "source_frame_count": len(frames),
            "used_frame_count": len(used_frames),
            "frames_limited": len(used_frames) < len(frames),
            "asr_included": bool(asr_text),
            "asr_segment_count": len(asr_segments),
            "asr_segments_used": asr_segments_used,
            "asr_truncated": asr_truncated,
            "used_frames": [
                {
                    "order": frame.get("order"),
                    "file": frame["file"],
                    "timestamp_seconds": frame.get("timestamp_seconds"),
                    "original_frame_index": frame.get("original_frame_index"),
                    "candidate_order": frame.get("candidate_order"),
                }
                for frame in used_frames
            ],
        }
        results_dir = output_dir / "vlm_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"{frame_set}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def saved_answer(
        self, output_dir: Path, frame_set: str
    ) -> Optional[dict[str, Any]]:
        path = output_dir / "vlm_results" / f"{frame_set}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _profile(profile_id: str) -> dict[str, Any]:
        profile = load_vlm_profiles().get(profile_id)
        if not profile:
            raise VlmRequestError(f"未知 VLM 配置：{profile_id}")
        if not profile.get("enabled", True):
            raise VlmRequestError(f"VLM 配置已禁用：{profile_id}")
        if str(profile.get("backend", "mep")) != "mep":
            raise VlmRequestError("当前仅支持 MEP VLM 配置")
        return profile

    @staticmethod
    def _required_environment(config: dict[str, Any], field: str) -> str:
        name = str(config.get(field, "")).strip()
        if not name:
            raise VlmRequestError(f"VLM 配置缺少 {field}")
        value = os.getenv(name, "").strip()
        if not value:
            raise VlmRequestError(f"环境变量 {name} 尚未配置")
        return value

    @staticmethod
    def _frames(manifest: dict[str, Any], frame_set: str) -> list[dict[str, Any]]:
        frame_sets = manifest.get("frame_sets", {})
        entry = frame_sets.get(frame_set, {}) if isinstance(frame_sets, dict) else {}
        frames = entry.get("frames", []) if isinstance(entry, dict) else []
        if not frames:
            fallback = {
                "selected": manifest.get("selected_frames") or manifest.get("keyframes"),
                "uniform": manifest.get("uniform_frames"),
                "candidates": manifest.get("candidates"),
            }[frame_set]
            frames = fallback or []
        usable = [frame for frame in frames if isinstance(frame, dict) and frame.get("file")]
        return sorted(
            usable,
            key=lambda frame: (
                float(frame.get("timestamp_seconds", 0)),
                int(frame.get("order", 0)),
            ),
        )

    @staticmethod
    def _limit_frames(frames: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if len(frames) <= limit:
            return frames
        if limit == 1:
            return [frames[len(frames) // 2]]
        indices = [round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)]
        return [frames[index] for index in indices]

    @staticmethod
    def _asr_segments(
        manifest: dict[str, Any], media_roots: Sequence[Path] | None
    ) -> list[dict[str, Any]]:
        algorithm = manifest.get("algorithm", {})
        if not isinstance(algorithm, dict) or algorithm.get("id") != "sage":
            return []
        asr_metadata = manifest.get("asr", {})
        if (
            not isinstance(asr_metadata, dict)
            or asr_metadata.get("mode") not in {"remote", "upload"}
        ):
            return []

        for root in media_roots or []:
            asr_path = root / "preprocess" / "asr.json"
            if not asr_path.is_file():
                continue
            try:
                payload = json.loads(asr_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise VlmRequestError(f"无法读取 SAGE ASR：{error}") from error
            records = payload.get("segments", []) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise VlmRequestError("SAGE ASR JSON 的 segments 必须是数组")
            segments: list[dict[str, Any]] = []
            for index, item in enumerate(records):
                if not isinstance(item, dict):
                    raise VlmRequestError(f"SAGE ASR 第 {index + 1} 段不是对象")
                start = item.get("start", item.get("start_time"))
                end = item.get("end", item.get("end_time"))
                text = str(item.get("text", "")).strip()
                if start is None or end is None or not text:
                    continue
                try:
                    segments.append(
                        {"start": float(start), "end": float(end), "text": text}
                    )
                except (TypeError, ValueError) as error:
                    raise VlmRequestError(f"SAGE ASR 第 {index + 1} 段时间格式无效") from error
            return sorted(segments, key=lambda item: (item["start"], item["end"]))
        raise VlmRequestError("SAGE 会话中的 ASR 文件不存在，无法连同抽帧结果发送给 VLM")

    @staticmethod
    def _format_asr(
        segments: list[dict[str, Any]], max_characters: int
    ) -> tuple[str, int, bool]:
        if not segments or max_characters <= 0:
            return "", 0, bool(segments)
        lines: list[str] = []
        length = 0
        for segment in segments:
            line = (
                f"[{segment['start']:.3f}-{segment['end']:.3f} 秒] "
                f"{segment['text']}"
            )
            added = len(line) + (1 if lines else 0)
            if length + added > max_characters:
                break
            lines.append(line)
            length += added
        return "\n".join(lines), len(lines), len(lines) < len(segments)

    def _build_content(
        self,
        *,
        output_dir: Path,
        media_roots: Sequence[Path] | None = None,
        frames: list[dict[str, Any]],
        query: str,
        asr_text: str,
        max_dimension: int,
        jpeg_quality: int,
    ) -> list[dict[str, Any]]:
        introduction = (
            f"用户问题：{query}\n"
            "接下来是来自同一视频、按照时间顺序排列的视觉信息。"
            "请综合判断并直接回答用户问题。"
        )
        if asr_text:
            introduction = (
                f"用户问题：{query}\n"
                "以下是同一视频的带时间语音转写：\n"
                f"{asr_text}\n"
                "接下来是按照时间顺序排列的视觉信息。"
                "请综合语音和视觉信息，直接回答用户问题。"
            )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": introduction,
            }
        ]
        roots = [output_dir, *(media_roots or [])]
        for index, frame in enumerate(frames, 1):
            image_path = self._resolve_frame_path(roots, str(frame["file"]))
            if image_path is None:
                raise VlmRequestError(f"关键帧文件不存在或路径非法：{frame['file']}")
            timestamp = float(frame.get("timestamp_seconds", 0))
            content.append(
                {
                    "type": "text",
                    "text": f"视频时间：{timestamp:.3f} 秒",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_data_url(
                            image_path, max_dimension=max_dimension, jpeg_quality=jpeg_quality
                        )
                    },
                }
            )
        return content

    @staticmethod
    def _resolve_frame_path(roots: Sequence[Path], relative_path: str) -> Path | None:
        for root in roots:
            resolved_root = root.resolve()
            image_path = (root / relative_path).resolve()
            if (
                (resolved_root in image_path.parents or image_path == resolved_root)
                and image_path.is_file()
            ):
                return image_path
        return None

    @staticmethod
    def _image_data_url(path: Path, *, max_dimension: int, jpeg_quality: int) -> str:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_dimension, max_dimension))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
