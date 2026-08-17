from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_asr_segments(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 ASR JSON：{error}") from error
    return parse_asr_segments(payload)


def parse_asr_segments(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
        time_scale = 1.0
    elif isinstance(payload, dict) and "segments" in payload:
        records = payload["segments"]
        time_scale = 1.0
    elif isinstance(payload, dict) and isinstance(payload.get("content"), list):
        records = []
        for content in payload["content"]:
            if not isinstance(content, dict):
                continue
            sentence_info = content.get("sentence_info", [])
            if not isinstance(sentence_info, list):
                raise ValueError("ASR content.sentence_info 必须是数组")
            records.extend(sentence_info)
        # NSP sentence_info timestamps are milliseconds.
        time_scale = 0.001
    else:
        raise ValueError(
            "ASR JSON 必须是 segments 数组，或包含 content[].sentence_info 的 NSP 响应"
        )

    if not isinstance(records, list):
        raise ValueError("ASR segments 必须是数组")

    segments: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"ASR 第 {index + 1} 段不是对象")
        start = item.get("start", item.get("start_time"))
        end = item.get("end", item.get("end_time"))
        text = str(item.get("text", "")).strip()
        if start is None or end is None or not text:
            continue
        try:
            segment: dict[str, Any] = {
                "start": float(start) * time_scale,
                "end": float(end) * time_scale,
                "text": text,
            }
        except (TypeError, ValueError) as error:
            raise ValueError(f"ASR 第 {index + 1} 段时间格式无效") from error
        if item.get("spk") is not None:
            segment["speaker"] = item["spk"]
        segments.append(segment)

    return sorted(segments, key=lambda item: (item["start"], item["end"]))
