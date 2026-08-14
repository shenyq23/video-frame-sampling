from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..asr.client import SageAsrClient
from ..settings import (
    SAGE_CLIP_MODEL,
    SAGE_ROOT,
    SAGE_TEXT_MODEL,
    sage_asr_settings,
)
from ..torch_runtime import require_torch
from .base import AlgorithmAdapter, ProgressCallback


class _ZeroTextScorer:
    def score(self, text: str, query: str) -> float:
        return 0.0


class SAGEAdapter(AlgorithmAdapter):
    id = "sage"

    def __init__(self) -> None:
        self._clip_cache: dict[str, Any] = {}
        self._text_cache: dict[str, Any] = {}

    @staticmethod
    def _usable_file(path: Path, minimum_size: int = 1024 * 1024) -> bool:
        try:
            if not path.is_file() or path.stat().st_size < minimum_size:
                return False
            with path.open("rb") as source:
                return not source.read(128).startswith(
                    b"version https://git-lfs.github.com/spec/v1"
                )
        except OSError:
            return False

    @classmethod
    def _assets(cls) -> dict[str, dict[str, Any]]:
        text_ready = (
            SAGE_TEXT_MODEL.is_dir()
            and (SAGE_TEXT_MODEL / "config.json").is_file()
            and (SAGE_TEXT_MODEL / "modules.json").is_file()
            and cls._usable_file(SAGE_TEXT_MODEL / "model.safetensors")
        )
        remote = sage_asr_settings()
        return {
            "clip": {"label": "SAGE CLIP", "ready": cls._usable_file(SAGE_CLIP_MODEL)},
            "text_model": {"label": "字幕文本模型", "ready": text_ready},
            "remote_asr": {
                "label": "远程 ASR",
                "ready": bool(remote["base_url"] and remote["token"]),
            },
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "id": "sage",
            "name": "SAGE",
            "description": "Speech-Aware Guided Exploration：结合 ASR、视觉相关性与变化的查询抽帧。",
            "parameter_schema": {
                "asr_modes": ["remote", "upload", "none"],
                "defaults": {
                    "asr_mode": "remote",
                    "budget": 8,
                    "device": "cpu",
                    "save_uniform_baseline": True,
                    "save_candidate_frames": True,
                },
                "assets": self._assets(),
            },
        }

    @staticmethod
    def _ensure_source() -> None:
        if not SAGE_ROOT.is_dir():
            raise RuntimeError(f"SAGE directory is missing: {SAGE_ROOT}")
        if str(SAGE_ROOT) not in sys.path:
            sys.path.insert(0, str(SAGE_ROOT))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _video_metadata(video_path: Path) -> tuple[float, int, int, int]:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("SAGE requires opencv-python") from error
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"无法打开视频：{video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
            raise RuntimeError("无法读取有效的视频 FPS 或总帧数")
        return fps, total_frames, width, height

    @staticmethod
    def _find_video(session_dir: Path) -> Path:
        allowed = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
        for path in (session_dir / "source").iterdir():
            if path.is_file() and path.suffix.lower() in allowed:
                return path
        raise RuntimeError("SAGE session source video is missing")

    @staticmethod
    def _load_asr(path: Path) -> list[Any]:
        from sage_frame.io import load_asr_json

        return load_asr_json(path)

    def prepare_session(
        self,
        *,
        session_id: str,
        video_path: Path,
        original_filename: str,
        parameters: dict[str, Any],
        session_dir: Path,
        progress: ProgressCallback,
    ) -> Path:
        self._ensure_source()
        preprocess_dir = session_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        asr_path = preprocess_dir / "asr.json"
        progress("读取视频信息", 0.04)
        fps, total_frames, width, height = self._video_metadata(video_path)

        mode = str(parameters.get("asr_mode", "remote"))
        remote_job_id: str | None = None
        if mode == "remote":
            client = SageAsrClient(sage_asr_settings())
            remote_job_id = client.generate(
                video_path=video_path,
                destination=asr_path,
                state_path=preprocess_dir / "remote_asr_job.json",
                progress=progress,
            )
        elif mode == "upload":
            source = session_dir / "source" / "asr.json"
            if not source.is_file():
                raise ValueError("SAGE 上传 ASR 模式缺少 JSON 文件")
            progress("复制上传的 ASR JSON", 0.45)
            shutil.copy2(source, asr_path)
        elif mode == "none":
            progress("使用纯视觉 SAGE", 0.45)
            asr_path.write_text('{"segments": []}\n', encoding="utf-8")
        else:
            raise ValueError(f"Unknown SAGE ASR mode: {mode}")

        progress("校验 ASR JSON", 0.92)
        segments = self._load_asr(asr_path)
        if mode == "remote" and remote_job_id and sage_asr_settings()["delete_remote"]:
            SageAsrClient(sage_asr_settings()).delete(remote_job_id)

        metadata = {
            "schema_version": "1.0",
            "session_id": session_id,
            "video": {
                "filename": original_filename,
                "sha256": self._sha256(video_path),
                "fps": fps,
                "duration_seconds": total_frames / fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
            },
            "asr": {
                "mode": mode,
                "path": "asr.json",
                "segment_count": len(segments),
                "remote_job_id": remote_job_id,
            },
            "candidate_sampling": {
                "mode": "query-dependent",
                "interval_seconds": None,
                "effective_interval_seconds": None,
                "candidate_count": 0,
            },
            "parameters": parameters,
        }
        metadata_path = preprocess_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        progress("SAGE 视频已准备好", 1.0)
        return metadata_path

    def _clip_backend(self, device: str) -> Any:
        backend = self._clip_cache.get(device)
        if backend is None:
            from sage_frame.model_adapters import CLIPBackend

            backend = CLIPBackend(SAGE_CLIP_MODEL, device=device)
            self._clip_cache[device] = backend
        return backend

    def _text_scorer(self, device: str) -> Any:
        scorer = self._text_cache.get(device)
        if scorer is None:
            from sage_frame.model_adapters import SentenceTransformerTextScorer

            scorer = SentenceTransformerTextScorer(SAGE_TEXT_MODEL, device=device)
            self._text_cache[device] = scorer
        return scorer

    @staticmethod
    def _uniform_indices(total_frames: int, count: int) -> list[int]:
        if count <= 0 or total_frames <= 0:
            return []
        if count == 1:
            return [max(0, (total_frames - 1) // 2)]
        return sorted(
            {round(index * (total_frames - 1) / (count - 1)) for index in range(count)}
        )

    @staticmethod
    def _frame_index_at(time_seconds: float, fps: float, total_frames: int) -> int:
        return max(0, min(total_frames - 1, round(time_seconds * fps)))

    @staticmethod
    def _export_frames(
        video_path: Path,
        frame_specs: list[dict[str, Any]],
        output_dir: Path,
        relative_dir: str,
        fps: float,
    ) -> list[dict[str, Any]]:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("SAGE requires opencv-python") from error
        target_dir = output_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        records: list[dict[str, Any]] = []
        try:
            for order, spec in enumerate(frame_specs, 1):
                frame_index = int(spec["original_frame_index"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"无法解码 SAGE 帧：{frame_index}")
                timestamp = float(spec.get("timestamp_seconds", frame_index / fps))
                filename = f"{order:04d}_t{timestamp:010.3f}_f{frame_index}.jpg"
                if not cv2.imwrite(str(target_dir / filename), image):
                    raise RuntimeError(f"无法写入 SAGE 帧：{filename}")
                record = dict(spec)
                record.update(
                    {
                        "order": order,
                        "file": f"{relative_dir}/{filename}",
                        "original_frame_index": frame_index,
                        "timestamp_seconds": round(timestamp, 6),
                    }
                )
                records.append(record)
        finally:
            capture.release()
        return records

    def run_from_session(
        self,
        *,
        job_id: str,
        session_dir: Path,
        metadata_path: Path,
        query: str,
        parameters: dict[str, Any],
        output_dir: Path,
        progress: ProgressCallback,
    ) -> Path:
        require_torch()
        self._ensure_source()
        from sage_frame import SAGESelector
        from sage_frame.model_adapters import (
            CLIPVisualRelevanceScorer,
            OpenCVCLIPFrameProvider,
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        session_parameters = dict(metadata.get("parameters", {}))
        for key in ("asr_mode", "device"):
            if session_parameters.get(key) != parameters.get(key):
                raise ValueError(f"SAGE session preprocessing config does not match '{key}'")

        device = str(parameters.get("device", "cpu"))
        if not self._usable_file(SAGE_CLIP_MODEL):
            raise RuntimeError(f"SAGE CLIP 模型不存在或不完整：{SAGE_CLIP_MODEL}")
        video_path = self._find_video(session_dir)
        asr = self._load_asr(session_dir / "preprocess" / "asr.json")
        if asr and not self._assets()["text_model"]["ready"]:
            raise RuntimeError(f"SAGE 字幕文本模型不存在或不完整：{SAGE_TEXT_MODEL}")

        progress("加载 SAGE 模型", 0.08)
        backend = self._clip_backend(device)
        provider = OpenCVCLIPFrameProvider(video_path, backend)
        text_scorer = self._text_scorer(device) if asr else _ZeroTextScorer()
        trace: list[str] = []
        selector = SAGESelector(
            provider,
            CLIPVisualRelevanceScorer(backend),
            text_scorer,
            trace=trace.append,
        )
        progress("执行 SAGE 查询相关选帧", 0.20)
        keyframes = selector.select(
            duration=provider.duration,
            query=query,
            asr=asr,
            budget=int(parameters.get("budget", 8)),
            fps=provider.fps,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_position = {
            (candidate.time, candidate.segment_index): index
            for index, candidate in enumerate(selector.last_candidates)
        }
        selected_keys = {(frame.time, frame.segment_index) for frame in keyframes}
        selected_specs = [
            {
                "selected_order": order,
                "original_frame_index": self._frame_index_at(
                    frame.time, provider.fps, provider.frame_count
                ),
                "timestamp_seconds": min(
                    frame.time, (provider.frame_count - 1) / provider.fps
                ),
                "candidate_index": candidate_position.get((frame.time, frame.segment_index), -1),
                "candidate_order": candidate_position.get((frame.time, frame.segment_index), -1) + 1,
                "relevance_score": frame.relevance,
                "normalized_score": frame.relevance,
                "sage_score": frame.score,
                "change_score": frame.change,
                "segment_id": frame.segment_index,
                "selected_by_sage": True,
                "selected": True,
            }
            for order, frame in enumerate(keyframes, 1)
        ]
        progress("导出 SAGE 抽出帧", 0.76)
        selected_records = self._export_frames(
            video_path, selected_specs, output_dir, "frames", provider.fps
        )

        save_uniform = bool(parameters.get("save_uniform_baseline", True))
        uniform_specs = [
            {
                "original_frame_index": index,
                "timestamp_seconds": index / provider.fps,
                "candidate_index": order - 1,
                "candidate_order": order,
                "selected": False,
            }
            for order, index in enumerate(
                self._uniform_indices(provider.frame_count, len(keyframes)), 1
            )
        ]
        uniform_records = (
            self._export_frames(
                video_path, uniform_specs, output_dir, "uniform_frames", provider.fps
            )
            if save_uniform
            else []
        )

        save_candidates = bool(parameters.get("save_candidate_frames", True))
        candidate_specs = [
            {
                "original_frame_index": self._frame_index_at(
                    candidate.time, provider.fps, provider.frame_count
                ),
                "timestamp_seconds": min(
                    candidate.time, (provider.frame_count - 1) / provider.fps
                ),
                "candidate_index": index,
                "candidate_order": index + 1,
                "relevance_score": candidate.relevance,
                "normalized_score": candidate.relevance,
                "base_score": candidate.base_score,
                "change_score": candidate.change,
                "segment_id": candidate.segment_index,
                "selected_by_sage": (candidate.time, candidate.segment_index) in selected_keys,
                "selected": (candidate.time, candidate.segment_index) in selected_keys,
            }
            for index, candidate in enumerate(selector.last_candidates)
        ]
        candidate_records = (
            self._export_frames(
                video_path, candidate_specs, output_dir, "candidate_frames", provider.fps
            )
            if save_candidates
            else []
        )
        (output_dir / "sage_trace.log").write_text("\n".join(trace), encoding="utf-8")

        segments = [asdict(segment) for segment in selector.last_segments]
        manifest = {
            "schema_version": "1.0",
            "run_id": job_id,
            "session_id": metadata.get("session_id"),
            "algorithm": {
                "id": "sage",
                "name": "Speech-Aware Guided Exploration",
                "mode": "visual+asr" if asr else "visual-only",
            },
            "video": metadata["video"],
            "query": query,
            "parameters": parameters,
            "asr": metadata.get("asr", {}),
            "candidate_sampling": {
                "mode": "query-dependent-regions",
                "interval_seconds": 1.0,
                "effective_interval_seconds": 1.0,
                "candidate_count": len(selector.last_candidates),
            },
            "summary": {
                "requested_keyframes": int(parameters.get("budget", 8)),
                "selected_keyframes": len(selected_records),
                "candidate_frames": len(selector.last_candidates),
                "segments": len(segments),
                "asr_segments": len(asr),
            },
            "selected_frames": selected_records,
            "uniform_frames": uniform_records,
            "candidates": candidate_records,
            "frame_sets": {
                "selected": {
                    "available": True,
                    "count": len(selected_records),
                    "frames": selected_records,
                },
                "uniform": {
                    "available": save_uniform,
                    "selection_rule": "uniformly spaced over the full video",
                    "count": len(uniform_records),
                    "frames": uniform_records,
                },
                "candidates": {
                    "available": save_candidates,
                    "selection_rule": "query-dependent candidates explored by SAGE",
                    "count": len(candidate_records),
                    "frames": candidate_records,
                },
            },
            "diagnostics": {"trace": "sage_trace.log"},
            "algorithm_trace": {"segments": segments},
        }
        progress("生成 SAGE Manifest", 0.96)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        progress("完成", 1.0)
        return manifest_path

    def run(self, **_: Any) -> Path:
        raise RuntimeError("SAGE requires a prepared video session")
