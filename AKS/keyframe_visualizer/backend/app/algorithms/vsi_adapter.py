from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..settings import (
    VSI_BUNDLED_CLIP_MODEL,
    VSI_BUNDLED_EASYOCR_DIR,
    VSI_BUNDLED_TEXT_MODEL,
    VSI_BUNDLED_YOLO_MODEL,
    VSI_MODEL_CACHE_DIR,
    VSI_ROOT,
)
from ..torch_runtime import require_torch
from .base import AlgorithmAdapter, ProgressCallback


class VSIAdapter(AlgorithmAdapter):
    """Adapter for the standalone VSI_VideoFraming implementation."""

    id = "vsi"

    def __init__(self) -> None:
        self._matcher_cache: dict[tuple[str, str], Any] = {}

    @staticmethod
    def _usable_file(path: Path, minimum_size: int = 1024 * 1024) -> bool:
        """Reject missing, truncated and unexpanded Git LFS model files."""
        try:
            if not path.is_file() or path.stat().st_size < minimum_size:
                return False
            with path.open("rb") as file:
                return not file.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
        except OSError:
            return False

    @classmethod
    def _bundled_text_model(cls) -> Path | None:
        required_files = (
            "config.json",
            "modules.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "sentencepiece.bpe.model",
        )
        if not VSI_BUNDLED_TEXT_MODEL.is_dir():
            return None
        if not all((VSI_BUNDLED_TEXT_MODEL / filename).is_file() for filename in required_files):
            return None
        if not (VSI_BUNDLED_TEXT_MODEL / "1_Pooling" / "config.json").is_file():
            return None
        return (
            VSI_BUNDLED_TEXT_MODEL
            if cls._usable_file(VSI_BUNDLED_TEXT_MODEL / "model.safetensors")
            else None
        )

    @classmethod
    def _assets(cls) -> dict[str, dict[str, Any]]:
        text_model = cls._bundled_text_model()
        assets = {
            "yolo": cls._usable_file(VSI_BUNDLED_YOLO_MODEL),
            "easyocr": all(
                cls._usable_file(VSI_BUNDLED_EASYOCR_DIR / filename)
                for filename in ("craft_mlt_25k.pth", "zh_sim_g2.pth")
            ),
            "text_model": text_model is not None,
            "clip": cls._usable_file(VSI_BUNDLED_CLIP_MODEL),
        }
        labels = {
            "yolo": "YOLO-World",
            "easyocr": "EasyOCR",
            "text_model": "字幕文本模型",
            "clip": "CLIP",
        }
        return {
            key: {"label": labels[key], "ready": ready}
            for key, ready in assets.items()
        }

    @classmethod
    def _resolve_text_model(cls, model_name: str) -> str:
        default_models = {
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "weights/sentence_transformer/paraphrase-multilingual-mpnet-base-v2",
        }
        if model_name in default_models:
            bundled = cls._bundled_text_model()
            if bundled is not None:
                return str(bundled)
        requested = Path(model_name).expanduser()
        if not requested.is_absolute() and (VSI_ROOT / requested).is_dir():
            return str((VSI_ROOT / requested).resolve())
        return model_name

    @classmethod
    def _resolve_yolo_model(cls, model_name: str) -> str:
        requested = Path(model_name).expanduser()
        if requested.is_absolute():
            if not cls._usable_file(requested):
                raise ValueError(f"YOLO-World 模型不存在、未完整下载或仍是 Git LFS 指针：{requested}")
            return str(requested)
        local = VSI_ROOT / requested
        if cls._usable_file(local):
            return str(local)
        if model_name == VSI_BUNDLED_YOLO_MODEL.name and VSI_BUNDLED_YOLO_MODEL.exists():
            raise ValueError(
                "VSI 自带的 YOLO-World 模型未完整下载。请在 VSI_VideoFraming 目录执行 git lfs pull。"
            )
        return model_name

    @staticmethod
    def _ensure_runtime() -> None:
        if not VSI_ROOT.is_dir():
            raise RuntimeError(f"VSI_VideoFraming directory is missing: {VSI_ROOT}")
        # Torch must already be loaded before EasyOCR/Ultralytics pull it in from
        # this worker thread; app/__init__.py does that at startup and this call
        # only reports the failure with an actionable message.
        require_torch()
        if str(VSI_ROOT) not in sys.path:
            sys.path.insert(0, str(VSI_ROOT))
        os.environ.setdefault("HF_HOME", str(VSI_MODEL_CACHE_DIR / "huggingface"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(VSI_MODEL_CACHE_DIR / "ultralytics"))
        os.environ.setdefault("MPLCONFIGDIR", str(VSI_MODEL_CACHE_DIR / "matplotlib"))
        for directory in (
            VSI_MODEL_CACHE_DIR / "huggingface",
            VSI_MODEL_CACHE_DIR / "ultralytics",
            VSI_MODEL_CACHE_DIR / "matplotlib",
            VSI_MODEL_CACHE_DIR / "easyocr_models",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _configure_ultralytics() -> None:
        """Point Ultralytics' auxiliary CLIP lookup at VSI's bundled weights."""
        try:
            from ultralytics import settings
        except ImportError as error:
            raise RuntimeError("VSI requires ultralytics; install the backend requirements") from error
        settings.update({"weights_dir": str(VSI_ROOT / "weights")})

    def metadata(self) -> dict[str, Any]:
        return {
            "id": "vsi",
            "name": "VSI",
            "description": "Visual Subtitle Integration：融合目标检测和字幕语义的自适应抽帧。",
            "parameter_schema": {
                "subtitle_modes": ["ocr", "upload", "none"],
                "defaults": {
                    "subtitle_mode": "ocr",
                    "ocr_fps": 2.0,
                    "ocr_crop_top": 0.62,
                    "text_model": "weights/sentence_transformer/paraphrase-multilingual-mpnet-base-v2",
                    "objects": [],
                    "top_k": 8,
                    "detection_budget": 64,
                    "samples_per_round": 16,
                    "text_weight": 0.3,
                    "model": "yolov8s-worldv2.pt",
                    "device": "cpu",
                    "seed": 0,
                },
                "assets": self._assets(),
            },
        }

    @staticmethod
    def _video_metadata(video_path: Path) -> tuple[float, int, int, int]:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("VSI requires opencv-python") from error
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频：{video_path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
            raise RuntimeError("无法读取有效的视频 FPS 或总帧数")
        return fps, total_frames, width, height

    @staticmethod
    def _find_video(session_dir: Path) -> Path:
        allowed = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
        for path in (session_dir / "source").iterdir():
            if path.is_file() and path.suffix.lower() in allowed:
                return path
        raise RuntimeError("VSI session source video is missing")

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
        self._ensure_runtime()
        from vsi.io import load_subtitles
        from vsi.ocr import extract_burned_subtitles

        preprocess_dir = session_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        progress("读取视频信息", 0.08)
        fps, total_frames, width, height = self._video_metadata(video_path)

        mode = str(parameters.get("subtitle_mode", "ocr"))
        subtitles: list[Any] = []
        subtitle_source: str | None = None
        if mode == "upload":
            subtitle_path = next(
                (
                    path for path in (session_dir / "source").iterdir()
                    if path.is_file() and path.suffix.lower() in {".srt", ".json"}
                ),
                None,
            )
            if subtitle_path is None:
                raise ValueError("选择上传字幕时必须提供 .srt 或 .json 文件")
            progress("解析上传字幕", 0.35)
            subtitles = load_subtitles(subtitle_path)
            subtitle_source = f"upload:{subtitle_path.name}"
        elif mode == "ocr":
            progress("识别烧录字幕", 0.15)
            bundled_ocr_ready = bool(self._assets()["easyocr"]["ready"])
            subtitles = extract_burned_subtitles(
                video_path,
                sample_fps=float(parameters.get("ocr_fps", 2.0)),
                crop_top_ratio=float(parameters.get("ocr_crop_top", 0.62)),
                cache_path=preprocess_dir / "subtitles.json",
                device=str(parameters.get("device", "cpu")),
                model_storage_directory=(
                    VSI_BUNDLED_EASYOCR_DIR
                    if bundled_ocr_ready
                    else VSI_MODEL_CACHE_DIR / "easyocr_models"
                ),
            )
            subtitle_source = "ocr"
        elif mode != "none":
            raise ValueError(f"不支持的 VSI 字幕模式：{mode}")

        subtitles_path = preprocess_dir / "subtitles.json"
        if mode != "ocr":
            subtitles_path.write_text(
                json.dumps([asdict(item) for item in subtitles], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        metadata = {
            "schema_version": "1.0",
            "session_id": session_id,
            "algorithm": "vsi",
            "video": {
                "filename": original_filename,
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": total_frames / fps,
                "width": width,
                "height": height,
            },
            "subtitle": {
                "mode": mode,
                "source": subtitle_source,
                "count": len(subtitles),
                "path": "subtitles.json",
            },
            "parameters": parameters,
        }
        metadata_path = preprocess_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        progress("完成", 1.0)
        return metadata_path

    def _matcher(self, model_name: str, device: str) -> Any:
        self._ensure_runtime()
        resolved_model = self._resolve_text_model(model_name)
        key = (resolved_model, device)
        if key not in self._matcher_cache:
            from vsi.adapters import SentenceTransformerMatcher
            self._matcher_cache[key] = SentenceTransformerMatcher(model=resolved_model, device=device)
        return self._matcher_cache[key]

    @staticmethod
    def _uniform_indices(total_frames: int, count: int) -> list[int]:
        if total_frames <= 0 or count <= 0:
            return []
        return np.unique(np.rint(np.linspace(0, total_frames - 1, min(count, total_frames))).astype(int)).tolist()

    @staticmethod
    def _export_frames(
        video_path: Path,
        indices: list[int],
        output_dir: Path,
        relative_dir: str,
        fps: float,
        scores: dict[int, dict[str, float]],
        selected: set[int],
    ) -> list[dict[str, Any]]:
        import cv2

        target_dir = output_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")
        records: list[dict[str, Any]] = []
        try:
            for order, index in enumerate(indices, 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"无法解码 VSI 帧：{index}")
                timestamp = index / fps
                filename = f"{order:04d}_t{timestamp:010.3f}_f{index}.jpg"
                path = target_dir / filename
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"无法保存 VSI 帧：{path}")
                records.append({
                    "order": order,
                    "file": f"{relative_dir}/{filename}",
                    "original_frame_index": int(index),
                    "timestamp_seconds": round(timestamp, 6),
                    "candidate_index": order - 1,
                    "candidate_order": order,
                    "selected": index in selected,
                    "selected_by_vsi": index in selected,
                    **scores.get(index, {}),
                })
        finally:
            cap.release()
        return records

    @staticmethod
    def _write_visited_scores(output_dir: Path, result: Any, fps: float) -> Path:
        path = output_dir / "visited_frame_scores.csv"
        selected = set(result.frame_indices)
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["frame_index", "timestamp_seconds", "selected", "fused_score", "sampling_probability"])
            for index in result.visited_indices:
                writer.writerow([
                    index, index / fps, int(index in selected),
                    float(result.fused_scores[index]), float(result.sampling_probabilities[index]),
                ])
        return path

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
        self._ensure_runtime()
        from vsi.adapters import UltralyticsYOLOWorldScorer
        from vsi.core import Subtitle, VSIConfig, select_keyframes, soft_threshold, subtitle_frame_scores

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        session_parameters = dict(metadata.get("parameters", {}))
        for key in ("subtitle_mode", "ocr_fps", "ocr_crop_top", "text_model", "device"):
            if session_parameters.get(key) != parameters.get(key):
                raise ValueError(f"VSI session preprocessing config does not match '{key}'")
        video_path = self._find_video(session_dir)
        video = metadata["video"]
        fps = float(video["fps"])
        total_frames = int(video["total_frames"])
        objects = [str(item).strip() for item in parameters.get("objects", []) if str(item).strip()]
        if not objects:
            raise ValueError("VSI objects 不能为空")
        output_dir.mkdir(parents=True, exist_ok=True)

        progress("计算 query 与字幕相关性", 0.08)
        subtitle_path = session_dir / "preprocess" / "subtitles.json"
        subtitle_data = json.loads(subtitle_path.read_text(encoding="utf-8")) if subtitle_path.is_file() else []
        subtitles = [Subtitle(float(item["start"]), float(item["end"]), str(item.get("text", ""))) for item in subtitle_data]
        text_scores = None
        effective_text_weight = 0.0
        mode = "visual-only"
        if subtitles:
            matcher = self._matcher(str(parameters.get("text_model")), str(parameters.get("device", "cpu")))
            similarities = soft_threshold(matcher(query, [subtitle.text for subtitle in subtitles]))
            text_scores = subtitle_frame_scores(subtitles, similarities, total_frames, fps)
            effective_text_weight = float(parameters.get("text_weight", 0.3))
            mode = "visual+subtitle"

        progress("加载 YOLO-World", 0.18)
        self._configure_ultralytics()
        yolo_model = self._resolve_yolo_model(str(parameters.get("model", "yolov8s-worldv2.pt")))
        detector = UltralyticsYOLOWorldScorer(
            str(video_path), objects, model=yolo_model,
            device=str(parameters.get("device", "cpu")),
        )
        progress("执行 VSI 自适应采样", 0.30)
        result = select_keyframes(
            total_frames, fps, detector, text_scores=text_scores,
            config=VSIConfig(
                top_k=int(parameters.get("top_k", 8)),
                samples_per_round=int(parameters.get("samples_per_round", 16)),
                detection_budget=int(parameters.get("detection_budget", 64)),
                text_weight=effective_text_weight,
                seed=int(parameters.get("seed", 0)),
            ),
        )
        selected = set(result.frame_indices)
        scores = {
            int(index): {
                "relevance_score": float(result.fused_scores[index]),
                "normalized_score": float(result.fused_scores[index]),
                "fused_score": float(result.fused_scores[index]),
                "sampling_probability": float(result.sampling_probabilities[index]),
                "visited_order": order,
            }
            for order, index in enumerate(result.visited_indices, 1)
        }
        progress("导出 VSI 帧", 0.72)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_records = self._export_frames(video_path, result.frame_indices, output_dir, "frames", fps, scores, selected)
        uniform_indices = self._uniform_indices(total_frames, len(result.frame_indices))
        save_uniform = bool(parameters.get("save_uniform_baseline", True))
        save_candidates = bool(parameters.get("save_candidate_frames", True))
        uniform_records = self._export_frames(video_path, uniform_indices, output_dir, "uniform_frames", fps, scores, selected) if save_uniform else []
        candidate_records = self._export_frames(video_path, result.visited_indices, output_dir, "visited_frames", fps, scores, selected) if save_candidates else []
        scores_path = self._write_visited_scores(output_dir, result, fps)
        np.savez_compressed(output_dir / "sampling_history.npz", history=np.asarray(result.history), visited_indices=np.asarray(result.visited_indices))

        manifest = {
            "schema_version": "1.0",
            "run_id": job_id,
            "session_id": metadata.get("session_id"),
            "algorithm": {"id": "vsi", "name": "Visual Subtitle Integration", "mode": mode},
            "video": video,
            "query": query,
            "parameters": parameters,
            "subtitle": metadata.get("subtitle", {}),
            "summary": {
                "requested_keyframes": int(parameters.get("top_k", 8)),
                "selected_keyframes": len(selected_records),
                "candidate_frames": len(result.visited_indices),
                "visited_frames": len(result.visited_indices),
                "rounds": result.rounds,
            },
            "selected_frames": selected_records,
            "uniform_frames": uniform_records,
            "candidates": candidate_records,
            "frame_sets": {
                "selected": {"available": True, "count": len(selected_records), "frames": selected_records},
                "uniform": {"available": save_uniform, "selection_rule": "uniformly spaced over the full video", "count": len(uniform_records), "frames": uniform_records},
                "candidates": {"available": save_candidates, "selection_rule": "frames visited by the VSI detector", "count": len(candidate_records), "frames": candidate_records},
            },
            "diagnostics": {"visited_scores_csv": scores_path.name, "sampling_history": "sampling_history.npz"},
            "algorithm_trace": {"rounds": result.rounds},
        }
        progress("生成 VSI Manifest", 0.96)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        progress("完成", 1.0)
        return manifest_path

    def run(self, **_: Any) -> Path:
        raise RuntimeError("VSI requires a prepared video session")
