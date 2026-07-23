from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..models import ClipModelStore
from ..settings import AKS_ROOT, CLIP_MODELS_DIR, feature_profile_status, load_feature_profiles
from ..video_reader import open_video
from .base import AlgorithmAdapter, ProgressCallback


class AKSAdapter(AlgorithmAdapter):
    id = "aks"

    def metadata(self) -> dict[str, Any]:
        profiles = load_feature_profiles()
        statuses = feature_profile_status()
        public_profiles = [
            {
                "id": key,
                "name": value.get("name", key),
                "backend": value.get("backend"),
                **statuses.get(key, {}),
            }
            for key, value in profiles.items()
            if value.get("enabled", True)
        ]
        return {
            "id": "aks",
            "name": "AKS",
            "description": "Adaptive Keyframe Sampling：兼顾 query 相关性与时间覆盖。",
            "parameter_schema": {
                "aks_modes": ["robust", "original"],
                "candidate_sampling_modes": ["interval", "original"],
                "feature_backends": ["clip", "pangu", "mep"],
                "feature_profiles": public_profiles,
                "defaults": {
                    "aks_mode": "robust",
                    "max_num_frames": 32,
                    "candidate_sampling": "interval",
                    "sample_interval": 1.0,
                    "feature_backend": "clip",
                    "model_name": "openai/clip-vit-base-patch32",
                },
            },
        }

    def _feature_config(self, parameters: dict[str, Any]) -> dict[str, Any]:
        profile_id = parameters.get("feature_profile")
        if not profile_id:
            return {}
        profiles = load_feature_profiles()
        if profile_id not in profiles:
            raise ValueError(f"Unknown feature profile: {profile_id}")
        profile = profiles[profile_id]
        if profile.get("backend") != parameters["feature_backend"]:
            raise ValueError("Feature profile backend does not match the selected backend")
        return dict(profile.get("config", {}))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        original_filename: str,
        query: str,
        parameters: dict[str, Any],
        output_dir: Path,
        progress: ProgressCallback,
    ) -> Path:
        if str(AKS_ROOT) not in sys.path:
            sys.path.insert(0, str(AKS_ROOT))
        from PIL import Image
        from aks_core import normalize_scores, select_frames
        from aks_keyframes_v2 import (
            choose_device,
            sample_candidate_indices,
            sample_uniform_indices,
        )
        from feature_backends import create_relevance_scorer

        output_dir.mkdir(parents=True, exist_ok=True)

        progress("解析视频", 0.04)
        reader = open_video(video_path, int(parameters["decode_threads"]))
        fps = reader.fps
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("无法读取有效的视频 FPS")
        total_frames = reader.total_frames
        candidate_indices = sample_candidate_indices(
            total_frames,
            fps,
            parameters["candidate_sampling"],
            float(parameters["sample_interval"]),
        )
        if not candidate_indices:
            raise ValueError("视频中没有可用候选帧")

        feature_config = self._feature_config(parameters)
        model_name = parameters["model_name"]
        if parameters.get("clip_model_id"):
            if parameters["feature_backend"] != "clip":
                raise ValueError("clip_model_id can only be used with the CLIP backend")
            model_name = str(ClipModelStore(CLIP_MODELS_DIR).resolve(parameters["clip_model_id"]))
        device = (
            choose_device(parameters["device"])
            if parameters["feature_backend"] == "clip"
            else "remote"
        )
        progress("加载特征模型", 0.08)
        scorer = create_relevance_scorer(
            parameters["feature_backend"],
            feature_config,
            model_name=model_name,
            device=device,
            batch_size=int(parameters["batch_size"]),
        )
        scorer.prepare_query(query)

        scores: list[float] = []
        batch_size = int(parameters["batch_size"])
        total_batches = math.ceil(len(candidate_indices) / batch_size)
        for batch_number, start in enumerate(range(0, len(candidate_indices), batch_size), 1):
            indices = candidate_indices[start : start + batch_size]
            arrays = reader.get_batch(indices)
            scores.extend(scorer.score_images([Image.fromarray(array) for array in arrays]))
            progress(
                f"计算候选帧特征 {min(start + len(indices), len(candidate_indices))}/{len(candidate_indices)}",
                0.10 + 0.66 * batch_number / total_batches,
            )

        progress("执行 AKS 选择", 0.80)
        selection = select_frames(
            scores,
            candidate_indices,
            max_num_frames=int(parameters["max_num_frames"]),
            threshold=float(parameters["threshold"]),
            std_threshold=float(parameters["std_threshold"]),
            max_depth=int(parameters["max_depth"]),
            mode=parameters["aks_mode"],
        )

        normalized_scores = normalize_scores(scores).tolist()
        position_by_frame = {frame: position for position, frame in enumerate(candidate_indices)}
        score_by_frame = {frame: float(score) for frame, score in zip(candidate_indices, scores)}
        normalized_by_frame = {
            frame: float(score) for frame, score in zip(candidate_indices, normalized_scores)
        }
        segment_by_position: dict[int, tuple[int, Any]] = {}
        rank_by_position: dict[int, int] = {}
        for segment_id, segment in enumerate(selection.segments):
            for position in range(segment.candidate_start, segment.candidate_end + 1):
                segment_by_position[position] = (segment_id, segment)
            ranked = sorted(
                range(segment.candidate_start, segment.candidate_end + 1),
                key=lambda position: (-normalized_scores[position], position),
            )
            rank_by_position.update({position: rank + 1 for rank, position in enumerate(ranked)})

        selected_indices = selection.frame_indices
        selected_positions = {position_by_frame[index] for index in selected_indices}
        save_uniform = bool(parameters.get("save_uniform_baseline", True))
        save_candidates = bool(parameters.get("save_candidate_frames", True))
        uniform_indices = (
            sample_uniform_indices(candidate_indices, len(selected_indices))
            if save_uniform
            else []
        )
        export_total = len(selected_indices) + len(uniform_indices)
        if save_candidates:
            export_total += len(candidate_indices)
        exported = 0

        def export_frame_group(
            frame_indices: list[int], relative_dir: str, *, include_aks_trace: bool = False
        ) -> list[dict[str, Any]]:
            nonlocal exported
            target_dir = output_dir / relative_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, Any]] = []
            for start in range(0, len(frame_indices), batch_size):
                indices = frame_indices[start : start + batch_size]
                arrays = reader.get_batch(indices)
                for frame_index, array in zip(indices, arrays):
                    order = len(records) + 1
                    candidate_position = position_by_frame[frame_index]
                    timestamp = frame_index / fps
                    filename = f"{order:04d}_t{timestamp:010.3f}_f{frame_index}.jpg"
                    Image.fromarray(array).save(
                        target_dir / filename, quality=int(parameters["jpeg_quality"])
                    )
                    record: dict[str, Any] = {
                        "order": order,
                        "file": f"{relative_dir}/{filename}",
                        "original_frame_index": frame_index,
                        "timestamp_seconds": round(timestamp, 6),
                        "candidate_index": candidate_position,
                        "candidate_order": candidate_position + 1,
                        "relevance_score": score_by_frame[frame_index],
                        "normalized_score": normalized_by_frame[frame_index],
                        "selected_by_aks": candidate_position in selected_positions,
                        "selected": candidate_position in selected_positions,
                    }
                    if include_aks_trace:
                        segment_id, segment = segment_by_position.get(
                            candidate_position, (-1, None)
                        )
                        record.update(
                            {
                                "selected_order": order,
                                "segment_id": segment_id,
                                "segment_depth": segment.depth if segment else None,
                                "segment_quota": segment.quota if segment else None,
                                "rank_in_segment": rank_by_position.get(candidate_position),
                            }
                        )
                    records.append(record)
                    exported += 1
                if export_total:
                    progress(
                        f"导出帧图片 {exported}/{export_total}",
                        0.82 + 0.15 * exported / export_total,
                    )
            return records

        progress("导出 AKS 关键帧", 0.82)
        selected_records = export_frame_group(
            selected_indices, "frames", include_aks_trace=True
        )
        uniform_records = (
            export_frame_group(uniform_indices, "uniform_frames") if save_uniform else []
        )
        candidate_records = (
            export_frame_group(candidate_indices, "candidate_frames")
            if save_candidates
            else [
                {
                    "candidate_index": position,
                    "candidate_order": position + 1,
                    "original_frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 6),
                    "relevance_score": float(scores[position]),
                    "normalized_score": float(normalized_scores[position]),
                    "selected_by_aks": position in selected_positions,
                    "selected": position in selected_positions,
                }
                for position, frame_index in enumerate(candidate_indices)
            ]
        )

        progress("生成 Manifest", 0.98)
        manifest = {
            "schema_version": "1.0",
            "run_id": job_id,
            "algorithm": {
                "id": "aks",
                "name": "Adaptive Keyframe Sampling",
                "mode": selection.mode,
                "shared_core": "aks_core.py",
            },
            "video": {
                "filename": original_filename,
                "sha256": self._sha256(video_path),
                "fps": fps,
                "duration_seconds": total_frames / fps,
                "total_frames": total_frames,
            },
            "query": query,
            "parameters": parameters,
            "candidate_sampling": {
                "mode": parameters["candidate_sampling"],
                "interval_seconds": parameters["sample_interval"]
                if parameters["candidate_sampling"] == "interval"
                else None,
                "effective_interval_seconds": (
                    (candidate_indices[1] - candidate_indices[0]) / fps
                    if parameters["candidate_sampling"] == "interval"
                    and len(candidate_indices) > 1
                    else None
                ),
                "candidate_count": len(candidate_indices),
            },
            "feature_extraction": scorer.metadata,
            "video_decoder": reader.backend,
            "summary": {
                "requested_keyframes": parameters["max_num_frames"],
                "selected_keyframes": len(selected_records),
                "candidate_frames": len(candidate_indices),
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
                    "selection_rule": "uniformly spaced over the AKS candidate pool",
                    "count": len(uniform_records),
                    "frames": uniform_records,
                },
                "candidates": {
                    "available": save_candidates,
                    "count": len(candidate_records) if save_candidates else 0,
                    "frames": candidate_records if save_candidates else [],
                },
            },
            "algorithm_trace": {
                "segments": [asdict(segment) for segment in selection.segments]
            },
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        progress("完成", 1.0)
        return manifest_path
