from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..settings import AKS_ROOT, load_feature_profiles
from ..video_reader import open_video
from .base import AlgorithmAdapter, ProgressCallback


class AKSAdapter(AlgorithmAdapter):
    id = "aks"

    def metadata(self) -> dict[str, Any]:
        profiles = load_feature_profiles()
        public_profiles = [
            {"id": key, "name": value.get("name", key), "backend": value.get("backend")}
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
        from aks_keyframes_v2 import choose_device, sample_candidate_indices
        from feature_backends import create_relevance_scorer

        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

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
        device = (
            choose_device(parameters["device"])
            if parameters["feature_backend"] == "clip"
            else "remote"
        )
        progress("加载特征模型", 0.08)
        scorer = create_relevance_scorer(
            parameters["feature_backend"],
            feature_config,
            model_name=parameters["model_name"],
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

        progress("导出关键帧", 0.84)
        selected_records: list[dict[str, Any]] = []
        selected_indices = selection.frame_indices
        for start in range(0, len(selected_indices), batch_size):
            indices = selected_indices[start : start + batch_size]
            arrays = reader.get_batch(indices)
            for frame_index, array in zip(indices, arrays):
                selected_order = len(selected_records) + 1
                candidate_position = position_by_frame[frame_index]
                timestamp = frame_index / fps
                filename = (
                    f"{selected_order:03d}_t{timestamp:010.3f}_f{frame_index}.jpg"
                )
                Image.fromarray(array).save(
                    frames_dir / filename, quality=int(parameters["jpeg_quality"])
                )
                segment_id, segment = segment_by_position.get(candidate_position, (-1, None))
                selected_records.append(
                    {
                        "selected_order": selected_order,
                        "file": f"frames/{filename}",
                        "original_frame_index": frame_index,
                        "timestamp_seconds": round(timestamp, 6),
                        "candidate_index": candidate_position,
                        "candidate_order": candidate_position + 1,
                        "relevance_score": score_by_frame[frame_index],
                        "normalized_score": normalized_by_frame[frame_index],
                        "segment_id": segment_id,
                        "segment_depth": segment.depth if segment else None,
                        "segment_quota": segment.quota if segment else None,
                        "rank_in_segment": rank_by_position.get(candidate_position),
                    }
                )

        selected_positions = {position_by_frame[index] for index in selected_indices}
        candidate_records = [
            {
                "candidate_index": position,
                "candidate_order": position + 1,
                "original_frame_index": frame_index,
                "timestamp_seconds": round(frame_index / fps, 6),
                "relevance_score": float(scores[position]),
                "normalized_score": float(normalized_scores[position]),
                "selected": position in selected_positions,
            }
            for position, frame_index in enumerate(candidate_indices)
        ]

        progress("生成 Manifest", 0.96)
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
            "candidates": candidate_records,
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
