from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import samplers as _samplers  # noqa: F401 - imports register implementations
from .exporter import export_selection
from .features import CLIPFeatureStore
from .registry import get_sampler
from .schemas import SamplingRequest, Selection
from .video import candidate_frame_indices, open_video


ProgressCallback = Callable[[float, str], None]


def _validate_request(request: SamplingRequest) -> None:
    if not request.video_path.is_file():
        raise ValueError(f"Video does not exist: {request.video_path}")
    if request.max_frames is not None and request.max_frames <= 0:
        raise ValueError("max_frames must be positive or omitted")
    if request.min_frames < 0:
        raise ValueError("min_frames cannot be negative")
    if request.max_frames is not None and request.min_frames > request.max_frames:
        raise ValueError("min_frames cannot exceed max_frames")
    if request.multi_query_mode not in ("independent", "union", "joint"):
        raise ValueError("multi_query_mode must be independent, union, or joint")
    if any(not query.strip() for query in request.queries):
        raise ValueError("queries cannot contain empty strings")
    if len(set(request.queries)) != len(request.queries):
        raise ValueError("queries must be unique")
    if request.batch_size <= 0 or request.decode_threads <= 0:
        raise ValueError("batch_size and decode_threads must be positive")
    if not 1 <= request.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")


def _join_queries(queries: list[str]) -> str:
    lines = [f"{index}. {query}" for index, query in enumerate(queries, 1)]
    return "Find visual evidence for all of the following:\n" + "\n".join(lines)


def _union_selections(
    selections: list[tuple[str, Selection]],
    request: SamplingRequest,
    candidate_indices: list[int],
    scores_by_query: dict[str, list[float]],
) -> Selection:
    selected_by: dict[int, list[str]] = {}
    traces = []
    for query, selection in selections:
        traces.append(selection.trace)
        for frame_index in selection.selected_indices:
            selected_by.setdefault(frame_index, []).append(query)

    indices = sorted(selected_by)
    position_by_index = {value: position for position, value in enumerate(candidate_indices)}
    combined_scores = {
        frame_index: max(
            scores_by_query[query][position_by_index[frame_index]]
            for query in scores_by_query
        )
        for frame_index in indices
    }
    if request.max_frames is not None and len(indices) > request.max_frames:
        ranked = sorted(indices, key=lambda index: (-combined_scores[index], index))
        kept = set(ranked[: request.max_frames])
        indices = sorted(kept)
        selected_by = {index: selected_by[index] for index in indices}
        combined_scores = {index: combined_scores[index] for index in indices}

    return Selection(
        algorithm=f"{request.algorithm}:union",
        selected_indices=indices,
        scores=combined_scores,
        selected_by=selected_by,
        trace={
            "multi_query_mode": "union",
            "per_query_traces": traces,
            "global_max_frames": request.max_frames,
        },
    )


def _request_manifest(request: SamplingRequest) -> dict:
    return {
        "video_path": str(request.video_path),
        "queries": request.queries,
        "algorithm": request.algorithm,
        "max_frames": request.max_frames,
        "min_frames": request.min_frames,
        "multi_query_mode": request.multi_query_mode,
        "candidate_sampling": request.candidate_sampling,
        "sample_interval": request.sample_interval,
        "threshold_t1": request.threshold,
        "score_threshold": request.score_threshold,
        "std_threshold_t2": request.std_threshold,
        "max_depth": request.max_depth,
        "model_name": request.model_name,
        "device": request.device,
        "batch_size": request.batch_size,
    }


def run_sampling(
    request: SamplingRequest,
    progress: ProgressCallback | None = None,
) -> dict:
    """Execute one frame-sampling job and return its manifest and paths."""

    _validate_request(request)
    callback = progress or (lambda ratio, message: print(f"[{ratio:5.1%}] {message}"))
    sampler_class = get_sampler(request.algorithm)
    sampler = sampler_class()

    project_dir = Path(__file__).resolve().parents[2]
    output_root = (request.output_dir or project_dir / "outputs").expanduser().resolve()
    cache_root = (request.cache_dir or project_dir / "cache").expanduser().resolve()
    run_dir = output_root / (
        f"{request.video_path.stem}_{request.algorithm}_{uuid.uuid4().hex[:8]}"
    )
    callback(0.03, "Opening video")
    context = open_video(request.video_path, request.decode_threads, cache_root / "frames")

    query_scores: dict[str, list[float]] = {}
    scoring_queries = request.queries
    joint_query = None
    if sampler.query_aware:
        if not request.queries:
            raise ValueError(f"{request.algorithm} requires at least one query")
        context.candidate_indices = candidate_frame_indices(
            context.info.total_frames,
            context.info.fps,
            request.candidate_sampling,
            request.sample_interval,
        )
        if not context.candidate_indices:
            raise ValueError("Candidate sampling produced no frames")
        context.candidate_timestamps = [
            index / context.info.fps for index in context.candidate_indices
        ]
        if request.multi_query_mode == "joint" and len(request.queries) > 1:
            joint_query = _join_queries(request.queries)
            scoring_queries = [joint_query]
        feature_store = CLIPFeatureStore(
            cache_root,
            request.model_name,
            request.device,
            request.batch_size,
            callback,
        )
        query_scores = feature_store.score_queries(context, scoring_queries)

    callback(0.72, f"Running sampler: {request.algorithm}")
    grouped: list[tuple[str, Selection]] = []
    if not sampler.query_aware:
        grouped.append(("global", sampler.select(context, request, None, None)))
    elif joint_query is not None:
        grouped.append(
            ("joint", sampler.select(context, request, joint_query, query_scores[joint_query]))
        )
    else:
        per_query = [
            (query, sampler.select(context, request, query, query_scores[query]))
            for query in request.queries
        ]
        if request.multi_query_mode == "union" and len(per_query) > 1:
            grouped.append(
                (
                    "union",
                    _union_selections(
                        per_query,
                        request,
                        context.candidate_indices,
                        query_scores,
                    ),
                )
            )
        else:
            grouped.extend(
                (f"query_{index:03d}", selection)
                for index, (_query, selection) in enumerate(per_query, 1)
            )

    callback(0.82, "Exporting selected frames")
    group_manifests = []
    for group_name, selection in grouped:
        records, group_manifest_path = export_selection(
            context,
            selection,
            run_dir,
            group_name,
            query_scores,
            request.jpeg_quality,
        )
        group_manifests.append(
            {
                "name": group_name,
                "algorithm": selection.algorithm,
                "selected_frame_count": len(records),
                "manifest": str(group_manifest_path.relative_to(run_dir)),
                "frames": [asdict(record) for record in records],
            }
        )

    manifest = {
        "framework": "video-frame-sampling-demo",
        "request": _request_manifest(request),
        "video": asdict(context.info),
        "candidate_count": len(context.candidate_indices)
        if sampler.query_aware
        else context.info.total_frames,
        "groups": group_manifests,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    callback(1.0, f"Done: {manifest_path}")
    return {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }
