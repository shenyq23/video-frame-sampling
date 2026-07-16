from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    total_frames: int
    duration_seconds: float


@dataclass(frozen=True)
class SamplingRequest:
    video_path: Path
    queries: list[str]
    algorithm: str
    max_frames: Optional[int] = None
    min_frames: int = 0
    multi_query_mode: str = "independent"
    candidate_sampling: str = "interval"
    sample_interval: float = 1.0
    threshold: float = 0.8
    score_threshold: float = 0.35
    std_threshold: float = -100.0
    max_depth: int = 5
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    batch_size: int = 16
    decode_threads: int = 2
    jpeg_quality: int = 92
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None


@dataclass(frozen=True)
class FrameRecord:
    order: int
    file: str
    frame_index: int
    timestamp_seconds: float
    score: Optional[float] = None
    query_scores: dict[str, float] = field(default_factory=dict)
    selected_by: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Selection:
    algorithm: str
    selected_indices: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    selected_by: dict[int, list[str]] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoContext:
    reader: Any
    info: VideoInfo
    candidate_indices: list[int]
    candidate_timestamps: list[float]
    frame_cache_dir: Path
    image_embeddings: Any = None
    query_embeddings: dict[str, Any] = field(default_factory=dict)

