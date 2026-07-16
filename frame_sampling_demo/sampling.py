"""Frame extraction services used by the Gradio demo.

AKS intentionally remains a sibling module.  This file only adds the sibling
``AKS`` directory to ``sys.path`` when AKS is selected, so opening the demo or
using uniform sampling does not load Torch/Transformers/CLIP.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    total_frames: int
    duration_seconds: float


@dataclass(frozen=True)
class FrameRecord:
    order: int
    file: str
    frame_index: int
    timestamp_seconds: float
    relevance_score: Optional[float] = None


def uniform_frame_indices(total_frames: int, count: int) -> list[int]:
    """Return up to ``count`` unique indices spanning the complete video."""

    if total_frames < 0:
        raise ValueError("total_frames cannot be negative")
    if count <= 0:
        raise ValueError("count must be positive")
    target = min(total_frames, count)
    if target == 0:
        return []
    if target == 1:
        return [0]

    # Integer interpolation includes both endpoints and avoids floating-point
    # rounding differences across Python/NumPy versions.
    last = total_frames - 1
    return [(position * last) // (target - 1) for position in range(target)]


def _load_video(video_path: Path, decode_threads: int):
    try:
        from decord import VideoReader, cpu
    except ImportError as error:
        raise RuntimeError(
            "缺少 decord，请先安装 frame_sampling_demo/requirements.txt"
        ) from error

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=decode_threads)
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("无法读取有效的视频 FPS")
    if len(reader) <= 0:
        raise ValueError("视频中没有可读取的帧")
    return reader, VideoInfo(fps, len(reader), len(reader) / fps)


def _load_aks_modules():
    demo_dir = Path(__file__).resolve().parent
    aks_dir = demo_dir.parent / "AKS"
    if not (aks_dir / "aks_core.py").is_file():
        raise RuntimeError(f"未找到同级 AKS 模块：{aks_dir}")
    aks_path = str(aks_dir)
    if aks_path not in sys.path:
        sys.path.insert(0, aks_path)

    try:
        from aks_core import select_frames
        from aks_keyframes_v2 import (
            choose_device,
            sample_candidate_indices,
            score_with_clip,
        )
    except ImportError as error:
        raise RuntimeError(f"AKS 模块加载失败：{error}") from error
    return select_frames, choose_device, sample_candidate_indices, score_with_clip


def _save_frames(
    reader: Any,
    frame_indices: Sequence[int],
    fps: float,
    frames_dir: Path,
    scores: Optional[dict[int, float]],
    jpeg_quality: int,
) -> list[FrameRecord]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "缺少 Pillow，请先安装 frame_sampling_demo/requirements.txt"
        ) from error

    frames_dir.mkdir(parents=True, exist_ok=True)
    arrays = reader.get_batch(list(frame_indices)).asnumpy() if frame_indices else []
    records: list[FrameRecord] = []
    for order, (frame_index, array) in enumerate(zip(frame_indices, arrays), 1):
        timestamp = frame_index / fps
        filename = f"{order:03d}_t{timestamp:010.3f}_f{frame_index}.jpg"
        Image.fromarray(array).save(frames_dir / filename, quality=jpeg_quality)
        records.append(
            FrameRecord(
                order=order,
                file=f"frames/{filename}",
                frame_index=int(frame_index),
                timestamp_seconds=round(timestamp, 6),
                relevance_score=(scores or {}).get(int(frame_index)),
            )
        )
    return records


def extract_frames(
    video: Union[str, Path],
    method: str,
    frame_count: int,
    query: str = "",
    sample_interval: float = 1.0,
    aks_mode: str = "robust",
    device: str = "auto",
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 16,
    decode_threads: int = 2,
    jpeg_quality: int = 92,
    output_root: Optional[Union[str, Path]] = None,
    progress: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Run one extraction job and return UI-ready paths plus its manifest."""

    if not video:
        raise ValueError("请先上传视频")
    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        raise ValueError(f"视频不存在：{video_path}")
    if frame_count <= 0:
        raise ValueError("抽帧数量必须大于 0")
    if sample_interval <= 0:
        raise ValueError("候选帧间隔必须大于 0")
    if batch_size <= 0 or decode_threads <= 0:
        raise ValueError("batch_size 和 decode_threads 必须大于 0")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG 质量必须在 1 到 100 之间")

    callback = progress or (lambda _ratio, _message: None)
    callback(0.05, "正在读取视频")
    reader, info = _load_video(video_path, decode_threads)

    root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else Path(__file__).resolve().parent / "outputs"
    )
    run_dir = root / f"{video_path.stem}_{uuid.uuid4().hex[:8]}"
    frames_dir = run_dir / "frames"

    method_key = method.strip().lower()
    scores_by_index: Optional[dict[int, float]] = None
    method_manifest: dict[str, Any]
    if method_key in {"uniform", "均匀抽帧"}:
        callback(0.25, "正在计算均匀采样位置")
        selected_indices = uniform_frame_indices(info.total_frames, frame_count)
        method_manifest = {
            "name": "uniform",
            "requested_frame_count": frame_count,
            "selected_frame_count": len(selected_indices),
        }
    elif method_key in {"aks", "aks 关键帧"}:
        if not query.strip():
            raise ValueError("AKS 抽帧需要填写检索问题或画面描述")
        callback(0.15, "正在加载同级 AKS 模块")
        (
            select_frames,
            choose_device,
            sample_candidate_indices,
            score_with_clip,
        ) = _load_aks_modules()
        candidate_indices = sample_candidate_indices(
            info.total_frames, info.fps, "interval", sample_interval
        )
        if not candidate_indices:
            raise ValueError("没有生成候选帧")
        selected_device = choose_device(device)
        callback(0.25, f"正在使用 CLIP 为 {len(candidate_indices)} 个候选帧评分")
        scores = score_with_clip(
            reader,
            candidate_indices,
            query.strip(),
            model_name,
            selected_device,
            batch_size,
        )
        callback(0.78, "正在执行 AKS 自适应选帧")
        selection = select_frames(
            scores,
            candidate_indices,
            max_num_frames=frame_count,
            threshold=0.8,
            std_threshold=-100.0,
            max_depth=5,
            mode=aks_mode,
        )
        selected_indices = selection.frame_indices
        scores_by_index = dict(zip(candidate_indices, scores))
        method_manifest = {
            "name": "AKS",
            "module": str((Path(__file__).resolve().parent.parent / "AKS").resolve()),
            "mode": selection.mode,
            "query": query.strip(),
            "model_name": model_name,
            "device": selected_device,
            "sample_interval_seconds": sample_interval,
            "candidate_count": len(candidate_indices),
            "requested_frame_count": frame_count,
            "selected_frame_count": len(selected_indices),
            "segments": [asdict(segment) for segment in selection.segments],
        }
    else:
        raise ValueError(f"不支持的抽帧方式：{method}")

    callback(0.85, "正在导出图片")
    records = _save_frames(
        reader,
        selected_indices,
        info.fps,
        frames_dir,
        scores_by_index,
        jpeg_quality,
    )
    manifest = {
        "video": str(video_path),
        "video_info": asdict(info),
        "method": method_manifest,
        "keyframes": [asdict(record) for record in records],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive_path = Path(
        shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    ).resolve()
    callback(1.0, "抽帧完成")

    return {
        "gallery": [
            (
                str(run_dir / record.file),
                f"#{record.order} · {record.timestamp_seconds:.3f}s · 帧 {record.frame_index}",
            )
            for record in records
        ],
        "summary": (
            f"完成：从 {info.duration_seconds:.2f} 秒视频中导出 {len(records)} 帧，"
            f"结果目录为 {run_dir}"
        ),
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "archive_path": str(archive_path),
    }
