from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .schemas import VideoContext, VideoInfo


def open_video(video_path: Path, decode_threads: int, frame_cache_dir: Path) -> VideoContext:
    try:
        from decord import VideoReader, cpu
    except ImportError:
        reader = _OpenCVReader(video_path)
    else:
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=decode_threads)
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Could not determine a valid FPS for {video_path}")
    if len(reader) <= 0:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    info = VideoInfo(
        path=str(video_path),
        fps=fps,
        total_frames=len(reader),
        duration_seconds=len(reader) / fps,
    )
    return VideoContext(reader, info, [], [], frame_cache_dir)


class _ArrayBatch:
    def __init__(self, arrays: np.ndarray):
        self._arrays = arrays

    def asnumpy(self) -> np.ndarray:
        return self._arrays


class _OpenCVReader:
    """Small Decord-compatible fallback for environments without Decord."""

    def __init__(self, video_path: Path):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Video decoding requires decord or opencv-python."
            ) from error
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(video_path))
        if not self._capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._length = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self) -> int:
        return self._length

    def get_avg_fps(self) -> float:
        return self._fps

    def get_batch(self, indices) -> _ArrayBatch:
        arrays = []
        for index in indices:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = self._capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame {index}")
            arrays.append(self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB))
        return _ArrayBatch(np.stack(arrays, axis=0))


def candidate_frame_indices(
    total_frames: int,
    fps: float,
    mode: str,
    sample_interval: float,
) -> list[int]:
    if total_frames <= 0:
        return []
    if mode == "original":
        integer_fps = int(fps)
        if integer_fps <= 0:
            raise ValueError("original candidate sampling requires FPS >= 1")
        count = int(total_frames / integer_fps)
        return [position * integer_fps for position in range(count)]
    if mode == "interval":
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        step = max(1, int(round(fps * sample_interval)))
        return list(range(0, total_frames, step))
    raise ValueError("candidate_sampling must be 'original' or 'interval'")


def uniform_frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames < 0:
        raise ValueError("total_frames cannot be negative")
    if count <= 0:
        raise ValueError("count must be positive")
    target = min(total_frames, count)
    if target == 0:
        return []
    if target == 1:
        return [0]
    last = total_frames - 1
    return [(position * last) // (target - 1) for position in range(target)]
