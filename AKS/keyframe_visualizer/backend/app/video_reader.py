from __future__ import annotations

from pathlib import Path
import threading
from typing import Sequence

import numpy as np


_ACTIVE_SOURCES: dict[str, "VideoSource"] = {}
_ACTIVE_SOURCES_LOCK = threading.RLock()


class VideoSource:
    fps: float
    total_frames: int
    backend: str

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        """Release native decoder handles. Implementations must be idempotent."""


class DecordVideoSource(VideoSource):
    def __init__(self, path: Path, decode_threads: int):
        from decord import VideoReader, cpu

        self.reader = VideoReader(str(path), ctx=cpu(0), num_threads=decode_threads)
        self.fps = float(self.reader.get_avg_fps())
        self.total_frames = len(self.reader)
        self.backend = "decord"

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        return self.reader.get_batch(list(indices)).asnumpy()

    def close(self) -> None:
        reader = getattr(self, "reader", None)
        if reader is not None:
            self.reader = None
            del reader


class OpenCVVideoSource(VideoSource):
    """Portable fallback for environments where decord has no wheel."""

    def __init__(self, path: Path, decode_threads: int):
        import cv2

        cv2.setNumThreads(decode_threads)
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"无法打开视频：{path.name}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.backend = "opencv"

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        frames: list[np.ndarray] = []
        for index in indices:
            self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError(f"无法解码视频第 {index} 帧")
            frames.append(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
        return np.stack(frames)

    def close(self) -> None:
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.release()
            self.capture = None

    def __del__(self) -> None:
        self.close()


def open_video(path: Path, decode_threads: int) -> VideoSource:
    try:
        source: VideoSource = DecordVideoSource(path, decode_threads)
    except ImportError:
        try:
            source = OpenCVVideoSource(path, decode_threads)
        except ImportError as error:
            raise RuntimeError(
                "缺少视频解码器。请安装 decord 或 opencv-python-headless。"
            ) from error
    key = str(path.expanduser().resolve())
    with _ACTIVE_SOURCES_LOCK:
        previous = _ACTIVE_SOURCES.pop(key, None)
        if previous is not None:
            previous.close()
        _ACTIVE_SOURCES[key] = source
    return source


def release_video(path: Path) -> None:
    key = str(path.expanduser().resolve())
    with _ACTIVE_SOURCES_LOCK:
        source = _ACTIVE_SOURCES.pop(key, None)
    if source is not None:
        try:
            source.close()
        except Exception:
            # Cleanup must never hide the original task result or exception.
            pass
