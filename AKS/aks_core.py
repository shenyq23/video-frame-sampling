"""Shared, traceable implementation of Adaptive Keyframe Sampling (AKS).

The selection rule is derived from ``frame_select.py`` in this repository.
Two allocation modes are intentionally exposed:

``original``
    Uses the paper repository's independent ``int(N / 2**depth)`` quota for
    every terminal segment. This is the mode to use for benchmark reproduction.

``robust``
    Uses the same segmentation tree and relevance ranking, then redistributes
    integer-rounding/capacity leftovers so that the requested budget is filled
    whenever enough candidates exist. This is the recommended application mode.

Both modes fix non-algorithmic failures in the original script: JSON lists are
converted to NumPy arrays and constant scores are normalized safely.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


AKSMode = Literal["original", "robust"]


@dataclass(frozen=True)
class Segment:
    """One terminal temporal segment in the AKS binary partition tree."""

    scores: np.ndarray
    frame_indices: np.ndarray
    candidate_positions: np.ndarray
    depth: int


@dataclass(frozen=True)
class SegmentTrace:
    """Serializable information explaining one segment's allocation."""

    depth: int
    candidate_start: int
    candidate_end: int
    frame_start: int
    frame_end: int
    candidate_count: int
    quota: int


@dataclass(frozen=True)
class AKSSelection:
    """Selected original-video frame indices plus an allocation trace."""

    frame_indices: list[int]
    mode: AKSMode
    requested_frames: int
    candidate_count: int
    segments: list[SegmentTrace]


def normalize_scores(scores: Sequence[float]) -> np.ndarray:
    """Min-max normalize scores, safely handling empty and constant inputs."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be a one-dimensional sequence")
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    span = float(values.max() - values.min())
    if span == 0.0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def _top_positions(scores: np.ndarray, count: int) -> list[int]:
    """Match the repository's stable ``heapq.nlargest`` score ranking."""

    return heapq.nlargest(count, range(len(scores)), scores.__getitem__)


def _terminal_segments(
    segments: list[Segment],
    target_frames: int,
    threshold: float,
    std_threshold: float,
    max_depth: int,
    mode: AKSMode,
) -> list[Segment]:
    """Mirror the original ``meanstd`` breadth-recursive partition order."""

    terminal: list[Segment] = []
    split: list[Segment] = []
    for segment in segments:
        scores = segment.scores
        if scores.size == 0:
            continue

        top = _top_positions(scores, min(target_frames, scores.size))
        top_mean = float(np.mean([scores[position] for position in top]))
        mean_diff = top_mean - float(scores.mean())
        peaked = mean_diff > threshold and float(scores.std()) > std_threshold

        # Robust mode stops unsplittable leaves. Original mode deliberately
        # carries a one-frame leaf to max_depth, matching the source script's
        # quota behavior for small, non-standard budgets.
        if (
            peaked
            or segment.depth >= max_depth
            or (mode == "robust" and scores.size < 2)
        ):
            terminal.append(segment)
            continue

        middle = scores.size // 2
        split.extend(
            [
                Segment(
                    scores[:middle],
                    segment.frame_indices[:middle],
                    segment.candidate_positions[:middle],
                    segment.depth + 1,
                ),
                Segment(
                    scores[middle:],
                    segment.frame_indices[middle:],
                    segment.candidate_positions[middle:],
                    segment.depth + 1,
                ),
            ]
        )

    if split:
        terminal.extend(
            _terminal_segments(
                split,
                target_frames,
                threshold,
                std_threshold,
                max_depth,
                mode,
            )
        )
    return terminal


def _original_quotas(segments: Sequence[Segment], target: int) -> list[int]:
    return [min(int(target / (2**segment.depth)), len(segment.scores)) for segment in segments]


def _robust_quotas(segments: Sequence[Segment], target: int) -> list[int]:
    raw = [target / (2**segment.depth) for segment in segments]
    quotas = [min(int(value), len(segment.scores)) for value, segment in zip(raw, segments)]

    # Largest-remainder allocation first preserves the binary-tree weights.
    # Capacity left by a short segment is then redistributed deterministically.
    while sum(quotas) < target:
        candidates = [
            index for index, segment in enumerate(segments) if quotas[index] < len(segment.scores)
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda index: (
                raw[index] - quotas[index],
                raw[index],
                -int(segments[index].candidate_positions[0]),
            ),
        )
        quotas[best] += 1
    return quotas


def select_frames(
    scores: Sequence[float],
    frame_indices: Sequence[int],
    max_num_frames: int = 64,
    threshold: float = 0.8,
    std_threshold: float = -100.0,
    max_depth: int = 5,
    mode: AKSMode = "original",
) -> AKSSelection:
    """Select query-relevant frame indices with the shared AKS algorithm.

    ``frame_indices`` must be in temporal candidate order. Returned indices are
    sorted chronologically, matching the original repository's output contract.
    """

    if mode not in ("original", "robust"):
        raise ValueError("mode must be 'original' or 'robust'")
    if max_num_frames <= 0:
        raise ValueError("max_num_frames must be positive")
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")
    if len(scores) != len(frame_indices):
        raise ValueError("scores and frame_indices must have the same length")

    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("frame_indices must be a one-dimensional sequence")
    if indices.size and np.any(indices < 0):
        raise ValueError("frame_indices cannot contain negative values")
    if indices.size > 1 and np.any(indices[1:] < indices[:-1]):
        raise ValueError("frame_indices must be in chronological order")

    candidate_count = len(scores)
    target = min(max_num_frames, candidate_count)
    if target == 0:
        return AKSSelection([], mode, max_num_frames, 0, [])
    if candidate_count < max_num_frames or (
        mode == "robust" and candidate_count == max_num_frames
    ):
        selected = sorted(int(value) for value in indices)
        trace = [
            SegmentTrace(
                depth=0,
                candidate_start=0,
                candidate_end=candidate_count - 1,
                frame_start=int(indices[0]),
                frame_end=int(indices[-1]),
                candidate_count=candidate_count,
                quota=candidate_count,
            )
        ]
        return AKSSelection(selected, mode, max_num_frames, candidate_count, trace)

    normalized = normalize_scores(scores)
    positions = np.arange(candidate_count, dtype=np.int64)
    segments = _terminal_segments(
        [Segment(normalized, indices, positions, 0)],
        max_num_frames,
        threshold,
        std_threshold,
        max_depth,
        mode,
    )
    quotas = (
        _original_quotas(segments, target)
        if mode == "original"
        else _robust_quotas(segments, target)
    )

    selected: list[int] = []
    trace: list[SegmentTrace] = []
    for segment, quota in zip(segments, quotas):
        local_positions = _top_positions(segment.scores, quota)
        selected.extend(int(segment.frame_indices[position]) for position in local_positions)
        trace.append(
            SegmentTrace(
                depth=segment.depth,
                candidate_start=int(segment.candidate_positions[0]),
                candidate_end=int(segment.candidate_positions[-1]),
                frame_start=int(segment.frame_indices[0]),
                frame_end=int(segment.frame_indices[-1]),
                candidate_count=len(segment.scores),
                quota=quota,
            )
        )

    return AKSSelection(
        frame_indices=sorted(selected),
        mode=mode,
        requested_frames=max_num_frames,
        candidate_count=candidate_count,
        segments=trace,
    )


def select_frame_indices(
    scores: Sequence[float],
    frame_indices: Sequence[int],
    **kwargs,
) -> list[int]:
    """Convenience wrapper returning only selected indices."""

    return select_frames(scores, frame_indices, **kwargs).frame_indices
