"""Run AKS keyframe selection on one video and one text query.

This is a standalone path: video -> query relevance scores -> AKS -> JPG files.
It intentionally does not invoke an MLLM.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Segment:
    scores: np.ndarray
    frame_indices: np.ndarray
    depth: int


def _normalize(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values
    span = float(values.max() - values.min())
    if span == 0.0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def _adaptive_segments(
    segments: list[Segment],
    target_frames: int,
    threshold: float,
    std_threshold: float,
    max_depth: int,
) -> list[Segment]:
    """Recursively split flat regions and keep regions with strong peaks.

    This is the repository's ``meanstd`` procedure expressed iteratively. Keeping
    the segment tree as leaves preserves the temporal-coverage allocation used
    by AKS.
    """
    leaves: list[Segment] = []
    pending = list(segments)
    while pending:
        segment = pending.pop(0)
        scores = segment.scores
        if scores.size == 0:
            continue

        top_count = min(target_frames, scores.size)
        top_scores = np.partition(scores, scores.size - top_count)[-top_count:]
        mean_diff = float(top_scores.mean() - scores.mean())
        peaked = mean_diff > threshold and float(scores.std()) > std_threshold

        # A one-frame segment cannot be split further.
        if peaked or segment.depth >= max_depth or scores.size < 2:
            leaves.append(segment)
            continue

        middle = scores.size // 2
        pending.extend(
            [
                Segment(scores[:middle], segment.frame_indices[:middle], segment.depth + 1),
                Segment(scores[middle:], segment.frame_indices[middle:], segment.depth + 1),
            ]
        )
    return leaves


def select_aks_frames(
    scores: Sequence[float],
    frame_indices: Sequence[int],
    max_num_frames: int = 32,
    threshold: float = 0.8,
    std_threshold: float = -100.0,
    max_depth: int = 5,
) -> list[int]:
    """Select frame indices using the AKS implementation in ``frame_select.py``."""
    if max_num_frames <= 0:
        raise ValueError("max_num_frames must be positive")
    if len(scores) != len(frame_indices):
        raise ValueError("scores and frame_indices must have the same length")
    if len(scores) <= max_num_frames:
        return sorted(int(index) for index in frame_indices)

    normalized = _normalize(scores)
    indices = np.asarray(frame_indices, dtype=np.int64)
    leaves = _adaptive_segments(
        [Segment(normalized, indices, 0)],
        max_num_frames,
        threshold,
        std_threshold,
        max_depth,
    )

    # The original code uses int(N / 2**depth) independently for every leaf.
    # That can allocate zero frames when N < 2**max_depth. Largest-remainder
    # allocation retains the same binary-tree weights while honoring small N.
    target = min(max_num_frames, len(scores))
    raw_quotas = [target / (2**leaf.depth) for leaf in leaves]
    quotas = [min(int(raw), leaf.scores.size) for raw, leaf in zip(raw_quotas, leaves)]
    while sum(quotas) < target:
        candidates = [i for i, leaf in enumerate(leaves) if quotas[i] < leaf.scores.size]
        if not candidates:
            break
        best = max(candidates, key=lambda i: (raw_quotas[i] - quotas[i], raw_quotas[i], -i))
        quotas[best] += 1

    selected: list[int] = []
    for leaf, quota in zip(leaves, quotas):
        if quota <= 0:
            continue
        local = np.argpartition(leaf.scores, leaf.scores.size - quota)[-quota:]
        selected.extend(int(value) for value in leaf.frame_indices[local])
    return sorted(set(selected))


def _choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _score_with_clip(
    video_reader,
    frame_indices: Sequence[int],
    query: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> list[float]:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading relevance model: {model_name} ({device})")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    text_inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.inference_mode():
        text_feature = model.get_text_features(**text_inputs)
        text_feature = torch.nn.functional.normalize(text_feature, dim=-1)

    scores: list[float] = []
    total_batches = math.ceil(len(frame_indices) / batch_size)
    for batch_number, start in enumerate(range(0, len(frame_indices), batch_size), 1):
        batch_indices = frame_indices[start : start + batch_size]
        arrays = video_reader.get_batch(list(batch_indices)).asnumpy()
        images = [Image.fromarray(array) for array in arrays]
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        with torch.inference_mode():
            image_features = model.get_image_features(**image_inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            batch_scores = image_features @ text_feature.T
        scores.extend(batch_scores[:, 0].detach().cpu().float().tolist())
        print(f"Scoring sampled frames: {batch_number}/{total_batches}", end="\r")
    print()
    return scores


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "video"


def run(args: argparse.Namespace) -> Path:
    try:
        from decord import VideoReader, cpu
    except ImportError as error:
        raise SystemExit(
            "Missing dependency 'decord'. Install the packages in "
            "requirements-keyframes.txt first."
        ) from error

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video does not exist: {video_path}")
    query = args.query
    if args.query_file:
        query = Path(args.query_file).expanduser().read_text(encoding="utf-8").strip()
    if not query or not query.strip():
        raise SystemExit("Provide a non-empty --query or --query-file")

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=args.decode_threads)
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise SystemExit(f"Could not determine a valid FPS for {video_path}")
    step = max(1, int(round(fps * args.sample_interval)))
    sampled_indices = list(range(0, len(reader), step))
    if not sampled_indices:
        raise SystemExit("The video contains no decodable frames")

    device = _choose_device(args.device)
    scores = _score_with_clip(
        reader,
        sampled_indices,
        query.strip(),
        args.model_name,
        device,
        args.batch_size,
    )
    selected_indices = select_aks_frames(
        scores,
        sampled_indices,
        args.max_num_frames,
        args.threshold,
        args.std_threshold,
        args.max_depth,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("aks_output").resolve() / _safe_stem(video_path.stem)
    )
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    score_by_index = dict(zip(sampled_indices, scores))
    selected_arrays = reader.get_batch(selected_indices).asnumpy()
    records = []
    for order, (frame_index, array) in enumerate(zip(selected_indices, selected_arrays), 1):
        timestamp = frame_index / fps
        filename = f"{order:03d}_t{timestamp:010.3f}_f{frame_index}.jpg"
        Image.fromarray(array).save(frames_dir / filename, quality=args.jpeg_quality)
        records.append(
            {
                "order": order,
                "file": f"frames/{filename}",
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp, 6),
                "relevance_score": score_by_index[frame_index],
            }
        )

    manifest = {
        "video": str(video_path),
        "query": query.strip(),
        "model": args.model_name,
        "device": device,
        "video_fps": fps,
        "video_total_frames": len(reader),
        "sample_interval_seconds": args.sample_interval,
        "sampled_frame_count": len(sampled_indices),
        "requested_keyframe_count": args.max_num_frames,
        "selected_keyframe_count": len(records),
        "aks": {
            "threshold": args.threshold,
            "std_threshold": args.std_threshold,
            "max_depth": args.max_depth,
        },
        "keyframes": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Selected {len(records)} keyframes -> {frames_dir}")
    print(f"Manifest -> {output_dir / 'manifest.json'}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select and export AKS keyframes for one video and query (no MLLM)."
    )
    parser.add_argument("--video", required=True, help="Path to a local video file")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="Text describing the information to find")
    query_group.add_argument("--query-file", help="UTF-8 text file containing the query")
    parser.add_argument("--output-dir", help="Output directory (default: aks_output/<video>)")
    parser.add_argument("--max-num-frames", type=int, default=32)
    parser.add_argument(
        "--sample-interval", type=float, default=1.0, help="Candidate-frame interval in seconds"
    )
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.8, help="AKS peak threshold t1")
    parser.add_argument("--std-threshold", type=float, default=-100.0, help="AKS threshold t2")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_interval <= 0:
        raise SystemExit("--sample-interval must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_num_frames <= 0:
        raise SystemExit("--max-num-frames must be positive")
    if args.max_depth < 0:
        raise SystemExit("--max-depth cannot be negative")
    if args.decode_threads <= 0:
        raise SystemExit("--decode-threads must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")
    run(args)


if __name__ == "__main__":
    main()
