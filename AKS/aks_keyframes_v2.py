"""Custom-video AKS entry point backed by the shared ``aks_core`` module.

The existing ``aks_keyframes.py`` is intentionally kept as the legacy chain.
This V2 path adds traceable original/robust modes without loading an MLLM:

    video -> candidate frames -> pluggable relevance scores -> shared AKS -> JPG + manifest
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from aks_core import select_frames


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_candidate_indices(
    total_frames: int,
    fps: float,
    sampling_mode: str,
    sample_interval: float,
) -> list[int]:
    """Create candidates using either repository-exact or interval sampling."""

    if total_frames <= 0:
        return []
    if sampling_mode == "original":
        integer_fps = int(fps)
        if integer_fps <= 0:
            raise ValueError("original sampling requires FPS >= 1")
        # Exact structure used by feature_extract.py: range(int(len/fps)).
        candidate_count = int(total_frames / integer_fps)
        return [position * integer_fps for position in range(candidate_count)]
    if sampling_mode == "interval":
        step = max(1, int(round(fps * sample_interval)))
        return list(range(0, total_frames, step))
    raise ValueError("sampling_mode must be 'original' or 'interval'")


def sample_uniform_indices(frame_indices: Sequence[int], count: int) -> list[int]:
    """Select ``count`` temporally uniform frames from an ordered candidate pool."""

    if count <= 0 or len(frame_indices) == 0:
        return []
    if count >= len(frame_indices):
        return [int(index) for index in frame_indices]
    if count == 1:
        return [int(frame_indices[len(frame_indices) // 2])]

    last_position = len(frame_indices) - 1
    positions = [round(order * last_position / (count - 1)) for order in range(count)]
    return [int(frame_indices[position]) for position in positions]


def save_frame_set(
    video_reader,
    frame_indices: Sequence[int],
    fps: float,
    output_dir: Path,
    relative_dir: str,
    jpeg_quality: int,
    batch_size: int,
    score_by_index: dict[int, float],
) -> list[dict[str, object]]:
    """Save a frame set in bounded batches and return manifest records."""

    from PIL import Image

    frames_dir = output_dir / relative_dir
    frames_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for start in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[start : start + batch_size]
        arrays = video_reader.get_batch(list(batch_indices)).asnumpy()
        for offset, (frame_index, array) in enumerate(zip(batch_indices, arrays), start + 1):
            timestamp = frame_index / fps
            filename = f"{offset:03d}_t{timestamp:010.3f}_f{frame_index}.jpg"
            Image.fromarray(array).save(frames_dir / filename, quality=jpeg_quality)
            record: dict[str, object] = {
                "order": offset,
                "file": f"{relative_dir}/{filename}",
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp, 6),
            }
            if frame_index in score_by_index:
                record["relevance_score"] = score_by_index[frame_index]
            records.append(record)
    return records


def score_candidate_frames(
    video_reader,
    frame_indices: Sequence[int],
    query: str,
    feature_backend: str,
    feature_config: str | None,
    model_name: str,
    device: str,
    batch_size: int,
) -> tuple[list[float], dict[str, object]]:
    """Score candidates with a configured local or remote feature backend."""

    from PIL import Image
    from feature_backends import create_relevance_scorer, load_feature_config

    config = load_feature_config(feature_config)
    scorer = create_relevance_scorer(
        feature_backend,
        config,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    scorer.prepare_query(query)

    scores: list[float] = []
    total_batches = math.ceil(len(frame_indices) / batch_size)
    for batch_number, start in enumerate(range(0, len(frame_indices), batch_size), 1):
        batch_indices = frame_indices[start : start + batch_size]
        arrays = video_reader.get_batch(list(batch_indices)).asnumpy()
        images = [Image.fromarray(array) for array in arrays]
        scores.extend(scorer.score_images(images))
        print(f"Scoring candidate frames: {batch_number}/{total_batches}", end="\r")
    print()
    return scores, scorer.metadata


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "video"


def run(args: argparse.Namespace) -> Path:
    try:
        from decord import VideoReader, cpu
    except ImportError as error:
        raise SystemExit(
            "Missing dependency 'decord'. Install requirements-keyframes.txt first."
        ) from error

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video does not exist: {video_path}")

    query = args.query
    if args.query_file:
        query = Path(args.query_file).expanduser().read_text(encoding="utf-8").strip()
    if not query or not query.strip():
        raise SystemExit("Provide a non-empty --query or --query-file")
    query = query.strip()

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=args.decode_threads)
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise SystemExit(f"Could not determine a valid FPS for {video_path}")
    candidate_indices = sample_candidate_indices(
        len(reader), fps, args.candidate_sampling, args.sample_interval
    )
    if not candidate_indices:
        raise SystemExit("The video contains no candidate frames")

    device = choose_device(args.device) if args.feature_backend == "clip" else "remote"
    scores, feature_metadata = score_candidate_frames(
        reader,
        candidate_indices,
        query,
        args.feature_backend,
        args.feature_config,
        args.model_name,
        device,
        args.batch_size,
    )
    selection = select_frames(
        scores,
        candidate_indices,
        max_num_frames=args.max_num_frames,
        threshold=args.threshold,
        std_threshold=args.std_threshold,
        max_depth=args.max_depth,
        mode=args.aks_mode,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("aks_output_v2").resolve() / safe_stem(video_path.stem)
    )
    score_by_index = dict(zip(candidate_indices, scores))
    selected_indices = selection.frame_indices
    records = save_frame_set(
        reader,
        selected_indices,
        fps,
        output_dir,
        "frames",
        args.jpeg_quality,
        args.batch_size,
        score_by_index,
    )

    uniform_records: list[dict[str, object]] = []
    if args.save_uniform_baseline:
        uniform_indices = sample_uniform_indices(candidate_indices, len(selected_indices))
        uniform_records = save_frame_set(
            reader,
            uniform_indices,
            fps,
            output_dir,
            "uniform_frames",
            args.jpeg_quality,
            args.batch_size,
            score_by_index,
        )

    candidate_records: list[dict[str, object]] = []
    if args.save_candidate_frames:
        candidate_records = save_frame_set(
            reader,
            candidate_indices,
            fps,
            output_dir,
            "candidate_frames",
            args.jpeg_quality,
            args.batch_size,
            score_by_index,
        )

    manifest = {
        "algorithm": "AKS",
        "implementation": {
            "entry_point": "aks_keyframes_v2.py",
            "shared_core": "aks_core.py",
            "mode": selection.mode,
            "original_source": "frame_select.py",
        },
        "video": str(video_path),
        "query": query,
        "relevance_model": args.model_name if args.feature_backend == "clip" else None,
        "feature_extraction": feature_metadata,
        "device": device if args.feature_backend == "clip" else None,
        "video_fps": fps,
        "video_total_frames": len(reader),
        "candidate_sampling": {
            "mode": args.candidate_sampling,
            "interval_seconds": args.sample_interval
            if args.candidate_sampling == "interval"
            else None,
            "candidate_count": len(candidate_indices),
        },
        "aks": {
            "requested_keyframe_count": args.max_num_frames,
            "selected_keyframe_count": len(records),
            "threshold_t1": args.threshold,
            "std_threshold_t2": args.std_threshold,
            "max_depth": args.max_depth,
            "segments": [asdict(segment) for segment in selection.segments],
        },
        "keyframes": records,
        "uniform_baseline": {
            "saved": args.save_uniform_baseline,
            "selection_rule": "uniformly spaced over the AKS candidate pool",
            "frame_count": len(uniform_records),
            "frames": uniform_records,
        },
        "candidate_frames": {
            "saved": args.save_candidate_frames,
            "frame_count": len(candidate_records),
            "frames": candidate_records,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if len(records) < min(args.max_num_frames, len(candidate_indices)):
        print(
            f"Note: {args.aks_mode} mode selected {len(records)} frames from a "
            f"budget of {args.max_num_frames}. Use --aks-mode robust to fill the budget."
        )
    print(f"Selected {len(records)} keyframes -> {output_dir / 'frames'}")
    if uniform_records:
        print(
            f"Saved {len(uniform_records)} same-budget uniform frames -> "
            f"{output_dir / 'uniform_frames'}"
        )
    if candidate_records:
        print(
            f"Saved all {len(candidate_records)} candidate frames -> "
            f"{output_dir / 'candidate_frames'}"
        )
    print(f"Traceable manifest -> {manifest_path}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select AKS keyframes through the shared, traceable V2 pipeline."
    )
    parser.add_argument("--video", required=True, help="Path to a local video file")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="Text query used for frame retrieval")
    query_group.add_argument("--query-file", help="UTF-8 file containing the query")
    parser.add_argument("--output-dir", help="Default: aks_output_v2/<video name>")
    parser.add_argument("--max-num-frames", type=int, default=32)
    parser.add_argument(
        "--aks-mode",
        choices=("original", "robust"),
        default="robust",
        help="original reproduces repository quotas; robust fills the requested budget",
    )
    parser.add_argument(
        "--candidate-sampling",
        choices=("original", "interval"),
        default="interval",
        help="original uses int(FPS); interval uses --sample-interval seconds",
    )
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--feature-backend",
        choices=("clip", "pangu", "mep", "http", "python"),
        default="clip",
        help="feature scorer backend; python loads a custom plugin from JSON config",
    )
    parser.add_argument(
        "--feature-config",
        help="JSON config for pangu, mep, or generic http backends",
    )
    parser.add_argument("--model-name", default="./models/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.8, help="AKS t1")
    parser.add_argument("--std-threshold", type=float, default=-100.0, help="AKS t2")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--save-uniform-baseline",
        action="store_true",
        help="save a query-independent, same-count uniform baseline",
    )
    parser.add_argument(
        "--save-candidate-frames",
        action="store_true",
        help="save every uniformly sampled frame in the AKS candidate pool",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_num_frames <= 0:
        raise SystemExit("--max-num-frames must be positive")
    if args.sample_interval <= 0:
        raise SystemExit("--sample-interval must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.decode_threads <= 0:
        raise SystemExit("--decode-threads must be positive")
    if args.max_depth < 0:
        raise SystemExit("--max-depth cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")
    run(args)


if __name__ == "__main__":
    main()
