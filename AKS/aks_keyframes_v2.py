"""Custom-video AKS entry point backed by the shared ``aks_core`` module.

The existing ``aks_keyframes.py`` is intentionally kept as the legacy chain.
This V2 path adds traceable original/robust modes without loading an MLLM:

    video -> candidate frames -> CLIP scores -> shared AKS -> JPG + manifest
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


def score_with_clip(
    video_reader,
    frame_indices: Sequence[int],
    query: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> list[float]:
    """Compute cosine similarity exactly as the repository's CLIP branch."""

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading relevance model: {model_name} ({device})")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    text_inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {name: value.to(device) for name, value in text_inputs.items()}
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)

    scores: list[float] = []
    total_batches = math.ceil(len(frame_indices) / batch_size)
    for batch_number, start in enumerate(range(0, len(frame_indices), batch_size), 1):
        batch_indices = frame_indices[start : start + batch_size]
        arrays = video_reader.get_batch(list(batch_indices)).asnumpy()
        image_inputs = processor(
            images=[Image.fromarray(array) for array in arrays],
            return_tensors="pt",
            padding=True,
        )
        image_inputs = {name: value.to(device) for name, value in image_inputs.items()}
        with torch.inference_mode():
            image_features = model.get_image_features(**image_inputs)
            batch_scores = torch.nn.functional.cosine_similarity(
                text_features.expand_as(image_features), image_features, dim=-1
            )
        scores.extend(batch_scores.detach().cpu().float().tolist())
        print(f"Scoring candidate frames: {batch_number}/{total_batches}", end="\r")
    print()
    return scores


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

    device = choose_device(args.device)
    scores = score_with_clip(
        reader,
        candidate_indices,
        query,
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
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    score_by_index = dict(zip(candidate_indices, scores))
    selected_indices = selection.frame_indices
    selected_arrays = (
        reader.get_batch(selected_indices).asnumpy() if selected_indices else []
    )
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
        "algorithm": "AKS",
        "implementation": {
            "entry_point": "aks_keyframes_v2.py",
            "shared_core": "aks_core.py",
            "mode": selection.mode,
            "original_source": "frame_select.py",
        },
        "video": str(video_path),
        "query": query,
        "relevance_model": args.model_name,
        "device": device,
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
    print(f"Selected {len(records)} keyframes -> {frames_dir}")
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
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.8, help="AKS t1")
    parser.add_argument("--std-threshold", type=float, default=-100.0, help="AKS t2")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
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
