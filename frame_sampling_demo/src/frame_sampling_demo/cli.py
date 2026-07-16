from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import samplers as _samplers  # noqa: F401
from .pipeline import run_sampling
from .registry import available_samplers
from .schemas import SamplingRequest


def _read_queries(values: list[str], query_file: str | None) -> list[str]:
    queries = [value.strip() for value in values if value.strip()]
    if query_file:
        path = Path(query_file).expanduser()
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            loaded = json.loads(text)
            if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
                raise ValueError("query JSON must be an array of strings")
            queries.extend(item.strip() for item in loaded if item.strip())
        else:
            queries.extend(line.strip() for line in text.splitlines() if line.strip())
    return list(dict.fromkeys(queries))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract video frames with pluggable fixed or adaptive samplers."
    )
    parser.add_argument("--list-algorithms", action="store_true")
    parser.add_argument("--video")
    parser.add_argument("--algorithm", choices=available_samplers())
    parser.add_argument("--query", action="append", default=[], help="Repeat for multiple queries")
    parser.add_argument("--query-file", help="Text file (one query per line) or JSON string array")
    parser.add_argument(
        "--multi-query-mode",
        choices=("independent", "union", "joint"),
        default="independent",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--min-frames", type=int, default=0)
    parser.add_argument(
        "--candidate-sampling", choices=("interval", "original"), default="interval"
    )
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.8, help="AKS t1")
    parser.add_argument(
        "--score-threshold", type=float, default=0.35, help="clip_threshold cutoff"
    )
    parser.add_argument("--std-threshold", type=float, default=-100.0, help="AKS t2")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_algorithms:
        print("\n".join(available_samplers()))
        return
    if not args.video or not args.algorithm:
        parser.error("--video and --algorithm are required unless --list-algorithms is used")

    queries = _read_queries(args.query, args.query_file)
    request = SamplingRequest(
        video_path=Path(args.video).expanduser().resolve(),
        queries=queries,
        algorithm=args.algorithm,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        multi_query_mode=args.multi_query_mode,
        candidate_sampling=args.candidate_sampling,
        sample_interval=args.sample_interval,
        threshold=args.threshold,
        score_threshold=args.score_threshold,
        std_threshold=args.std_threshold,
        max_depth=args.max_depth,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        decode_threads=args.decode_threads,
        jpeg_quality=args.jpeg_quality,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    result = run_sampling(request)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
