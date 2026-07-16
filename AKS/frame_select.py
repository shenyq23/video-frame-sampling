"""Select AKS frames from repository score/frame JSON files.

This remains compatible with the original command-line workflow while routing
selection through :mod:`aks_core`, which is also used by the custom-video V2
entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aks_core import select_frame_indices


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select video frames with AKS")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="longvideobench",
        help="Output dataset directory name",
    )
    parser.add_argument(
        "--extract_feature_model",
        type=str,
        default="blip",
        help="Feature scorer name used in the output path (blip/clip/sevila)",
    )
    parser.add_argument(
        "--score_path",
        type=str,
        default="./outscores/longvideobench/blip/scores.json",
    )
    parser.add_argument(
        "--frame_path",
        type=str,
        default="./outscores/longvideobench/blip/frames.json",
    )
    parser.add_argument("--max_num_frames", type=int, default=64)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--t1", type=float, default=0.8)
    parser.add_argument("--t2", type=float, default=-100.0)
    parser.add_argument("--all_depth", type=int, default=5)
    parser.add_argument(
        "--aks_mode",
        choices=("original", "robust"),
        default="original",
        help="original reproduces repository quotas; robust fills the requested budget",
    )
    parser.add_argument("--output_file", type=str, default="./selected_frames")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if args.ratio <= 0:
        raise ValueError("ratio must be positive")

    with open(args.score_path, encoding="utf-8") as file:
        score_rows = json.load(file)
    with open(args.frame_path, encoding="utf-8") as file:
        frame_rows = json.load(file)
    if len(score_rows) != len(frame_rows):
        raise ValueError("score_path and frame_path must contain the same number of rows")

    outputs: list[list[int]] = []
    for row_number, (scores, frame_indices) in enumerate(zip(score_rows, frame_rows)):
        if len(scores) != len(frame_indices):
            raise ValueError(
                f"row {row_number}: scores and frame indices have different lengths"
            )
        sampled_scores = scores[:: args.ratio]
        sampled_indices = frame_indices[:: args.ratio]
        outputs.append(
            select_frame_indices(
                sampled_scores,
                sampled_indices,
                max_num_frames=args.max_num_frames,
                threshold=args.t1,
                std_threshold=args.t2,
                max_depth=args.all_depth,
                mode=args.aks_mode,
            )
        )

    output_dir = Path(args.output_file) / args.dataset_name / args.extract_feature_model
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_frames.json"
    output_path.write_text(json.dumps(outputs), encoding="utf-8")
    manifest = {
        "algorithm": "AKS",
        "implementation": {
            "entry_point": "frame_select.py",
            "shared_core": "aks_core.py",
            "mode": args.aks_mode,
        },
        "inputs": {
            "scores": {
                "path": str(Path(args.score_path).expanduser().resolve()),
                "sha256": file_sha256(args.score_path),
            },
            "frames": {
                "path": str(Path(args.frame_path).expanduser().resolve()),
                "sha256": file_sha256(args.frame_path),
            },
            "rows": len(score_rows),
        },
        "parameters": {
            "max_num_frames": args.max_num_frames,
            "ratio": args.ratio,
            "threshold_t1": args.t1,
            "std_threshold_t2": args.t2,
            "max_depth": args.all_depth,
        },
        "output": {
            "path": str(output_path.resolve()),
            "selected_counts": [len(row) for row in outputs],
        },
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def main() -> None:
    output_path = run(parse_arguments())
    print(f"Selected-frame indices -> {output_path}")


if __name__ == "__main__":
    main()
