from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .schemas import FrameRecord, Selection, VideoContext


def export_selection(
    context: VideoContext,
    selection: Selection,
    output_dir: Path,
    group_name: str,
    query_scores: dict[str, list[float]],
    jpeg_quality: int,
) -> tuple[list[FrameRecord], Path]:
    from PIL import Image

    group_dir = output_dir / group_name
    frames_dir = group_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    indices = selection.selected_indices
    arrays = context.reader.get_batch(indices).asnumpy() if indices else []
    candidate_position = {
        frame_index: position
        for position, frame_index in enumerate(context.candidate_indices)
    }
    records = []
    for order, (frame_index, array) in enumerate(zip(indices, arrays), 1):
        timestamp = frame_index / context.info.fps
        filename = f"{order:03d}_t{timestamp:010.3f}_f{frame_index}.jpg"
        Image.fromarray(array).save(frames_dir / filename, quality=jpeg_quality)
        per_query = {}
        position = candidate_position.get(frame_index)
        if position is not None:
            per_query = {
                query: float(scores[position]) for query, scores in query_scores.items()
            }
        records.append(
            FrameRecord(
                order=order,
                file=f"{group_name}/frames/{filename}",
                frame_index=int(frame_index),
                timestamp_seconds=round(timestamp, 6),
                score=selection.scores.get(frame_index),
                query_scores=per_query,
                selected_by=selection.selected_by.get(frame_index, []),
            )
        )

    group_manifest = {
        "group": group_name,
        "algorithm": selection.algorithm,
        "selected_frame_count": len(records),
        "trace": selection.trace,
        "frames": [asdict(record) for record in records],
    }
    manifest_path = group_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(group_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records, manifest_path
