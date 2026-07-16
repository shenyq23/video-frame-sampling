from __future__ import annotations

import heapq

from ..registry import register_sampler
from ..schemas import Selection
from .base import FrameSampler


@register_sampler("clip_threshold")
class AdaptiveThresholdSampler(FrameSampler):
    query_aware = True
    adaptive_count = True

    def select(self, context, request, query, scores):
        if query is None or scores is None:
            raise ValueError("clip_threshold requires one query and its CLIP scores")

        positions = [
            position for position, score in enumerate(scores) if score >= request.score_threshold
        ]
        if len(positions) < request.min_frames:
            positions = heapq.nlargest(
                min(request.min_frames, len(scores)), range(len(scores)), scores.__getitem__
            )
        if request.max_frames is not None and len(positions) > request.max_frames:
            selected_set = set(positions)
            positions = heapq.nlargest(
                request.max_frames,
                selected_set,
                scores.__getitem__,
            )

        indices = sorted(context.candidate_indices[position] for position in positions)
        score_map = {
            context.candidate_indices[position]: float(scores[position]) for position in positions
        }
        return Selection(
            algorithm=self.name,
            selected_indices=indices,
            scores=score_map,
            selected_by={index: [query] for index in indices},
            trace={
                "query": query,
                "score_threshold": request.score_threshold,
                "min_frames": request.min_frames,
                "max_frames": request.max_frames,
            },
        )
