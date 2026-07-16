from __future__ import annotations

import heapq

from ..registry import register_sampler
from ..schemas import Selection
from .base import FrameSampler


@register_sampler("clip_topk")
class CLIPTopKSampler(FrameSampler):
    query_aware = True
    adaptive_count = False

    def select(self, context, request, query, scores):
        if query is None or scores is None:
            raise ValueError("clip_topk requires one query and its CLIP scores")
        count = min(request.max_frames or 32, len(scores))
        positions = heapq.nlargest(count, range(len(scores)), scores.__getitem__)
        indices = sorted(context.candidate_indices[position] for position in positions)
        score_map = {
            context.candidate_indices[position]: float(scores[position]) for position in positions
        }
        return Selection(
            algorithm=self.name,
            selected_indices=indices,
            scores=score_map,
            selected_by={index: [query] for index in indices},
            trace={"query": query, "requested_frames": request.max_frames or 32},
        )
