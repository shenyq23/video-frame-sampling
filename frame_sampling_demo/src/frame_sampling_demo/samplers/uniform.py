from __future__ import annotations

from ..registry import register_sampler
from ..schemas import SamplingRequest, Selection, VideoContext
from ..video import uniform_frame_indices
from .base import FrameSampler


@register_sampler("uniform")
class UniformSampler(FrameSampler):
    query_aware = False
    adaptive_count = False

    def select(self, context, request, query, scores):
        count = request.max_frames or 32
        indices = uniform_frame_indices(context.info.total_frames, count)
        return Selection(
            algorithm=self.name,
            selected_indices=indices,
            trace={"requested_frames": count, "sampling": "full_video_uniform"},
        )
