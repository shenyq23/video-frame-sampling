from __future__ import annotations

from dataclasses import asdict

from ..registry import register_sampler
from ..schemas import Selection
from .base import FrameSampler


class _AKSSampler(FrameSampler):
    query_aware = True
    adaptive_count = False
    mode: str

    def select(self, context, request, query, scores):
        if query is None or scores is None:
            raise ValueError(f"{self.name} requires one query and its CLIP scores")
        try:
            from aks_core import select_frames
        except ImportError as error:
            raise RuntimeError(
                "AKS core is not installed. Run: pip install -e ../AKS"
            ) from error

        selection = select_frames(
            scores,
            context.candidate_indices,
            max_num_frames=request.max_frames or 32,
            threshold=request.threshold,
            std_threshold=request.std_threshold,
            max_depth=request.max_depth,
            mode=self.mode,
        )
        position_by_index = {
            frame_index: position
            for position, frame_index in enumerate(context.candidate_indices)
        }
        score_map = {
            frame_index: float(scores[position_by_index[frame_index]])
            for frame_index in selection.frame_indices
        }
        return Selection(
            algorithm=self.name,
            selected_indices=selection.frame_indices,
            scores=score_map,
            selected_by={index: [query] for index in selection.frame_indices},
            trace={
                "query": query,
                "mode": selection.mode,
                "requested_frames": request.max_frames or 32,
                "threshold_t1": request.threshold,
                "std_threshold_t2": request.std_threshold,
                "max_depth": request.max_depth,
                "segments": [asdict(segment) for segment in selection.segments],
            },
        )


@register_sampler("aks_original")
class AKSOriginalSampler(_AKSSampler):
    mode = "original"


@register_sampler("aks_robust")
class AKSRobustSampler(_AKSSampler):
    mode = "robust"
