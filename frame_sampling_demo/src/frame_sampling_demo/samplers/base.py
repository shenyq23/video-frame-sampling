from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import SamplingRequest, Selection, VideoContext


class FrameSampler(ABC):
    name: str
    query_aware: bool = False
    adaptive_count: bool = False

    @abstractmethod
    def select(
        self,
        context: VideoContext,
        request: SamplingRequest,
        query: str | None,
        scores: list[float] | None,
    ) -> Selection:
        raise NotImplementedError
