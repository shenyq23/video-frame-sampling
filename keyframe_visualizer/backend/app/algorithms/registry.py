from __future__ import annotations

from .aks_adapter import AKSAdapter
from .base import AlgorithmAdapter


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, AlgorithmAdapter] = {"aks": AKSAdapter()}

    def get(self, algorithm_id: str) -> AlgorithmAdapter:
        try:
            return self._adapters[algorithm_id]
        except KeyError as error:
            raise ValueError(f"Unknown algorithm: {algorithm_id}") from error

    def metadata(self) -> list[dict]:
        return [adapter.metadata() for adapter in self._adapters.values()]

