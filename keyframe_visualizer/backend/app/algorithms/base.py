from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[str, float], None]


class AlgorithmAdapter(ABC):
    id: str

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        original_filename: str,
        query: str,
        parameters: dict[str, Any],
        output_dir: Path,
        progress: ProgressCallback,
    ) -> Path:
        raise NotImplementedError

