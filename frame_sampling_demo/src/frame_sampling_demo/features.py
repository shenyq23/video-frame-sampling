from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .schemas import VideoContext


ProgressCallback = Callable[[float, str], None]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CLIPFeatureStore:
    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        device: str,
        batch_size: int,
        progress: ProgressCallback,
    ) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.device = choose_device(device)
        self.batch_size = batch_size
        self.progress = progress
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return self._model, self._processor
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "CLIP algorithms require the 'clip' extra: pip install -e '.[clip]'"
            ) from error
        self.progress(0.15, f"Loading CLIP model {self.model_name} on {self.device}")
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._model, self._processor

    def _cache_path(self, context: VideoContext) -> tuple[Path, dict]:
        video_path = Path(context.info.path)
        video_hash = file_sha256(video_path)
        model_hash = hashlib.sha256(self.model_name.encode("utf-8")).hexdigest()[:16]
        indices_bytes = np.asarray(context.candidate_indices, dtype=np.int64).tobytes()
        candidates_hash = hashlib.sha256(indices_bytes).hexdigest()[:16]
        directory = self.cache_dir / video_hash / "clip" / model_hash
        metadata = {
            "video": str(video_path),
            "video_sha256": video_hash,
            "model_name": self.model_name,
            "candidate_hash": candidates_hash,
            "candidate_count": len(context.candidate_indices),
        }
        return directory / f"{candidates_hash}.npz", metadata

    def image_embeddings(self, context: VideoContext) -> np.ndarray:
        cache_path, metadata = self._cache_path(context)
        if cache_path.is_file():
            cached = np.load(cache_path)
            cached_indices = cached["frame_indices"].astype(np.int64)
            expected = np.asarray(context.candidate_indices, dtype=np.int64)
            if np.array_equal(cached_indices, expected):
                self.progress(0.2, f"Using cached CLIP embeddings: {cache_path}")
                return cached["embeddings"].astype(np.float32)

        import torch
        from PIL import Image

        model, processor = self._load_model()
        embeddings = []
        total = len(context.candidate_indices)
        for start in range(0, total, self.batch_size):
            indices = context.candidate_indices[start : start + self.batch_size]
            arrays = context.reader.get_batch(indices).asnumpy()
            inputs = processor(
                images=[Image.fromarray(array) for array in arrays], return_tensors="pt"
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                values = model.get_image_features(**inputs)
                values = torch.nn.functional.normalize(values, dim=-1)
            embeddings.append(values.detach().cpu().float().numpy())
            ratio = (start + len(indices)) / max(total, 1)
            self.progress(0.2 + ratio * 0.45, f"Encoding candidate frames {start + len(indices)}/{total}")

        result = np.concatenate(embeddings, axis=0).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=result,
            frame_indices=np.asarray(context.candidate_indices, dtype=np.int64),
        )
        cache_path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def score_queries(
        self, context: VideoContext, queries: Sequence[str]
    ) -> dict[str, list[float]]:
        if not queries:
            return {}
        import torch

        image_embeddings = self.image_embeddings(context)
        model, processor = self._load_model()
        inputs = processor(
            text=list(queries), return_tensors="pt", padding=True, truncation=True
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            text_embeddings = model.get_text_features(**inputs)
            text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
        scores = text_embeddings.detach().cpu().float().numpy() @ image_embeddings.T
        return {
            query: scores[index].astype(float).tolist()
            for index, query in enumerate(queries)
        }
