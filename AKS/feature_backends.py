"""Pluggable text/image feature backends for AKS relevance scoring.

AKS only consumes one relevance score per candidate frame.  This module keeps
model loading, remote API protocols, and cosine similarity outside the AKS
selection algorithm so new feature services can be added without changing the
sampling pipeline.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import importlib
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


DEFAULT_QUERY_INSTRUCTION = "Retrieve relevant images for the user's query."
DEFAULT_IMAGE_INSTRUCTION = "Represent this image for text-to-image retrieval."
PANGU_QUERY_INSTRUCTION = "Retrieve relevant documents for the user's query."


class FeatureBackendError(RuntimeError):
    """Raised when a feature backend cannot produce trustworthy embeddings."""


class EmbeddingBackend(ABC):
    """Common contract for models that embed text and images into one space."""

    name = "embedding"

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return a two-dimensional array with one row per input text."""

    @abstractmethod
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return a two-dimensional array with one row per input image."""

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": self.name}


def _as_embedding_matrix(value: Any, expected_rows: int, source: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if expected_rows == 1 and matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise FeatureBackendError(f"{source} must return a 2-D embedding matrix")
    if matrix.shape[0] != expected_rows:
        raise FeatureBackendError(
            f"{source} returned {matrix.shape[0]} rows for {expected_rows} inputs"
        )
    if matrix.shape[1] == 0:
        raise FeatureBackendError(f"{source} returned an empty embedding")
    if not np.all(np.isfinite(matrix)):
        raise FeatureBackendError(f"{source} returned NaN or infinite values")
    return matrix


def _read_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, Mapping):
            current = current[part]
        else:
            raise KeyError(path)
    return current


def _first_path(data: Any, paths: Sequence[str], source: str) -> Any:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            pass
    for path in paths:
        try:
            value = _read_path(data, path)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if value is not None and value != "":
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return value
    if isinstance(data, Mapping):
        fields = ", ".join(str(key) for key in data.keys())
        diagnostics = []
        for key in ("code", "msg", "message", "error"):
            value = data.get(key)
            if isinstance(value, (str, int, float, bool)):
                rendered = str(value)
                diagnostics.append(f"{key}={rendered[:500]!r}")
        shape = f"available top-level fields: {fields}"
        if diagnostics:
            shape += "; " + ", ".join(diagnostics)
    else:
        shape = f"response type: {type(data).__name__}"
    raise FeatureBackendError(
        f"{source} response contains none of the configured embedding paths: "
        f"{', '.join(paths)}; {shape}"
    )


def _secret(config: Mapping[str, Any], value_key: str, env_key: str) -> str:
    value = config.get(value_key)
    env_name = config.get(env_key)
    if value:
        return str(value)
    if env_name and os.getenv(str(env_name)):
        return os.environ[str(env_name)]
    raise FeatureBackendError(
        f"Missing {value_key}; set it directly or configure {env_key}"
    )


def _image_bytes(image: Image.Image, image_format: str, quality: int) -> tuple[bytes, str, str]:
    normalized_format = image_format.upper()
    if normalized_format == "JPG":
        normalized_format = "JPEG"
    output = io.BytesIO()
    converted = image.convert("RGB") if normalized_format == "JPEG" else image
    save_kwargs = {"quality": quality} if normalized_format == "JPEG" else {}
    converted.save(output, format=normalized_format, **save_kwargs)
    mime = "image/jpeg" if normalized_format == "JPEG" else f"image/{normalized_format.lower()}"
    extension = "jpg" if normalized_format == "JPEG" else normalized_format.lower()
    return output.getvalue(), mime, extension


class LocalClipBackend(EmbeddingBackend):
    """Original Hugging Face CLIP implementation used by the AKS demo."""

    name = "clip"

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name
        self.device = device
        self._torch = torch
        print(f"Loading relevance model: {model_name} ({device})")
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return _as_embedding_matrix(
            features.detach().cpu().float().numpy(), len(texts), "CLIP text encoder"
        )

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt", padding=True)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return _as_embedding_matrix(
            features.detach().cpu().float().numpy(), len(images), "CLIP image encoder"
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model_name, "device": self.device}


class MultipartEmbeddingBackend(EmbeddingBackend):
    """Configurable multipart HTTP embedding backend.

    It supports services that accept one text or image per request.  Pangu is a
    preset of this backend; future APIs using the same broad protocol can use
    ``http`` and override endpoint/field/response names in a JSON config file.
    """

    name = "http"

    def __init__(self, config: Mapping[str, Any], session: Any = None):
        import requests

        self.config = dict(config)
        self.base_url = str(self.config["base_url"]).rstrip("/")
        self.text_endpoint = str(self.config.get("text_endpoint", "/embed"))
        self.image_endpoint = str(self.config.get("image_endpoint", "/embed"))
        self.text_field = str(self.config.get("text_field", "text"))
        self.image_field = str(self.config.get("image_field", "image"))
        self.instruction_field = self.config.get("instruction_field", "instruction")
        self.text_instruction = str(
            self.config.get("text_instruction", DEFAULT_QUERY_INSTRUCTION)
        )
        self.image_instruction = str(
            self.config.get("image_instruction", DEFAULT_IMAGE_INSTRUCTION)
        )
        response_paths = self.config.get("response_embedding_paths", ["embedding"])
        self.response_paths = [str(path) for path in response_paths]
        self.timeout = float(self.config.get("timeout_seconds", 30))
        self.max_retries = int(self.config.get("max_retries", 2))
        self.image_format = str(self.config.get("image_format", "JPEG"))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 95))
        self.session = session or requests.Session()
        self.headers = {str(k): str(v) for k, v in self.config.get("headers", {}).items()}
        if self.config.get("api_key") or self.config.get("api_key_env"):
            token = _secret(self.config, "api_key", "api_key_env")
            header = str(self.config.get("auth_header", "Authorization"))
            scheme = str(self.config.get("auth_scheme", "Bearer"))
            self.headers[header] = f"{scheme} {token}".strip()

    def _post(self, endpoint: str, request_factory: Any) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data, files = request_factory()
                response = self.session.post(
                    url,
                    headers=self.headers,
                    data=data,
                    files=files,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 2.0))
        raise FeatureBackendError(f"HTTP embedding request failed: {last_error}") from last_error

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            def request_factory(text: str = text) -> tuple[dict[str, str], dict[str, Any]]:
                data = {self.text_field: text}
                if self.instruction_field:
                    data[str(self.instruction_field)] = self.text_instruction
                return data, {}

            result = self._post(self.text_endpoint, request_factory)
            embeddings.append(_first_path(result, self.response_paths, "text embedding"))
        return _as_embedding_matrix(embeddings, len(texts), "HTTP text encoder")

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        embeddings = []
        for image in images:
            content, mime, extension = _image_bytes(
                image, self.image_format, self.jpeg_quality
            )

            def request_factory(
                content: bytes = content, mime: str = mime, extension: str = extension
            ) -> tuple[dict[str, str], dict[str, Any]]:
                data: dict[str, str] = {}
                files: dict[str, Any] = {
                    self.image_field: (f"frame.{extension}", io.BytesIO(content), mime)
                }
                if self.instruction_field:
                    data[str(self.instruction_field)] = self.image_instruction
                return data, files

            result = self._post(self.image_endpoint, request_factory)
            embeddings.append(_first_path(result, self.response_paths, "image embedding"))
        return _as_embedding_matrix(embeddings, len(images), "HTTP image encoder")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "base_url": self.base_url,
            "text_instruction": self.text_instruction,
            "image_instruction": self.image_instruction,
        }


class PanguEmbeddingBackend(MultipartEmbeddingBackend):
    name = "pangu"

    def __init__(self, config: Mapping[str, Any], session: Any = None):
        pangu_config = dict(config)
        pangu_config.setdefault("text_instruction", PANGU_QUERY_INSTRUCTION)
        pangu_config.setdefault("image_instruction", PANGU_QUERY_INSTRUCTION)
        pangu_config.setdefault("image_format", "PNG")
        super().__init__(pangu_config, session=session)


class MepEmbeddingBackend(EmbeddingBackend):
    """Configurable synchronous MEP text/image embedding backend."""

    name = "mep"

    def __init__(self, config: Mapping[str, Any], session: Any = None):
        import requests

        self.config = dict(config)
        self.elb = str(self.config["elb"])
        self.appid = _secret(self.config, "appid", "appid_env")
        self.secret_key = _secret(self.config, "secret_key", "secret_key_env")
        self.b_id = str(self.config["b_id"])
        self.flow_id = str(self.config["flow_id"])
        self.model_version = self.config.get("model_version")
        self.text_task = str(self.config.get("text_task", "text_embedding"))
        self.image_task = str(self.config.get("image_task", "image_embedding"))
        self.text_field = str(self.config.get("text_field", "text"))
        self.image_field = str(self.config.get("image_field", "image_url"))
        self.image_encoding = str(self.config.get("image_encoding", "base64"))
        shared_paths = self.config.get("response_embedding_paths")
        text_paths = self.config.get(
            "text_response_embedding_paths",
            shared_paths or ["text_embedding"],
        )
        image_paths = self.config.get(
            "image_response_embedding_paths",
            shared_paths or ["image_embedding"],
        )
        self.text_response_paths = [str(path) for path in text_paths]
        self.image_response_paths = [str(path) for path in image_paths]
        self.timeout = float(self.config.get("timeout_seconds", 30))
        self.max_retries = int(self.config.get("max_retries", 2))
        self.image_format = str(self.config.get("image_format", "JPEG"))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 95))
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        timestamp = str(round(time.time() * 1000))
        string_to_sign = (
            f"POST&/service&&&appid={self.appid}&timestamp={timestamp}"
        ).encode("utf-8")
        signature = base64.b64encode(
            hmac.new(self.secret_key.encode("utf-8"), string_to_sign, hashlib.sha256).digest()
        ).decode("utf-8")
        token = (
            f"CLOUDSOA-HMAC-SHA256 appid={self.appid}, timestamp={timestamp}, "
            f"signmode=easy, signature=\"{signature}\""
        )
        return {"Content-Type": "application/json", "Authorization": token}

    def _request(self, data: dict[str, Any]) -> Any:
        if self.model_version is not None:
            data["model_version"] = self.model_version
        payload = {
            "version": "1.0",
            "data": data,
            "meta": {
                "bId": self.b_id,
                "flowId": self.flow_id,
                "isPressureTest": False,
                "uuId": str(uuid.uuid4()),
            },
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.elb, headers=self._headers(), json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                body = response.json()

                # 处理网关包装的响应
                # 格式1: {"result": {"code": 200, "content": [...]}}
                # 格式2: {"result": {"code": 200, "text_embedding": [...]}}
                # 格式3: 直接返回 {"code": 200, "text_embedding": [...]}

                if "result" in body:
                    result = body["result"]

                    # 如果result是字符串，尝试解析
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except json.JSONDecodeError:
                            pass

                    # 如果result是dict
                    if isinstance(result, dict):
                        # 检查是否有content字段
                        if "content" in result:
                            content = result["content"]
                            # content可能是列表
                            if isinstance(content, list) and len(content) > 0:
                                content = content[0]
                            # 如果content是字符串，尝试解析JSON
                            if isinstance(content, str):
                                try:
                                    content = json.loads(content)
                                except json.JSONDecodeError:
                                    pass
                            # 如果content是dict，返回它（包含嵌入向量）
                            if isinstance(content, dict):
                                return content
                        # 如果result本身包含嵌入字段，返回result
                        if any(path in result for path in self.text_response_paths + self.image_response_paths):
                            return result
                        # 否则返回result（让调用方处理）
                        return result

                # 如果没有result字段，直接返回body
                return body
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2 ** attempt), 2.0))
        raise FeatureBackendError(f"MEP embedding request failed: {last_error}") from last_error

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        values = []
        for text in texts:
            result = self._request({"task": self.text_task, self.text_field: text})

            # 检查错误码
            code = result.get("code")
            if code is not None and str(code) != "200":
                msg = result.get("msg", "Unknown error")
                raise FeatureBackendError(f"MEP error (code={code}): {msg}")

            values.append(
                _first_path(result, self.text_response_paths, "MEP text embedding")
            )
        return _as_embedding_matrix(values, len(texts), "MEP text encoder")

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        values = []
        for image in images:
            content, mime, _ = _image_bytes(image, self.image_format, self.jpeg_quality)
            encoded = base64.b64encode(content).decode("ascii")
            if self.image_encoding == "data_url":
                encoded = f"data:{mime};base64,{encoded}"
            elif self.image_encoding != "base64":
                raise FeatureBackendError(
                    "MEP image_encoding must be 'base64' or 'data_url'"
                )
            result = self._request({"task": self.image_task, self.image_field: encoded})

            # 检查错误码
            code = result.get("code")
            if code is not None and str(code) != "200":
                msg = result.get("msg", "Unknown error")
                raise FeatureBackendError(f"MEP error (code={code}): {msg}")

            values.append(
                _first_path(result, self.image_response_paths, "MEP image embedding")
            )
        return _as_embedding_matrix(values, len(images), "MEP image encoder")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "b_id": self.b_id,
            "flow_id": self.flow_id,
            "model_version": self.model_version,
            "text_task": self.text_task,
            "image_task": self.image_task,
        }


class EmbeddingRelevanceScorer:
    """Turn any shared-space text/image embedding backend into AKS scores."""

    def __init__(self, backend: EmbeddingBackend, batch_size: int = 16):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.backend = backend
        self.batch_size = batch_size
        self.embedding_dimension: int | None = None
        self._query_embedding: np.ndarray | None = None

    def prepare_query(self, query: str) -> None:
        """Embed and cache a query once before processing bounded image batches."""
        text = _as_embedding_matrix(
            self.backend.embed_texts([query]), 1, f"{self.backend.name} text encoder"
        )
        text_norm = np.linalg.norm(text, axis=1, keepdims=True)
        if np.any(text_norm == 0):
            raise FeatureBackendError("Text encoder returned a zero-norm embedding")
        self._query_embedding = text / text_norm
        self.embedding_dimension = int(text.shape[1])

    def score_images(self, images: Sequence[Image.Image]) -> list[float]:
        """Score one image batch against the query cached by ``prepare_query``."""
        if self._query_embedding is None:
            raise FeatureBackendError("prepare_query must be called before score_images")
        text = self._query_embedding
        scores: list[float] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            image_features = _as_embedding_matrix(
                self.backend.embed_images(batch),
                len(batch),
                f"{self.backend.name} image encoder",
            )
            if image_features.shape[1] != text.shape[1]:
                raise FeatureBackendError(
                    "Text/image embedding dimensions differ: "
                    f"{text.shape[1]} != {image_features.shape[1]}"
                )
            norms = np.linalg.norm(image_features, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise FeatureBackendError("Image encoder returned a zero-norm embedding")
            scores.extend(((image_features / norms) @ text.T).reshape(-1).tolist())
        return [float(value) for value in scores]

    def score(self, query: str, images: Sequence[Image.Image]) -> list[float]:
        self.prepare_query(query)
        return self.score_images(images)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self.backend.metadata)
        metadata.update(
            {
                "scoring": "cosine_similarity",
                "embedding_dimension": self.embedding_dimension,
            }
        )
        return metadata


def load_feature_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureBackendError(f"Cannot load feature config {config_path}: {error}") from error
    if not isinstance(value, dict):
        raise FeatureBackendError("Feature config must contain a JSON object")
    return value


def create_relevance_scorer(
    backend_name: str,
    config: Mapping[str, Any],
    *,
    model_name: str,
    device: str,
    batch_size: int,
) -> Any:
    if backend_name == "clip":
        backend: EmbeddingBackend = LocalClipBackend(
            str(config.get("model_name", model_name)), str(config.get("device", device))
        )
    elif backend_name == "pangu":
        merged = {
            "base_url": os.getenv("PANGU_EMBED_BASE_URL", ""),
            "api_key_env": "PANGU_EMBED_API_KEY",
            **config,
        }
        if not merged["base_url"]:
            raise FeatureBackendError(
                "Pangu base_url is required in --feature-config or PANGU_EMBED_BASE_URL"
            )
        backend = PanguEmbeddingBackend(merged)
    elif backend_name == "mep":
        backend = MepEmbeddingBackend(config)
    elif backend_name == "http":
        backend = MultipartEmbeddingBackend(config)
    elif backend_name == "python":
        class_path = str(config.get("class_path", ""))
        if ":" not in class_path:
            raise FeatureBackendError(
                "Python backend class_path must use 'module:ClassName'"
            )
        module_name, class_name = class_path.split(":", 1)
        try:
            component_class = getattr(importlib.import_module(module_name), class_name)
            component = component_class(config.get("options", {}))
        except Exception as error:
            raise FeatureBackendError(
                f"Cannot initialize Python feature backend {class_path}: {error}"
            ) from error
        if all(hasattr(component, name) for name in ("prepare_query", "score_images")):
            if not hasattr(component, "metadata"):
                raise FeatureBackendError("Custom scorer must expose metadata")
            return component
        if not all(hasattr(component, name) for name in ("embed_texts", "embed_images")):
            raise FeatureBackendError(
                "Custom backend must implement embed_texts/embed_images or "
                "prepare_query/score_images"
            )
        if not hasattr(component, "name"):
            component.name = class_name
        if not hasattr(component, "metadata"):
            component.metadata = {"backend": component.name, "class_path": class_path}
        backend = component
    else:
        raise FeatureBackendError(f"Unknown feature backend: {backend_name}")
    return EmbeddingRelevanceScorer(backend, batch_size=batch_size)