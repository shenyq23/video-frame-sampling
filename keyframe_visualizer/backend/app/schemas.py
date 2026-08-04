from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class AKSParameters(BaseModel):
    aks_mode: Literal["original", "robust"] = "robust"
    max_num_frames: int = Field(default=32, ge=1, le=512)
    candidate_sampling: Literal["original", "interval"] = "interval"
    sample_interval: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    feature_backend: Literal["clip", "pangu", "mep"] = "clip"
    feature_profile: Optional[str] = None
    clip_model_id: Optional[str] = None
    model_name: str = "openai/clip-vit-base-patch32"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    batch_size: int = Field(default=16, ge=1, le=256)
    decode_threads: int = Field(default=2, ge=1, le=32)
    threshold: float = 0.8
    std_threshold: float = -100.0
    max_depth: int = Field(default=5, ge=0, le=16)
    jpeg_quality: int = Field(default=92, ge=1, le=100)
    save_uniform_baseline: bool = True
    save_candidate_frames: bool = True

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_name cannot be empty")
        return value.strip()

    @field_validator("clip_model_id")
    @classmethod
    def validate_clip_model_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value or not value.replace("-", "").isalnum():
            raise ValueError("clip_model_id is invalid")
        return value

    @model_validator(mode="after")
    def validate_remote_profile(self) -> "AKSParameters":
        if self.feature_backend in {"pangu", "mep"} and not self.feature_profile:
            raise ValueError("Pangu and MEP require a server-side feature profile")
        return self


class VSIParameters(BaseModel):
    subtitle_mode: Literal["ocr", "upload", "none"] = "ocr"
    ocr_fps: float = Field(default=2.0, gt=0, le=10, allow_inf_nan=False)
    ocr_crop_top: float = Field(default=0.62, ge=0, lt=1, allow_inf_nan=False)
    ocr_confidence: float = Field(default=0.30, ge=0, le=1, allow_inf_nan=False)
    text_model: str = "weights/sentence_transformer/paraphrase-multilingual-mpnet-base-v2"
    device: Literal["cuda", "mps", "cpu"] = "cpu"
    objects: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=512)
    detection_budget: int = Field(default=64, ge=1, le=10000)
    samples_per_round: int = Field(default=16, ge=1, le=10000)
    text_weight: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    model: str = "yolov8s-worldv2.pt"
    seed: int = 0
    save_uniform_baseline: bool = True
    save_candidate_frames: bool = True

    @field_validator("objects", mode="before")
    @classmethod
    def parse_objects(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, list):
            raise ValueError("objects must be a list or comma-separated string")
        result = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(result))

    @field_validator("text_model", "model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model name cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def validate_budget(self) -> "VSIParameters":
        if self.samples_per_round > self.detection_budget:
            raise ValueError("samples_per_round cannot exceed detection_budget")
        return self


class CreateJobConfig(BaseModel):
    algorithm: Literal["aks", "vsi"] = "aks"
    query: str = Field(min_length=1, max_length=8000)
    parameters: Union[AKSParameters, VSIParameters] = Field(default_factory=AKSParameters)

    @model_validator(mode="before")
    @classmethod
    def select_parameter_model(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw = data.get("parameters", {})
        data["parameters"] = (
            VSIParameters.model_validate(raw)
            if data.get("algorithm", "aks") == "vsi"
            else AKSParameters.model_validate(raw)
        )
        return data

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def validate_vsi_query(self) -> "CreateJobConfig":
        if self.algorithm == "vsi" and isinstance(self.parameters, VSIParameters):
            if not self.parameters.objects:
                raise ValueError("VSI query requires at least one object")
        return self


class CreateSessionConfig(BaseModel):
    algorithm: Literal["aks", "vsi"] = "aks"
    parameters: Union[AKSParameters, VSIParameters] = Field(default_factory=AKSParameters)

    @model_validator(mode="before")
    @classmethod
    def select_parameter_model(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw = data.get("parameters", {})
        data["parameters"] = (
            VSIParameters.model_validate(raw)
            if data.get("algorithm", "aks") == "vsi"
            else AKSParameters.model_validate(raw)
        )
        return data


class SessionRecord(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str
    progress: float
    algorithm: str
    original_filename: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    candidate_count: int = 0
    error: Optional[str] = None


class JobRecord(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str
    progress: float
    algorithm: str
    query: str
    original_filename: str
    session_id: Optional[str] = None
    owns_video: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    manifest_available: bool = False


class AlgorithmMetadata(BaseModel):
    id: str
    name: str
    description: str
    parameter_schema: dict[str, Any]


class VlmAnswerRequest(BaseModel):
    frame_set: Literal["selected", "uniform", "candidates"] = "selected"
    query: str = Field(min_length=1, max_length=8000)
    vlm_profile: str = Field(min_length=1, max_length=128)

    @field_validator("query", "vlm_profile")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()


class VlmUsedFrame(BaseModel):
    order: Optional[int] = None
    file: str
    timestamp_seconds: Optional[float] = None
    original_frame_index: Optional[int] = None
    candidate_order: Optional[int] = None


class VlmAnswerResult(BaseModel):
    schema_version: str
    job_id: str
    created_at: datetime
    profile_id: str
    profile_name: str
    frame_set: Literal["selected", "uniform", "candidates"]
    frame_set_name: str
    query: str
    answer: str
    source_frame_count: int
    used_frame_count: int
    frames_limited: bool
    used_frames: list[VlmUsedFrame]
