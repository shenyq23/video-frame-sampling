from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AKSParameters(BaseModel):
    aks_mode: Literal["original", "robust"] = "robust"
    max_num_frames: int = Field(default=32, ge=1, le=512)
    candidate_sampling: Literal["original", "interval"] = "interval"
    sample_interval: float = Field(default=1.0, gt=0, le=60)
    feature_backend: Literal["clip", "pangu", "mep"] = "clip"
    feature_profile: Optional[str] = None
    model_name: str = "openai/clip-vit-base-patch32"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    batch_size: int = Field(default=16, ge=1, le=256)
    decode_threads: int = Field(default=2, ge=1, le=32)
    threshold: float = 0.8
    std_threshold: float = -100.0
    max_depth: int = Field(default=5, ge=0, le=16)
    jpeg_quality: int = Field(default=92, ge=1, le=100)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_name cannot be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_remote_profile(self) -> "AKSParameters":
        if self.feature_backend in {"pangu", "mep"} and not self.feature_profile:
            raise ValueError("Pangu and MEP require a server-side feature profile")
        return self


class CreateJobConfig(BaseModel):
    algorithm: Literal["aks"] = "aks"
    query: str = Field(min_length=1, max_length=8000)
    parameters: AKSParameters = Field(default_factory=AKSParameters)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


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
    error: Optional[str] = None
    manifest_available: bool = False


class AlgorithmMetadata(BaseModel):
    id: str
    name: str
    description: str
    parameter_schema: dict[str, Any]
