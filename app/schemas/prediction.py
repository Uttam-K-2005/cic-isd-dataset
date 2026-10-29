"""Prediction API schemas."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel


class PredictionRecord(RootModel[dict[str, Any]]):
    """One raw JSON feature object; exact keys are validated against the artifact."""


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PredictionRecord] = Field(min_length=1, max_length=256)


class PredictionResponse(BaseModel):
    request_id: UUID
    predicted_class: str
    is_attack: bool
    probabilities: dict[str, float]
    model_name: str
    model_version: str | None
    predicted_at: str


class BatchPredictionResponse(BaseModel):
    request_id: UUID
    predictions: list[PredictionResponse]
    model_name: str
    model_version: str | None
    predicted_at: str


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelMetadataResponse(BaseModel):
    model_name: str
    model_version: str | None
    supported_classes: list[str]

