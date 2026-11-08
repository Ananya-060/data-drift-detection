from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DriftComparisonRequest(BaseModel):
    reference: dict[str, list[float]] = Field(min_length=1)
    current: dict[str, list[float]] = Field(min_length=1)


class DatasetDriftRequest(BaseModel):
    cohort: Literal[
        "2014-Q1",
        "2014-Q2",
        "2014-Q3",
        "2014-Q4",
        "2015-Q1",
        "2015-Q2",
        "2015-Q3",
        "2015-Q4",
    ]


class DatasetStatusResponse(BaseModel):
    available: bool
    file_name: str
    size_bytes: int
    missing_columns: list[str]
    train_end_year: int
    evaluation_cohorts: list[str]


class ModelStatusResponse(BaseModel):
    trained: bool
    metadata: dict | None


class DriftRunResponse(BaseModel):
    id: int
    created_at: datetime
    status: str
    source: str
    feature_count: int
    drifted_feature_count: int
    results: dict[str, dict[str, float | str | bool]]
    notification_delivery: dict[str, str] = Field(default_factory=dict)

    model_config = {"from_attributes": True}
