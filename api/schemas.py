import enum
from typing import List, Optional
from pydantic import BaseModel, Field, StringConstraints
from typing_extensions import Annotated

# Custom constrained string types for payload validation
TextPayload = Annotated[
    str, 
    StringConstraints(strip_whitespace=True, min_length=20, max_length=5000)
]
BatchTextsPayload = Annotated[
    List[TextPayload], 
    Field(min_length=1, max_length=100)
]

class ReportReasonEnum(str, enum.Enum):
    WRONG_LANGUAGE = "wrong_language"
    MIXED_LANGUAGES = "mixed_languages"
    NOT_A_LANGUAGE = "not_a_language"
    OTHER = "other"

class ReportStatusEnum(str, enum.Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


# -----------------------------------------------------------------------------
# 1. Prediction Schemas
# -----------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: TextPayload


class Candidate(BaseModel):
    code: str = Field(..., description="ISO 639-3 language code", example="eng")
    name: str = Field(..., description="Full language name", example="English")
    confidence: float = Field(..., description="Confidence probability score", example=0.95)


class PredictResponse(BaseModel):
    prediction_id: str = Field(..., description="Unique UUID for prediction tracing", example="3f6c2a9e-8b1a-4c2d-9e1a-2b3c4d5e6f7a")
    model_version: str = Field(..., description="Active model version used for inference", example="v1")
    language_code: str = Field(..., description="Primary predicted ISO language code", example="eng")
    language_name: str = Field(..., description="Primary predicted language name", example="English")
    confidence: float = Field(..., description="Top prediction confidence score", example=0.95)
    is_unsure: bool = Field(..., description="Flag indicating low confidence below threshold", example=False)
    explanation: List[str] = Field(..., description="Heuristic explanation messages")
    top_candidates: List[Candidate] = Field(..., description="Top ranked language candidate predictions")


class BatchPredictRequest(BaseModel):
    texts: BatchTextsPayload


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


# -----------------------------------------------------------------------------
# 2. Metadata & Auxiliary Schemas
# -----------------------------------------------------------------------------

class LanguageItem(BaseModel):
    code: str = Field(..., example="eng")
    name: str = Field(..., example="English")


class ModelVersion(BaseModel):
    version: str = Field(..., example="v1")
    algorithm: str = Field(..., example="Multinomial Naive Bayes")
    status: str = Field(..., example="active", description="Status: active, candidate, archived")
    created_at: Optional[str] = Field(None, example="2026-11-01T10:00:00Z")
    accuracy: float = Field(..., example=0.944)
    macro_f1: float = Field(..., example=0.949)


class PromoteRequest(BaseModel):
    version: str = Field(..., example="v2")


# -----------------------------------------------------------------------------
# 3. Reporting Schemas
# -----------------------------------------------------------------------------

class ReportRequest(BaseModel):
    prediction_id: str = Field(..., example="3f6c2a9e-8b1a-4c2d-9e1a-2b3c4d5e6f7a")
    reason: ReportReasonEnum = Field(..., example=ReportReasonEnum.WRONG_LANGUAGE)
    suggested_language: Optional[str] = Field(None, example="eng")
    comment: Optional[str] = Field(None, max_length=500, example="This text is clearly English.")


class ReportUpdate(BaseModel):
    status: ReportStatusEnum = Field(..., example=ReportStatusEnum.ACCEPTED)
    verified_language: Optional[str] = Field(None, example="eng")


class ReportItem(BaseModel):
    report_id: int = Field(..., example=1)
    prediction_id: str = Field(..., example="3f6c2a9e-8b1a-4c2d-9e1a-2b3c4d5e6f7a")
    created_at: str = Field(..., example="2026-11-11 12:00:00")
    reason: str = Field(..., example="wrong_language")
    suggested_language: Optional[str] = Field(None, example="eng")
    comment: Optional[str] = Field(None, example="This is English")
    text: str = Field(..., example="Sample text content")
    predicted_language: str = Field(..., example="vie")
    confidence: float = Field(..., example=0.75)
    model_version: str = Field(..., example="v1")
    status: str = Field(..., example="new")
    verified_language: Optional[str] = Field(None, example="eng")
