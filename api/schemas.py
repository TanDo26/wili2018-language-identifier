"""Request/response schema của API. openapi.yaml được sinh từ đây và api/main.py (python -m api.export_openapi)."""
import enum
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints

MIN_TEXT_CHARS = 20
MAX_TEXT_CHARS = 5000
MAX_BATCH_ITEMS = 100
MAX_COMMENT_CHARS = 500

# Độ dài tính sau khi bỏ khoảng trắng hai đầu (PR-1, PR-3)
TextPayload = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=MIN_TEXT_CHARS, max_length=MAX_TEXT_CHARS),
]
BatchTextsPayload = Annotated[
    List[TextPayload],
    Field(min_length=1, max_length=MAX_BATCH_ITEMS),
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


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["text: String should have at least 20 characters"])


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., examples=[True])
    active_version: Optional[str] = Field(None, examples=["v1"])


# -----------------------------------------------------------------------------
# 1. Prediction
# -----------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: TextPayload = Field(..., examples=["Xin chào, hôm nay trời đẹp quá nên chúng ta đi dạo nhé."])


class Candidate(BaseModel):
    code: str = Field(..., description="Mã ngôn ngữ WiLI-2018", examples=["eng"])
    name: str = Field(..., description="Tên ngôn ngữ", examples=["English"])
    confidence: float = Field(..., description="Xác suất của ngôn ngữ này", examples=[0.95])


class PredictResponse(BaseModel):
    prediction_id: str = Field(..., description="UUID của lần dự đoán, dùng khi gửi /report",
                               examples=["3f6c2a9e-8b1a-4c2d-9e1a-2b3c4d5e6f7a"])
    model_version: str = Field(..., description="Version mô hình đã dự đoán", examples=["v1"])
    language_code: str = Field(..., description="Mã ngôn ngữ dự đoán", examples=["eng"])
    language_name: str = Field(..., description="Tên ngôn ngữ dự đoán", examples=["English"])
    confidence: float = Field(..., description="Xác suất của ngôn ngữ dự đoán", examples=[0.95])
    is_unsure: bool = Field(..., description="True khi confidence thấp hơn ngưỡng UNSURE_THRESHOLD",
                            examples=[False])
    explanation: List[str] = Field(..., description="Các cảnh báo tính từ kết quả mô hình")
    top_candidates: List[Candidate] = Field(..., description="Tối đa 3 ngôn ngữ có xác suất cao nhất")


class BatchPredictRequest(BaseModel):
    texts: BatchTextsPayload


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


# -----------------------------------------------------------------------------
# 2. Ngôn ngữ, version mô hình
# -----------------------------------------------------------------------------

class LanguageItem(BaseModel):
    code: str = Field(..., examples=["eng"])
    name: str = Field(..., examples=["English"])


class ModelVersion(BaseModel):
    version: str = Field(..., examples=["v1"])
    algorithm: Optional[str] = Field(None, description="Từ meta.json", examples=["naive_bayes"])
    status: Literal["active", "candidate", "archived"] = Field(
        ..., description="active: đang chạy; archived: từng active; candidate: chưa từng active")
    created_at: Optional[str] = Field(None, description="Từ meta.json", examples=["2026-11-01T10:00:00+00:00"])
    accuracy: Optional[float] = Field(None, description="Từ metrics.json (tập test)", examples=[0.948])
    macro_f1: Optional[float] = Field(None, description="Từ metrics.json (tập test)", examples=[0.950])


class PromoteRequest(BaseModel):
    version: str = Field(..., examples=["v2"])


class ModelActionResponse(BaseModel):
    message: str
    active_version: str = Field(..., examples=["v2"])


# -----------------------------------------------------------------------------
# 3. Báo cáo dự đoán sai
# -----------------------------------------------------------------------------

class ReportRequest(BaseModel):
    prediction_id: str = Field(..., examples=["3f6c2a9e-8b1a-4c2d-9e1a-2b3c4d5e6f7a"])
    reason: ReportReasonEnum = Field(..., examples=[ReportReasonEnum.WRONG_LANGUAGE])
    suggested_language: Optional[str] = Field(None, description="Một mã trong GET /api/v1/languages",
                                              examples=["eng"])
    comment: Optional[str] = Field(None, max_length=MAX_COMMENT_CHARS, examples=["Đoạn này là tiếng Anh."])


class ReportCreated(BaseModel):
    message: str
    report_id: int = Field(..., examples=[1])


class ReportUpdate(BaseModel):
    status: ReportStatusEnum = Field(..., examples=[ReportStatusEnum.ACCEPTED])
    verified_language: Optional[str] = Field(None, description="Bắt buộc khi status là accepted",
                                             examples=["eng"])


class MessageResponse(BaseModel):
    message: str


class ReportItem(BaseModel):
    report_id: int = Field(..., examples=[1])
    prediction_id: str = Field(..., examples=["3f6c2a9e-8b1a-4c2d-9e1a-2b3c4d5e6f7a"])
    created_at: str = Field(..., description="Giờ UTC", examples=["2026-11-11 12:00:00"])
    reason: ReportReasonEnum = Field(..., examples=[ReportReasonEnum.WRONG_LANGUAGE])
    suggested_language: Optional[str] = Field(None, examples=["eng"])
    comment: Optional[str] = Field(None, examples=["Đoạn này là tiếng Anh."])
    text: str = Field(..., examples=["Sample text content"])
    predicted_language: str = Field(..., examples=["vie"])
    confidence: float = Field(..., examples=[0.75])
    model_version: str = Field(..., examples=["v1"])
    status: ReportStatusEnum = Field(..., examples=[ReportStatusEnum.NEW])
    verified_language: Optional[str] = Field(None, examples=["eng"])
