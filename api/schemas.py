"""Pydantic schema – kiểm tra input/output của API theo openapi.yaml."""
from typing import List, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. Schemas cho Endpoint Dự Đoán (/predict)
# ==========================================

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        max_length=5000,
        description="Văn bản đầu vào (từ 20 đến 5000 ký tự)",
        example="This is a sample text to test language classification."
    )


class Candidate(BaseModel):
    code: str = Field(..., description="Mã ISO của ngôn ngữ", example="eng")
    name: str = Field(..., description="Tên đầy đủ của ngôn ngữ", example="English")
    confidence: float = Field(..., description="Xác suất độ tin cậy (% / 1.0)", example=0.95)


class PredictResponse(BaseModel):
    language_code: str = Field(..., description="Mã ISO ngôn ngữ dự đoán chính", example="eng")
    language_name: str = Field(..., description="Tên ngôn ngữ dự đoán chính", example="English")
    confidence: float = Field(..., description="Độ tin cậy của class được chọn", example=0.95)
    is_unsure: bool = Field(
        ..., 
        description="True nếu confidence < threshold (ví dụ < 0.4)", 
        example=False
    )
    explanation: List[str] = Field(
        ..., 
        description="Giải thích lý do mô hình phân loại (ví dụ các n-gram ký tự đặc trưng)",
        example=["Contains common English trigrams: 'thi', 'his', 'sam'"]
    )
    top_candidates: List[Candidate] = Field(..., description="Danh sách các ngôn ngữ ứng viên top đầu")


# ===============================================
# 2. Schemas cho Batch Processing (/predict-batch)
# ===============================================

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="Danh sách các đoạn văn bản cần xử lý hàng loạt",
        example=["First sample text to test language.", "Le français est một ngôn ngữ tuyệt vời."]
    )


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


# ==========================================
# 3. Schema cho Báo Cáo Sai (/report)
# ==========================================

class ReportRequest(BaseModel):
    text: str = Field(..., description="Văn bản bị phân loại sai", example="Văn bản bị đoán sai")
    predicted_language: str = Field(..., description="Ngôn ngữ mô hình đã đoán", example="vie")
    correct_language: str = Field(..., description="Ngôn ngữ đúng thực tế", example="eng")
    user_feedback: Optional[str] = Field(None, description="Ghi chú thêm từ người dùng", example="Tiếng Anh pha tiếng Việt")


# ==========================================
# 4. Schema cho Quản Lý Mô Hình (/models)
# ==========================================

class ModelVersion(BaseModel):
    version: str = Field(..., example="v1")
    algorithm: str = Field(..., example="Multinomial Naive Bayes")
    is_active: bool = Field(..., example=True)
    accuracy: float = Field(..., example=0.925)
    macro_f1: float = Field(..., example=0.918)
