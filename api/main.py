"""FastAPI app – phục vụ mô hình cho web (xem openapi.yaml)."""
import os
import sqlite3
from typing import List
from fastapi import FastAPI, HTTPException, status

from api.schemas import (
    PredictRequest,
    PredictResponse,
    Candidate,
    BatchPredictRequest,
    BatchPredictResponse,
    ReportRequest,
    ModelVersion
)

app = FastAPI(
    title="HTTM API - Language Identification System",
    version="0.1.0",
    description="Backend API phục vụ phân loại ngôn ngữ WiLI-2018"
)

# -----------------------------------------------------------------------------
# 1. Khởi tạo SQLite Database cho User Feedback Report
# -----------------------------------------------------------------------------
DB_PATH = "reports.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            predicted_language TEXT NOT NULL,
            correct_language TEXT NOT NULL,
            user_feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


# -----------------------------------------------------------------------------
# 2. Mock Model (Tạm thời phục vụ test API trước khi nạp model ML thật)
# -----------------------------------------------------------------------------
class MockModel:
    def __init__(self):
        self.active_version = "v1"

    def predict(self, text: str) -> PredictResponse:
        text_lower = text.lower()
        
        # Giả lập logic dự đoán ngẫu nhiên dựa trên từ vựng
        if any(w in text_lower for w in ["the", "is", "and", "this"]):
            lang_code, lang_name = "eng", "English"
            confidence = 0.92
        elif any(w in text_lower for w in ["ngôn", "ngữ", "bài", "tập", "chào"]):
            lang_code, lang_name = "vie", "Vietnamese"
            confidence = 0.88
        elif any(w in text_lower for w in ["le", "la", "est", "un", "des"]):
            lang_code, lang_name = "fra", "French"
            confidence = 0.85
        else:
            lang_code, lang_name = "unk", "Unknown / Ambiguous"
            confidence = 0.35

        # Flag unsurity nếu confidence < 0.4
        is_unsure = confidence < 0.4
        
        explanation = (
            ["Văn bản quá ngắn hoặc chứa từ vựng không thuộc n-gram đặc trưng của ngôn ngữ cụ thể (Unsure)"]
            if is_unsure
            else [f"Chứa các n-gram ký tự đặc trưng chuẩn của tiếng {lang_name}"]
        )

        top_candidates = [
            Candidate(code=lang_code, name=lang_name, confidence=confidence),
            Candidate(
                code="eng" if lang_code != "eng" else "fra", 
                name="English" if lang_code != "eng" else "French", 
                confidence=round((1 - confidence) * 0.7, 2)
            ),
            Candidate(
                code="vie" if lang_code != "vie" else "deu", 
                name="Vietnamese" if lang_code != "vie" else "German", 
                confidence=round((1 - confidence) * 0.3, 2)
            )
        ]

        return PredictResponse(
            language_code=lang_code,
            language_name=lang_name,
            confidence=confidence,
            is_unsure=is_unsure,
            explanation=explanation,
            top_candidates=top_candidates
        )

mock_model = MockModel()


# -----------------------------------------------------------------------------
# 3. API Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", summary="Kiểm tra API còn hoạt động")
def health():
    return {"status": "ok"}


@app.post("/api/v1/predict", response_model=PredictResponse, summary="Dự đoán ngôn ngữ cho 1 đoạn văn bản")
def predict(payload: PredictRequest):
    text_len = len(payload.text)
    if text_len < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text length must be between 20 and 5000 characters."
        )
    if text_len > 5000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Text length exceeds maximum allowed limit of 5000 characters."
        )
    
    return mock_model.predict(payload.text)


@app.post("/api/v1/predict-batch", response_model=BatchPredictResponse, summary="Dự đoán hàng loạt (Batch Processing)")
def predict_batch(payload: BatchPredictRequest):
    results = []
    for item in payload.texts:
        if len(item) < 20 or len(item) > 5000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Each text in batch must be between 20 and 5000 characters."
            )
        results.append(mock_model.predict(item))
    return BatchPredictResponse(results=results)


@app.post("/api/v1/report", status_code=status.HTTP_201_CREATED, summary="Gửi báo cáo dự đoán sai")
def report_issue(payload: ReportRequest):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (text, predicted_language, correct_language, user_feedback)
        VALUES (?, ?, ?, ?)
    """, (payload.text, payload.predicted_language, payload.correct_language, payload.user_feedback))
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {"message": "Report saved successfully", "report_id": report_id}


@app.get("/api/v1/models", response_model=List[ModelVersion], summary="Lấy danh sách các phiên bản mô hình")
def list_models():
    return [
        ModelVersion(
            version="v1",
            algorithm="Multinomial Naive Bayes",
            is_active=(mock_model.active_version == "v1"),
            accuracy=0.915,
            macro_f1=0.908
        ),
        ModelVersion(
            version="v2",
            algorithm="Linear SVM",
            is_active=(mock_model.active_version == "v2"),
            accuracy=0.932,
            macro_f1=0.926
        )
    ]


@app.post("/api/v1/models/promote", summary="Promote mô hình")
def promote_model(payload: dict):
    version = payload.get("version")
    if version not in ["v1", "v2"]:
        raise HTTPException(status_code=400, detail="Invalid model version.")
    mock_model.active_version = version
    return {"message": f"Promoted model {version} to active successfully", "active_version": version}


@app.post("/api/v1/models/rollback", summary="Rollback mô hình")
def rollback_model():
    mock_model.active_version = "v1"
    return {"message": "Rolled back to model v1 successfully", "active_version": "v1"}
