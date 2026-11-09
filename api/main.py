import json
import sqlite3
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import joblib
from fastapi import FastAPI, HTTPException, status

# Import schemas từ file schemas.py của mày
from app.schemas import (
    PredictRequest,
    PredictResponse,
    Candidate,
    BatchPredictRequest,
    BatchPredictResponse,
    ReportRequest,
    ModelVersion
)

# -----------------------------------------------------------------------------
# 1. Biến toàn cục lưu trữ Model & Metadata
# -----------------------------------------------------------------------------
ML_COMPONENTS: Dict[str, Any] = {}
DB_PATH = "database/reports.db"


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            predicted_language TEXT NOT NULL,
            correct_language TEXT,
            user_feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# 2. Lifespan: Load file 1GB và Config vào RAM khi server start
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo DB
    init_db()

    print("--> Đang load Model 1GB và các file cấu hình vào RAM...")
    
    # Đường dẫn đến các file
    model_path = "artifacts/model.joblib"       # Thay bằng tên file 1GB thực tế của mày
    config_path = "artifacts/config_2.json"
    metrics_path = "artifacts/metrics_2.json"

    # Load Config & Metrics
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            ML_COMPONENTS["config"] = json.load(f)

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            ML_COMPONENTS["metrics"] = json.load(f)

    # Load Model nặng ~1GB
    if os.path.exists(model_path):
        ML_COMPONENTS["model"] = joblib.load(model_path)
        print("--> Load Model thành công!")
    else:
        print(f"--> WARN: Không tìm thấy file model tại {model_path}")

    ML_COMPONENTS["active_version"] = ML_COMPONENTS.get("config", {}).get("version", "v1")

    yield  # Ứng dụng chạy ở đây

    # Clean up khi shutdown server (nếu cần)
    ML_COMPONENTS.clear()
    print("--> Đã giải phóng bộ nhớ RAM.")


# -----------------------------------------------------------------------------
# 3. Khởi tạo FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="HTTM API - Language Identification System",
    version="0.1.0",
    description="Backend API phục vụ phân loại ngôn ngữ WiLI-2018",
    lifespan=lifespan
)


# -----------------------------------------------------------------------------
# 4. Logic Dự đoán thực tế (Inference Logic)
# -----------------------------------------------------------------------------
def predict_text(text: str) -> PredictResponse:
    model = ML_COMPONENTS.get("model")
    config = ML_COMPONENTS.get("config", {})

    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model chưa sẵn sàng hoặc load thất bại."
        )

    text_len = len(text)

    # 1. Dự đoán xác suất (Predict Probabilities)
    # Giả sử model là Pipeline (Vectorizer + Classifier) có predict_proba
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_  # Danh sách nhãn ISO code

    # Sắp xếp lấy Top candidates
    top_indices = probabilities.argsort()[::-1][:3]
    top_candidates = []

    for idx in top_indices:
        code = str(classes[idx])
        conf = float(round(probabilities[idx], 4))
        # Có thể dùng config["iso_codes"] để lấy tên chuẩn nếu có dict ánh xạ
        name = config.get("iso_codes", {}).get(code, code)
        top_candidates.append(Candidate(code=code, name=name, confidence=conf))

    top_1 = top_candidates[0]
    confidence = top_1.confidence
    is_unsure = confidence < 0.4 or text_len < 140

    # Cảnh báo / giải thích
    explanations = []
    if text_len < 140:
        explanations.append("Warning: Text length is under 140 characters, prediction confidence may be lower.")
    if confidence < 0.4:
        explanations.append("Low confidence score: text contains ambiguous features.")
    else:
        explanations.append(f"Contains strong characteristic n-gram features for {top_1.name}.")

    return PredictResponse(
        language_code=top_1.code,
        language_name=top_1.name,
        confidence=confidence,
        is_unsure=is_unsure,
        explanation=explanations,
        top_candidates=top_candidates
    )


# -----------------------------------------------------------------------------
# 5. API Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", summary="Kiểm tra API còn hoạt động")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in ML_COMPONENTS,
        "active_version": ML_COMPONENTS.get("active_version", "unknown")
    }


@app.post("/api/v1/predict", response_model=PredictResponse, summary="Dự đoán ngôn ngữ cho 1 đoạn văn bản")
def predict(payload: PredictRequest):
    text_len = len(payload.text.strip())
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

    return predict_text(payload.text)


@app.post("/api/v1/predict-batch", response_model=BatchPredictResponse, summary="Dự đoán hàng loạt")
def predict_batch(payload: BatchPredictRequest):
    results = []
    for item in payload.texts:
        item_len = len(item.strip())
        if item_len < 20 or item_len > 5000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Each text in batch must be between 20 and 5000 characters."
            )
        results.append(predict_text(item))
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
    metrics = ML_COMPONENTS.get("metrics", {}).get("val", {})
    acc = metrics.get("accuracy", 0.944)
    f1 = metrics.get("macro_f1", 0.949)
    active_ver = ML_COMPONENTS.get("active_version", "v1")

    return [
        ModelVersion(
            version="v1",
            algorithm="Multinomial Naive Bayes",
            is_active=(active_ver == "v1"),
            accuracy=acc,
            macro_f1=f1
        ),
        ModelVersion(
            version="v2",
            algorithm="Linear SVM",
            is_active=(active_ver == "v2"),
            accuracy=0.932,
            macro_f1=0.926
        )
    ]
