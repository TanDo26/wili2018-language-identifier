"""FastAPI app phục vụ mô hình nhận diện ngôn ngữ. Hợp đồng API: openapi.yaml (sinh bằng api.export_openapi).

Biến môi trường:
  MODELS_DIR        thư mục chứa các version models/vN/ do `make retrain` tạo (mặc định <repo>/models)
  DB_PATH           file SQLite lưu dự đoán, báo cáo và trạng thái version (mặc định <repo>/database/reports.db)
  ADMIN_TOKEN       giá trị header X-Admin-Token cho các endpoint admin; không đặt thì các endpoint này trả 503
  UNSURE_THRESHOLD  confidence thấp hơn ngưỡng này thì is_unsure = true (mặc định 0.8)

Mô hình đang chạy nằm trong RAM của tiến trình, nên chạy một worker (như api/Dockerfile).
"""
import csv
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import joblib
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Candidate,
    ErrorResponse,
    HealthResponse,
    LanguageItem,
    MessageResponse,
    ModelActionResponse,
    ModelVersion,
    PredictRequest,
    PredictResponse,
    PromoteRequest,
    ReportCreated,
    ReportItem,
    ReportRequest,
    ReportStatusEnum,
    ReportUpdate,
)
from data.common import normalize_text  # chuẩn hóa giống lúc train

log = logging.getLogger("api")

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels.csv"
VERSION_RE = re.compile(r"v(\d+)")  # tên thư mục scripts/retrain.py tạo: v1, v2, ...
TOP_K = 3
CLOSE_MARGIN = 0.15
# Vượt độ dài ở các trường này thì trả 413 (PR-1, PR-3); trường khác (vd. comment) trả 400
PAYLOAD_FIELDS = {"text", "texts"}


# -----------------------------------------------------------------------------
# Cấu hình (đọc lúc gọi để test đổi được bằng biến môi trường)
# -----------------------------------------------------------------------------
def get_models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR") or BASE_DIR / "models")


def get_db_path() -> Path:
    return Path(os.getenv("DB_PATH") or BASE_DIR / "database" / "reports.db")


def get_admin_token() -> str:
    return os.getenv("ADMIN_TOKEN", "")


def get_unsure_threshold() -> float:
    return float(os.getenv("UNSURE_THRESHOLD") or 0.8)


# -----------------------------------------------------------------------------
# Trạng thái trong RAM
# -----------------------------------------------------------------------------
# "active" là tuple (version, model) để mỗi request luôn đọc được cặp khớp nhau
STATE: Dict[str, Any] = {}
STATE_LOCK = threading.Lock()  # promote / rollback chạy lần lượt


def reset_state() -> None:
    STATE.clear()
    STATE.update(active=None, labels={})


reset_state()


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    """Mở kết nối, commit khi khối lệnh chạy xong, rollback nếu có exception."""
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                predicted_language TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                suggested_language TEXT,
                comment TEXT,
                status TEXT DEFAULT 'new',
                verified_language TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
            );
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)


def load_model_state() -> Dict[str, Any]:
    """Trạng thái version lưu trong DB: version active, lịch sử để rollback, các version từng active."""
    with db() as conn:
        row = conn.execute("SELECT value FROM system_config WHERE key = 'model_state'").fetchone()
    saved = json.loads(row["value"]) if row else {}
    return {
        "active": saved.get("active"),
        "history": list(saved.get("history", [])),
        "activated": list(saved.get("activated", [])),
    }


def save_model_state(state: Dict[str, Any]) -> None:
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO system_config (key, value) VALUES ('model_state', ?)",
                     (json.dumps(state),))


# -----------------------------------------------------------------------------
# Ngôn ngữ và version mô hình
# -----------------------------------------------------------------------------
def load_labels() -> Dict[str, str]:
    if not LABELS_PATH.exists():
        log.warning("Không thấy %s: /languages trả danh sách rỗng và mọi mã ngôn ngữ bị từ chối", LABELS_PATH)
        return {}
    with open(LABELS_PATH, encoding="utf-8", newline="") as f:
        return {row["code"]: row["name"] for row in csv.DictReader(f)}


def available_versions() -> List[str]:
    """Các thư mục models/vN có model.joblib, theo thứ tự N tăng dần."""
    models_dir = get_models_dir()
    if not models_dir.is_dir():
        return []
    found = [p.name for p in models_dir.iterdir()
             if p.is_dir() and VERSION_RE.fullmatch(p.name) and (p / "model.joblib").is_file()]
    return sorted(found, key=lambda v: int(v[1:]))


def active_version() -> Optional[str]:
    active = STATE["active"]
    return active[0] if active else None


def activate(version: str) -> None:
    """Nạp model.joblib xong mới thay mô hình đang chạy, nên nạp lỗi thì mô hình cũ vẫn giữ nguyên."""
    model = joblib.load(get_models_dir() / version / "model.joblib")
    STATE["active"] = (version, model)


def restore_active_model() -> None:
    """Nạp version active lưu trong DB; nếu DB chưa có (lần chạy đầu) thì dùng version mới nhất."""
    versions = available_versions()
    state = load_model_state()
    version = state["active"]
    if version not in versions:
        if version:
            log.warning("Version active %s không còn trong %s", version, get_models_dir())
        version = versions[-1] if versions else None
    if version is None:
        log.warning("Không có version nào trong %s: /predict trả 503", get_models_dir())
        return
    try:
        activate(version)
    except Exception:
        log.exception("Không nạp được mô hình %s", version)
        return
    if version != state["active"]:
        state["active"] = version
        if version not in state["activated"]:
            state["activated"].append(version)
        save_model_state(state)
    log.info("Đang chạy mô hình %s", version)


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("Không đọc được %s", path)
        return {}


# -----------------------------------------------------------------------------
# App, xử lý lỗi, xác thực admin
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    reset_state()
    init_db()
    STATE["labels"] = load_labels()
    restore_active_model()
    yield
    reset_state()


app = FastAPI(
    title="HTTM API - Language Identification System",
    version="0.2.0",
    description="Thỏa thuận API giữa C (backend) và D (frontend). Mọi lỗi trả dạng {\"detail\": \"...\"}.",
    lifespan=lifespan,
)

ERROR_DESCRIPTIONS = {
    400: "Input không hợp lệ",
    401: "Thiếu hoặc sai header X-Admin-Token",
    404: "Không tìm thấy",
    409: "Xung đột trạng thái version",
    413: "Văn bản dài hơn 5.000 ký tự hoặc batch nhiều hơn 100 đoạn",
    500: "Không nạp được mô hình",
    503: "Chưa có mô hình (predict) hoặc server chưa đặt ADMIN_TOKEN (endpoint admin)",
}


def errors(*codes: int) -> Dict[int, Dict[str, Any]]:
    return {code: {"model": ErrorResponse, "description": ERROR_DESCRIPTIONS[code]} for code in codes}


def _is_payload_too_large(error: Dict[str, Any]) -> bool:
    loc = error.get("loc", ())
    return (error["type"] in ("string_too_long", "too_long")
            and len(loc) >= 2 and loc[0] == "body" and loc[1] in PAYLOAD_FIELDS)


def _error_message(error: Dict[str, Any]) -> str:
    field = ".".join(str(part) for part in error.get("loc", ())[1:])
    return f"{field}: {error['msg']}" if field else error["msg"]


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Lỗi validation trả 400, riêng văn bản / batch quá dài trả 413 (FastAPI mặc định trả 422)."""
    errs = exc.errors()
    code = status.HTTP_413_CONTENT_TOO_LARGE if any(map(_is_payload_too_large, errs)) else status.HTTP_400_BAD_REQUEST
    return JSONResponse(status_code=code, content={"detail": "; ".join(map(_error_message, errs))})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    log.exception("Lỗi không lường trước ở %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


admin_token_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


def require_admin(token: Optional[str] = Security(admin_token_header)) -> None:
    expected = get_admin_token()
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Admin API is disabled: ADMIN_TOKEN is not set on the server.")
    if not token or not secrets.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing admin token.")


ADMIN = {"dependencies": [Depends(require_admin)]}


# -----------------------------------------------------------------------------
# Dự đoán
# -----------------------------------------------------------------------------
def predict_texts(texts: List[str]) -> List[PredictResponse]:
    active = STATE["active"]
    if active is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not loaded or unavailable.")
    version, model = active
    labels = STATE["labels"]
    threshold = get_unsure_threshold()

    probabilities = model.predict_proba([normalize_text(text) for text in texts])
    classes = [str(c) for c in model.classes_]

    results, rows = [], []
    for text, probs in zip(texts, probabilities):
        candidates = [
            Candidate(code=classes[i], name=labels.get(classes[i], classes[i]), confidence=round(float(probs[i]), 4))
            for i in probs.argsort()[::-1][:TOP_K]
        ]
        best = candidates[0]
        is_unsure = best.confidence < threshold
        explanation = []
        if is_unsure:
            explanation.append(f"Low confidence: {best.confidence:.2f} is below the threshold {threshold:.2f}.")
        if len(candidates) > 1 and best.confidence - candidates[1].confidence < CLOSE_MARGIN:
            explanation.append(f"'{best.name}' and '{candidates[1].name}' have close confidence scores.")

        prediction_id = str(uuid.uuid4())
        rows.append((prediction_id, text, best.code, best.confidence, version))
        results.append(PredictResponse(
            prediction_id=prediction_id,
            model_version=version,
            language_code=best.code,
            language_name=best.name,
            confidence=best.confidence,
            is_unsure=is_unsure,
            explanation=explanation,
            top_candidates=candidates,
        ))

    with db() as conn:  # /report tra cứu lại văn bản và kết quả theo prediction_id
        conn.executemany(
            "INSERT INTO predictions (prediction_id, text, predicted_language, confidence, model_version) "
            "VALUES (?, ?, ?, ?, ?)", rows)
    return results


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, summary="Kiểm tra API còn hoạt động")
def health():
    version = active_version()
    return HealthResponse(status="ok", model_loaded=version is not None, active_version=version)


@app.get("/api/v1/languages", response_model=List[LanguageItem], summary="Danh sách ngôn ngữ mô hình nhận diện")
def list_languages():
    return [LanguageItem(code=code, name=name) for code, name in STATE["labels"].items()]


@app.post("/api/v1/predict", response_model=PredictResponse, responses=errors(400, 413, 503),
          summary="Dự đoán ngôn ngữ cho 1 đoạn văn bản")
def predict(payload: PredictRequest):
    return predict_texts([payload.text])[0]


@app.post("/api/v1/predict-batch", response_model=BatchPredictResponse, responses=errors(400, 413, 503),
          summary="Dự đoán ngôn ngữ cho tối đa 100 đoạn văn bản")
def predict_batch(payload: BatchPredictRequest):
    return BatchPredictResponse(results=predict_texts(payload.texts))


@app.post("/api/v1/report", status_code=status.HTTP_201_CREATED, response_model=ReportCreated,
          responses=errors(400, 404), summary="Báo cáo một dự đoán sai")
def create_report(payload: ReportRequest):
    if payload.suggested_language and payload.suggested_language not in STATE["labels"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Invalid suggested_language code: '{payload.suggested_language}'.")
    with db() as conn:
        found = conn.execute("SELECT 1 FROM predictions WHERE prediction_id = ?", (payload.prediction_id,)).fetchone()
        if not found:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Prediction ID '{payload.prediction_id}' not found.")
        report_id = conn.execute(
            "INSERT INTO reports (prediction_id, reason, suggested_language, comment) VALUES (?, ?, ?, ?)",
            (payload.prediction_id, payload.reason.value, payload.suggested_language, payload.comment)).lastrowid
    return ReportCreated(message="Report created successfully", report_id=report_id)


@app.get("/api/v1/reports", response_model=List[ReportItem], responses=errors(400, 401, 503),
         summary="Danh sách báo cáo (admin)", **ADMIN)
def list_reports(
    report_status: Optional[ReportStatusEnum] = Query(None, alias="status"),
    model_version: Optional[str] = Query(None),
):
    query = """
        SELECT r.id AS report_id, r.prediction_id, r.created_at, r.reason,
               r.suggested_language, r.comment, r.status, r.verified_language,
               p.text, p.predicted_language, p.confidence, p.model_version
        FROM reports r
        JOIN predictions p ON r.prediction_id = p.prediction_id
        WHERE 1 = 1
    """
    params: List[Any] = []
    if report_status:
        query += " AND r.status = ?"
        params.append(report_status.value)
    if model_version:
        query += " AND p.model_version = ?"
        params.append(model_version)
    query += " ORDER BY r.id DESC"

    with db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [ReportItem(**{**dict(row), "created_at": str(row["created_at"])}) for row in rows]


@app.patch("/api/v1/reports/{report_id}", response_model=MessageResponse, responses=errors(400, 401, 404, 503),
           summary="Duyệt một báo cáo (admin)", **ADMIN)
def update_report(report_id: int, payload: ReportUpdate):
    if payload.status == ReportStatusEnum.ACCEPTED and not payload.verified_language:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "verified_language is required when status is accepted.")
    if payload.verified_language and payload.verified_language not in STATE["labels"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Invalid verified_language code: '{payload.verified_language}'.")
    with db() as conn:
        cursor = conn.execute("UPDATE reports SET status = ?, verified_language = ? WHERE id = ?",
                              (payload.status.value, payload.verified_language, report_id))
        if cursor.rowcount == 0:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Report ID {report_id} not found.")
    return MessageResponse(message="Report status updated successfully")


@app.get("/api/v1/models", response_model=List[ModelVersion], summary="Danh sách version mô hình")
def list_models():
    activated = load_model_state()["activated"]
    current = active_version()
    versions = []
    for version in available_versions():
        version_dir = get_models_dir() / version
        metrics = read_json_file(version_dir / "metrics.json")
        meta = read_json_file(version_dir / "meta.json")
        if version == current:
            version_status = "active"
        elif version in activated:
            version_status = "archived"
        else:
            version_status = "candidate"
        versions.append(ModelVersion(
            version=version,
            algorithm=meta.get("algorithm"),
            status=version_status,
            created_at=meta.get("created_at"),
            accuracy=metrics.get("accuracy"),
            macro_f1=metrics.get("macro_f1"),
        ))
    return versions


@app.post("/api/v1/models/promote", response_model=ModelActionResponse,
          responses=errors(400, 401, 404, 409, 500, 503), summary="Đưa một version lên chạy (admin)", **ADMIN)
def promote_model(payload: PromoteRequest):
    version = payload.version
    with STATE_LOCK:
        if version not in available_versions():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model version '{version}' does not exist.")
        current = active_version()
        if version == current:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Model version '{version}' is already active.")
        try:
            activate(version)
        except Exception as exc:
            log.exception("Không nạp được mô hình %s", version)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                                f"Failed to load model version '{version}'.") from exc
        state = load_model_state()
        if current:
            state["history"].append(current)
        state["active"] = version
        if version not in state["activated"]:
            state["activated"].append(version)
        save_model_state(state)
    return ModelActionResponse(message=f"Promoted model version {version}.", active_version=version)


@app.post("/api/v1/models/rollback", response_model=ModelActionResponse,
          responses=errors(401, 404, 409, 500, 503), summary="Quay về version chạy trước đó (admin)", **ADMIN)
def rollback_model():
    with STATE_LOCK:
        state = load_model_state()
        if not state["history"]:
            raise HTTPException(status.HTTP_409_CONFLICT, "No previous model version available to roll back to.")
        previous = state["history"][-1]
        if previous not in available_versions():
            raise HTTPException(status.HTTP_404_NOT_FOUND,
                                f"Model artifact for previous version '{previous}' no longer exists.")
        try:
            activate(previous)
        except Exception as exc:
            log.exception("Không nạp được mô hình %s", previous)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                                f"Failed to load model version '{previous}'.") from exc
        state["history"].pop()
        state["active"] = previous
        save_model_state(state)
    return ModelActionResponse(message=f"Rolled back to model version {previous}.", active_version=previous)


# -----------------------------------------------------------------------------
# OpenAPI: bỏ response 422 FastAPI tự thêm, vì lỗi validation ở đây trả 400 / 413
# -----------------------------------------------------------------------------
_default_openapi = app.openapi


def custom_openapi() -> Dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = _default_openapi()
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            operation.get("responses", {}).pop("422", None)
    for name in ("HTTPValidationError", "ValidationError"):
        schema.get("components", {}).get("schemas", {}).pop(name, None)
    return schema


app.openapi = custom_openapi
