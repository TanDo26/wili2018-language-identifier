"""FastAPI app – phục vụ mô hình cho web (xem openapi.yaml)."""
from fastapi import FastAPI

app = FastAPI(title="HTTM API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}
