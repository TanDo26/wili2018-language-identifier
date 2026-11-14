FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Từ data/ API chỉ cần normalize_text và danh sách ngôn ngữ, không cần dataset
COPY data/__init__.py data/common.py ./data/
COPY data/processed/labels.csv ./data/processed/
COPY api/ ./api/

# models/ được mount lúc chạy (docker-compose.yml); DB nằm trên volume ghi được
ENV MODELS_DIR=/app/models \
    DB_PATH=/app/var/reports.db

EXPOSE 8000

# Một worker: mô hình đang active nằm trong RAM của tiến trình
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
