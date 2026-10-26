# HTTM Assignment

## Cấu trúc thư mục

```
├── data/              # A – tải, làm sạch, chia dữ liệu, kiểm tra rò rỉ
├── ml/                # B – train.py, evaluate.py, baseline, Naive Bayes, SVM
├── api/               # C – FastAPI, kiểm tra input, tests/
├── web/               # D – giao diện end-user và console Data Scientist
├── docs/              # D – báo cáo, slide, use-case diagram, kiến trúc
├── models/            # các version mô hình v1, v2… (không đưa lên git)
├── openapi.yaml       # thỏa thuận API giữa C và D
├── docker-compose.yml
└── README.md
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy

```bash
# Chạy API local
uvicorn api.main:app --reload

# Chạy test
pytest api/tests

# Chạy bằng Docker
docker compose up --build
```

API docs: http://localhost:8000/docs
