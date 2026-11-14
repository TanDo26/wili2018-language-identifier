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
# Dữ liệu -> train -> đánh giá -> models/vN/ (Windows không có make: python -m scripts.retrain)
make retrain
make retrain ARGS="--algorithm svm"      # baseline | naive_bayes | svm, xem ml/README.md

# Chạy API local
uvicorn api.main:app --reload

# Chạy test (Windows: python -m pytest data/tests experiments/tests scripts/tests ml/tests api/tests)
make test

# Chạy bằng Docker
docker compose up --build
```

API docs: http://localhost:8000/docs

### API

API đọc mô hình từ `models/vN/` (tạo bằng `make retrain`). Khi DB chưa lưu version nào, API dùng version có số lớn nhất; các version mới sau đó là `candidate` cho tới khi được promote. Cấu hình bằng biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODELS_DIR` | `models/` | Thư mục chứa các version `vN/` |
| `DB_PATH` | `database/reports.db` | SQLite lưu dự đoán, báo cáo, version đang active |
| `ADMIN_TOKEN` | (trống) | Giá trị header `X-Admin-Token` của `/reports`, `/models/promote`, `/models/rollback`; không đặt thì các endpoint này trả 503 |
| `UNSURE_THRESHOLD` | `0.8` | Confidence thấp hơn ngưỡng này thì `is_unsure = true` |

Với Docker, đặt `ADMIN_TOKEN=...` trong file `.env` cạnh `docker-compose.yml` (file này không đưa lên git).

`openapi.yaml` được sinh từ code: sau khi sửa endpoint hoặc schema, chạy `python -m api.export_openapi` (test sẽ báo nếu file đã cũ).
