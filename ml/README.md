# ml/ – thuật toán và đánh giá (B)

## Chạy

```bash
make retrain                              # = python -m scripts.retrain: dữ liệu -> train -> đánh giá -> models/vN/
make retrain ARGS="--algorithm svm"       # baseline | naive_bayes | svm
make retrain ARGS="--skip-data --tune"    # dữ liệu đã có; chọn lại tham số trên val
```

Version mới ở trạng thái `candidate`; đưa lên chạy bằng `POST /api/v1/models/promote` (header `X-Admin-Token`).

Thử nghiệm mà không tạo version (`experiments/output/` không đưa lên git):

```bash
python -m ml.train --out-dir experiments/output/ml/svm --algorithm svm --augment --tune
python -m ml.evaluate --model-dir experiments/output/ml/svm
python -m ml.predict experiments/output/ml/svm "Con sông chảy lặng lẽ qua phố cổ."
```

Luôn chạy dạng `python -m ml.<file>` từ gốc repo (không `cd ml/`), để `data` là package dữ liệu của nhóm.

## Các file

| File | Vai trò |
|---|---|
| `baseline.py`, `naive_bayes.py`, `svm.py` | Mỗi file một thuật toán: `PARAM`, `DEFAULT`, `GRID`, `build_model(value)`; `svm.py` có thêm `calibrate()` |
| `train.py` | Đọc `data/processed/`, tạo vectorizer từ `features.py`, train, lưu `model.joblib` và `train_info.json` |
| `evaluate.py` | Chấm trên `test.csv` và `test_short.csv`, ghi `metrics.json` |
| `predict.py` | Gọi mô hình từ Python hoặc dòng lệnh |
| `features.py` | Các preset đặc trưng TF-IDF n-gram ký tự (A) |

Thêm thuật toán mới: tạo `ml/<tên>.py` có 4 thành phần như trên, rồi thêm tên vào `ALGORITHMS` của `ml/train.py` và `scripts/retrain.py` (test báo nếu hai danh sách lệch nhau).

## Định dạng `models/vN/`

| File | Ghi bởi | Nội dung |
|---|---|---|
| `model.joblib` | `ml.train` | sklearn `Pipeline(features=TfidfVectorizer, classifier=...)`. API nạp trực tiếp và gọi `predict_proba` trên văn bản đã qua `data.common.normalize_text` |
| `train_info.json` | `ml.train` | thuật toán, tham số, kết quả `--tune`, macro-F1 trên val theo độ dài, số mẫu, thời gian |
| `metrics.json` | `ml.evaluate` | `accuracy`, `macro_f1`, `n` trên `test.csv`; `by_length` (20/30/50/100 ký tự và đầy đủ); `no_overlap`; `top_confused_pairs`; `error_samples` (tối đa 2 đoạn sai mỗi độ dài, cho tab Model Evaluation của `web/console`); `mixed_language_check`; `latency_ms_median`, `latency_ms_p95`. API trả nguyên file qua `GET /api/v1/models/{version}` |
| `meta.json` | `scripts/retrain.py` | version, thời điểm tạo, thuật toán, preset, tham số, phiên bản dữ liệu, git commit |

## Dữ liệu và cách chọn tham số

- Chỉ dùng split của nhóm trong `data/processed/` (`python -m data.pipeline`), đã qua kiểm tra rò rỉ.
- Vectorizer học từ vựng trên train gốc; bộ phân loại học thêm `train_crops.csv` (`--augment`, mặc định khi retrain) và `train_feedback.csv` (`--feedback`).
- `--tune` chọn tham số theo **điểm val** = trung bình macro-F1 của val đầy đủ và val cắt ngắn 20/30/50/100 ký tự, vì người dùng thường nhập câu ngắn. `alpha = 0.003` của Naive Bayes là giá trị `--tune` đã chọn; `C = 0.5` của SVM chưa được tune (xem bảng dưới).
- Tập test chỉ dùng trong `ml.evaluate`, không dùng để chọn tham số.
- **SVM:** `LinearSVC` không cho xác suất; softmax của điểm SVM chỉ cho confidence khoảng 0,03 nên API gắn `is_unsure` cho mọi kết quả. Sau khi train, `svm.calibrate()` học hàm sigmoid (Platt) trên val để đổi điểm thành xác suất; dự đoán không đổi, chỉ confidence thay đổi.
- `model.joblib` được thu nhỏ: bỏ `feature_count_` của Naive Bayes, lưu trọng số float32 theo thứ tự Fortran (dự đoán 1 đoạn từ 41 ms còn 1,5 ms với NB), nén joblib.

## Kết quả (12/11/2026)

Preset `char_wb_1_4_100k`, có `--augment`, train trên 301.778 đoạn (train + cắt ngắn). Tập test 113.283 đoạn; các cột "N ký tự" là chính các đoạn test cắt ngắn còn khoảng N ký tự. Mỗi ô: accuracy / macro-F1. "Fit" là thời gian train bộ phân loại, chưa tính khoảng 3 phút tạo đặc trưng.

| Mô hình | Tham số | 20 ký tự | 50 ký tự | 100 ký tự | Đầy đủ | Không chung với train | File | Fit | Dự đoán 1 đoạn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (đoán `ara`) | – | – | – | – | 0,4% / 0,0% | – | – | – | – |
| Naive Bayes | alpha 0,003 (tune) | 82,5% / 82,9% | 90,4% / 90,7% | 93,2% / 93,6% | 94,9% / 95,2% | 94,5% / 94,0% | 18 MB | 23 giây | 1,5 ms |
| Linear SVM + hiệu chỉnh | C 0,5 (chưa tune) | 79,8% / 80,0% | 91,3% / 91,3% | 94,8% / 94,8% | 96,8% / 96,9% | 96,4% / 96,3% | 90 MB | ~36 phút | 9 ms |

- SVM tốt hơn ở văn bản từ 50 ký tự trở lên, Naive Bayes tốt hơn ở văn bản rất ngắn; điểm val (trung bình 5 độ dài) gần bằng nhau: SVM 90,1%, NB 89,9%.
- Độ tin cậy của SVM sau hiệu chỉnh (20.000 đoạn test đầu): median 0,95; 6,3% kết quả dưới ngưỡng 0,8 của API. Phần ≥ 0,8 đúng 99,1%, phần < 0,8 đúng 61,0%, nên `is_unsure` có ý nghĩa.
- Ghép 2 đoạn khác ngôn ngữ: NB đoán ra một trong hai ngôn ngữ ở 94,8% số cặp, SVM 99,8%.
- `make retrain` mặc định dùng Naive Bayes (train nhanh, tốt hơn với câu ngắn); SVM: `make retrain ARGS="--algorithm svm"`.
