# Trên Windows không có make thì chạy trực tiếp lệnh python tương ứng.
PYTHON ?= python
ARGS ?=

.PHONY: data eda experiments retrain test

# Tải -> làm sạch -> chia -> kiểm tra rò rỉ -> cắt ngắn -> thống kê
data:
	$(PYTHON) -m data.pipeline

eda:
	$(PYTHON) -m data.pipeline --eda

# make experiments ARGS="--presets char_wb_1_3_100k --classifiers nb sgd --augment both"
experiments:
	$(PYTHON) -m experiments.feature_search $(ARGS)

# make retrain ARGS="--algorithm svm"
# make retrain ARGS="--feedback exports/accepted_reports.csv"  (file tạo bằng python -m data.export_reports)
retrain:
	$(PYTHON) -m scripts.retrain $(ARGS)

test:
	$(PYTHON) -m pytest data/tests experiments/tests scripts/tests api/tests
