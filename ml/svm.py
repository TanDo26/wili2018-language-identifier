"""Linear SVM trên TF-IDF n-gram ký tự (ml/features.py).

LinearSVC không cho xác suất; softmax của decision_function cho confidence khoảng 0,03 với 235 lớp, nên
mọi kết quả đều bị API gắn is_unsure. Vì vậy sau khi train, mô hình được hiệu chỉnh xác suất (sigmoid /
Platt) trên tập val; tập test không tham gia.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.svm import LinearSVC

PARAM = "C"
# Chưa tune trên toàn bộ dữ liệu: mỗi lần fit có --augment mất ~36 phút, --tune thử cả 4 giá trị của GRID
DEFAULT = 0.5
GRID = (0.1, 0.3, 1.0, 3.0)


def build_model(value=DEFAULT):
    return LinearSVC(C=value)


def calibrate(model, X_val, y_val):
    """Giữ nguyên SVM đã train, chỉ học thêm hàm sigmoid đổi điểm thành xác suất trên val."""
    return CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(X_val, y_val)
