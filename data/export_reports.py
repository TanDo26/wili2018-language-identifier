"""Xuất các báo cáo lỗi đã duyệt từ API thành CSV để retrain (yêu cầu RV-3).

    $env:ADMIN_TOKEN = "..."     # bash: export ADMIN_TOKEN=...
    python -m data.export_reports --api-url http://localhost:8000
    python -m scripts.retrain --feedback exports/accepted_reports.csv

Đọc GET /api/v1/reports?status=accepted (header X-Admin-Token) theo schema ReportItem, rồi ghi
CSV gồm text, label (verified_language do Data Scientist gán), report_id mà data/feedback.py nhận.
File có văn bản người dùng nhập nên thư mục exports/ không được đưa lên git.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from data.common import ROOT, rel, setup_console, write_split

DEFAULT_OUT = ROOT / "exports" / "accepted_reports.csv"
REPORTS_PATH = "/api/v1/reports?status=accepted"
TOKEN_ENV = "ADMIN_TOKEN"


def fetch_reports(api_url: str, token: str, timeout: int = 30) -> list[dict]:
    request = urllib.request.Request(api_url.rstrip("/") + REPORTS_PATH, headers={"X-Admin-Token": token})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def reports_to_feedback(reports: list[dict]) -> pd.DataFrame:
    """Chỉ giữ báo cáo accepted, kể cả khi API trả thêm trạng thái khác."""
    accepted = [r for r in reports if r["status"] == "accepted"]
    missing = [r["report_id"] for r in accepted if not r.get("verified_language")]
    if missing:
        raise ValueError(f"báo cáo accepted thiếu verified_language: {missing[:5]}")
    return pd.DataFrame(
        [{"text": r["text"], "label": r["verified_language"], "report_id": str(r["report_id"])} for r in accepted],
        columns=["text", "label", "report_id"],
    )


def main(argv=None) -> Path:
    setup_console()
    parser = argparse.ArgumentParser(description="Xuất báo cáo đã duyệt từ API thành CSV")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"Chưa đặt biến môi trường {TOKEN_ENV} (giá trị của header X-Admin-Token).")
    try:
        reports = fetch_reports(args.api_url, token)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"API trả mã {e.code} cho {REPORTS_PATH}.") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"Không kết nối được {args.api_url}: {e.reason}") from None
    try:
        feedback = reports_to_feedback(reports)
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f"Phản hồi API không đúng schema ReportItem: {e!r}") from None

    write_split(feedback, args.out)
    print(f"{len(feedback)}/{len(reports)} báo cáo accepted -> {rel(args.out)}")
    print(f"Retrain: python -m scripts.retrain --feedback {rel(args.out)}")
    return args.out


if __name__ == "__main__":
    main()
