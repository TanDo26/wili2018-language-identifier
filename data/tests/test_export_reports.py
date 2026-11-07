import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pandas as pd
import pytest

from data.common import read_split
from data.export_reports import TOKEN_ENV, main, reports_to_feedback
from data.feedback import merge_feedback

TOKEN = "test-token"
TEXT = "Hôm nay trời rất đẹp và chúng tôi đi dạo quanh hồ."


def _report(report_id, status="accepted", verified="vie", text=TEXT):
    return {"report_id": report_id, "status": status, "verified_language": verified, "text": text}


@pytest.fixture
def api():
    """API giả theo contract GET /api/v1/reports: yêu cầu đúng token, trả danh sách ReportItem."""
    state = {"reports": [], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            allowed = self.headers.get("X-Admin-Token") == TOKEN
            body = json.dumps(state["reports"] if allowed else {"detail": "unauthorized"}).encode("utf-8")
            self.send_response(200 if allowed else 401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()


def test_only_accepted_reports_are_exported_with_verified_label():
    feedback = reports_to_feedback([
        _report(1), _report(2, status="rejected"), _report(3, status="new", verified=None),
    ])
    assert feedback.to_dict("records") == [{"text": TEXT, "label": "vie", "report_id": "1"}]


def test_accepted_report_without_verified_language_is_rejected():
    with pytest.raises(ValueError, match="verified_language"):
        reports_to_feedback([_report(7, verified=None)])


def test_export_writes_csv_that_feedback_accepts(api, tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    api["reports"] = [_report(1), _report(2, status="rejected")]
    out = tmp_path / "exports" / "accepted.csv"

    main(["--api-url", api["url"], "--out", str(out)])

    assert api["requests"] == ["/api/v1/reports?status=accepted"]
    splits = {name: pd.DataFrame({"text": ["Một đoạn văn bản hoàn toàn khác"]}) for name in ("train", "val", "test")}
    accepted, _, report = merge_feedback(read_split(out), splits, {"vie"})
    assert report["accepted"] == 1 and accepted["id"].tolist() == ["fb_1"]


def test_export_without_any_accepted_report_writes_empty_csv(api, tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    out = tmp_path / "accepted.csv"

    main(["--api-url", api["url"], "--out", str(out)])

    splits = {name: pd.DataFrame({"text": ["Một đoạn văn bản"]}) for name in ("train", "val", "test")}
    assert merge_feedback(read_split(out), splits, {"vie"})[2]["accepted"] == 0


def test_export_stops_on_missing_or_wrong_token_and_bad_response(api, tmp_path, monkeypatch):
    out = tmp_path / "accepted.csv"
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit, match=TOKEN_ENV):
        main(["--api-url", api["url"], "--out", str(out)])

    monkeypatch.setenv(TOKEN_ENV, "wrong-token")
    with pytest.raises(SystemExit, match="401"):
        main(["--api-url", api["url"], "--out", str(out)])

    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    api["reports"] = [{"report_id": 1}]
    with pytest.raises(SystemExit, match="ReportItem"):
        main(["--api-url", api["url"], "--out", str(out)])
    assert not out.exists()
