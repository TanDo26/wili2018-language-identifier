"""Ghi openapi.yaml từ app FastAPI để hợp đồng API giữa C và D luôn khớp với code.

    python -m api.export_openapi            # ghi lại openapi.yaml sau khi sửa endpoint / schema
    python -m api.export_openapi --check    # báo lỗi nếu openapi.yaml đã cũ

Không cần PyYAML: dữ liệu OpenAPI chỉ gồm dict / list / chuỗi / số / bool, chuỗi được viết theo cú pháp
JSON (cũng hợp lệ trong YAML) trừ khi là một từ đơn giản.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List

from api.main import app
from data.common import setup_console

OPENAPI_PATH = Path(__file__).resolve().parent.parent / "openapi.yaml"
HEADER = "# Sinh tự động từ api/main.py và api/schemas.py, không sửa tay: python -m api.export_openapi\n"
_PLAIN = re.compile(r"[A-Za-z_/][A-Za-z0-9_./-]*")
_RESERVED = {"y", "n", "yes", "no", "on", "off", "true", "false", "null"}  # YAML đọc thành bool / null


def _scalar(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value)
    if isinstance(value, str) and _PLAIN.fullmatch(value) and value.lower() not in _RESERVED:
        return value
    return json.dumps(str(value), ensure_ascii=False)


def _inline(value: Any) -> str:
    if isinstance(value, dict):
        return "{}"
    if isinstance(value, list):
        return "[]"
    return _scalar(value)


def _lines(obj: Any, depth: int) -> List[str]:
    pad = "  " * depth
    out = []
    items = obj.items() if isinstance(obj, dict) else ((None, v) for v in obj)
    for key, value in items:
        prefix = f"{pad}{_scalar(str(key))}:" if isinstance(obj, dict) else f"{pad}-"
        if isinstance(value, (dict, list)) and value:
            if isinstance(obj, dict):
                out.append(prefix)
                out += _lines(value, depth + 1)
            else:  # phần tử list là dict / list: dòng đầu nằm cùng dòng với "-"
                nested = _lines(value, depth + 1)
                out.append(f"{prefix} {nested[0].lstrip()}")
                out += nested[1:]
        else:
            out.append(f"{prefix} {_inline(value)}")
    return out


def render(schema: dict) -> str:
    return HEADER + "\n".join(_lines(schema, 0)) + "\n"


def main(argv=None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Ghi openapi.yaml từ app FastAPI")
    parser.add_argument("--check", action="store_true", help="chỉ kiểm tra, không ghi file")
    args = parser.parse_args(argv)

    text = render(app.openapi())
    if args.check:
        current = OPENAPI_PATH.read_text(encoding="utf-8").replace("\r\n", "\n") if OPENAPI_PATH.exists() else ""
        if current != text:
            print("openapi.yaml đã cũ: chạy python -m api.export_openapi", file=sys.stderr)
            return 1
        print("openapi.yaml khớp với code")
        return 0
    OPENAPI_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"Đã ghi {OPENAPI_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
