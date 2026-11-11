"""Tải WiLI-2018 từ Zenodo về data/raw/ và giải nén vào data/raw/wili-2018/.

    python -m data.download            # bỏ qua nếu đã có file zip hợp lệ
    python -m data.download --force    # tải lại từ đầu
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from data.common import RAW_DIR, WILI_DIR, file_hash, read_lines, setup_console

RECORD_URL = "https://zenodo.org/records/841984"
DOI = "10.5281/zenodo.841984"
LICENSE = "ODbL-1.0"
URL = f"{RECORD_URL}/files/wili-2018.zip?download=1"
# Checksum công bố trên Zenodo (https://zenodo.org/api/records/841984)
EXPECTED_MD5 = "3dc5bd41587811ad6b0d04ae2f235f84"
ZIP_PATH = RAW_DIR / "wili-2018.zip"

REQUIRED_FILES = ("x_train.txt", "y_train.txt", "x_test.txt", "y_test.txt", "labels.csv")
EXPECTED_ROWS = 117_500  # 235 ngôn ngữ x 500 đoạn, cho mỗi tập train / test


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Đang tải {url}")
    with urllib.request.urlopen(url, timeout=60) as res, open(tmp, "wb") as f:
        total = int(res.headers.get("Content-Length") or 0)
        done = 0
        while chunk := res.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def verify_zip(path: Path, expected_md5: str | None) -> bool:
    if not path.exists():
        return False
    if expected_md5 is None:
        return True
    actual = file_hash(path, "md5")
    if actual != expected_md5:
        print(f"Sai checksum {path.name}: md5={actual}, mong đợi {expected_md5}")
        return False
    return True


def extract(zip_path: Path, out_dir: Path) -> None:
    """Giải nén và đưa các file cần thiết về thẳng out_dir (kể cả khi zip có thư mục con)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name or name.startswith("."):
                continue
            with zf.open(info) as src, open(out_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def check_extracted(out_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (out_dir / f).exists()]
    if missing:
        sys.exit(f"Thiếu file sau khi giải nén: {missing}")
    for split in ("train", "test"):
        n_x = len(read_lines(out_dir / f"x_{split}.txt"))
        n_y = len(read_lines(out_dir / f"y_{split}.txt"))
        if n_x != n_y:
            sys.exit(f"x_{split}.txt có {n_x} dòng nhưng y_{split}.txt có {n_y} dòng")
        if n_x != EXPECTED_ROWS:
            print(f"Cảnh báo: {split} có {n_x} dòng (mong đợi {EXPECTED_ROWS})")
        print(f"  {split}: {n_x} dòng")


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=URL, help="URL thay thế nếu Zenodo không truy cập được")
    parser.add_argument("--force", action="store_true", help="tải lại kể cả khi đã có file")
    parser.add_argument("--no-verify", action="store_true",
                        help="bỏ kiểm tra md5 (chỉ dùng khi tải từ nguồn khác)")
    args = parser.parse_args(argv)

    expected_md5 = None if args.no_verify else EXPECTED_MD5

    if args.force or not verify_zip(ZIP_PATH, expected_md5):
        download(args.url, ZIP_PATH)
        if not verify_zip(ZIP_PATH, expected_md5):
            sys.exit("File tải về không khớp checksum – hãy thử lại với --force.")
    else:
        print(f"Đã có {ZIP_PATH.name} (checksum hợp lệ), bỏ qua bước tải.")

    print(f"Giải nén vào {WILI_DIR}")
    extract(ZIP_PATH, WILI_DIR)
    check_extracted(WILI_DIR)
    print("Xong.")


if __name__ == "__main__":
    main()
