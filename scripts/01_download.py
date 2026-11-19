"""Download UAV-RSOD (Zenodo record 12606374) into data/raw/, verify md5, extract.

Usage: .venv/bin/python scripts/01_download.py [--keep-zip]
"""
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/12606374"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path, size: int):
    done = 0
    with urllib.request.urlopen(url) as r, open(dst, "wb") as w:
        while chunk := r.read(1 << 22):
            w.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:7.0f} / {size / 1e6:.0f} MB", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(RECORD))
    for f in meta["files"]:
        name, url = f["key"], f["links"]["self"]
        want = f["checksum"].split(":", 1)[1]
        out_dir = RAW / Path(name).stem
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[skip] {out_dir.relative_to(ROOT)} exists")
            continue
        zpath = RAW / name
        if not zpath.exists() or md5sum(zpath) != want:
            print(f"[download] {name} ({f['size'] / 1e9:.2f} GB)")
            fetch(url, zpath, f["size"])
        got = md5sum(zpath)
        if got != want:
            raise SystemExit(f"md5 mismatch for {name}: {got} != {want}")
        print(f"[ok] md5 {name}")
        with zipfile.ZipFile(zpath) as z:
            root = RAW.resolve()
            for m in z.infolist():
                if not (RAW / m.filename).resolve().is_relative_to(root):
                    raise SystemExit(f"unsafe path in zip: {m.filename}")
            z.extractall(RAW)
        print(f"[extract] -> {out_dir.relative_to(ROOT)}")
        if not args.keep_zip:
            zpath.unlink()


if __name__ == "__main__":
    main()
