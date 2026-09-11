"""Fetch the Runs N' Poses release files listed in config/datasets.yaml and verify sha256.

    uv run python scripts/01_fetch_data.py            # fetch what is missing, verify everything
    uv run python scripts/01_fetch_data.py --verify   # verify only

Files with `sha256: null` in the config are skipped (not needed for the table).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "datasets.yaml").read_text())["runs_n_poses"]
ZENODO_RECORD = CFG["zenodo_record"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest: Path) -> None:
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{name}/content"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    local = ROOT / CFG["local_dir"]
    local.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, info in CFG["files"].items():
        want = info.get("sha256")
        if not want or want == "PENDING":
            print(f"skip   {name} (no checksum in config)")
            continue
        dest = local / name
        if not dest.exists():
            if args.verify:
                print(f"MISSING {name}"); bad += 1; continue
            print(f"fetch  {name}"); fetch(name, dest)
        got = sha256(dest)
        ok = got == want
        bad += not ok
        print(f"{'ok    ' if ok else 'BAD   '} {name}  {got[:12]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
