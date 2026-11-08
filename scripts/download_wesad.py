"""Download WESAD from the official source (University of Siegen) and build a local cache.

Usage
-----
    python scripts/download_wesad.py                 # -> data/WESAD.zip, data/WESAD/cache/S*_all.npz
    python scripts/download_wesad.py --keep-pkl      # also keep the extracted pickles

WESAD is licensed for non-commercial academic use only and may not be
redistributed. Everything is written below ``data/``, which is git-ignored.
No unofficial mirrors are used.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from fusion.io import SUBJECT_IDS, cache_file, cache_subject

WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while block := r.read(chunk):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--keep-pkl", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    zip_path = args.data_dir / "WESAD.zip"
    out_dir = args.data_dir / "WESAD"
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading {WESAD_URL} -> {zip_path} (~2.2 GB)")
        download(WESAD_URL, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for extra in ["WESAD/wesad_readme.pdf"] + [f"WESAD/{s}/{s}_readme.txt" for s in SUBJECT_IDS]:
            if extra in names:
                z.extract(extra, args.data_dir)
        for s in SUBJECT_IDS:
            member = f"WESAD/{s}/{s}.pkl"
            cache = cache_file(out_dir, s)
            if cache.exists() and not args.rebuild:
                print(f"{s}: cached")
                continue
            if member not in names:
                print(f"{s}: not in archive, skipped")
                continue
            print(f"{s}: extracting and caching all chest + wrist channels", flush=True)
            z.extract(member, args.data_dir)
            pkl = args.data_dir / member
            cache_subject(pkl, cache)
            if not args.keep_pkl:
                pkl.unlink()
    shutil.rmtree(args.data_dir / "extract", ignore_errors=True)
    print(f"Done. Cache in {out_dir / 'cache'}")


if __name__ == "__main__":
    main()
