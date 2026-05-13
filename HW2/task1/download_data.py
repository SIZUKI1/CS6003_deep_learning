#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and prepare Oxford 102 Category Flower Dataset.

Required files:
  - 102flowers.tgz
  - imagelabels.mat
  - setid.mat

The segmentation masks and chi2 distances are not needed for this image
classification assignment.
"""
from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path


BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
FILES = {
    "102flowers.tgz": BASE_URL + "102flowers.tgz",
    "imagelabels.mat": BASE_URL + "imagelabels.mat",
    "setid.mat": BASE_URL + "setid.mat",
}


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] {out_path} already exists")
        return

    print(f"[download] {url}")
    print(f"           -> {out_path}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(100.0, downloaded * 100.0 / total_size)
        print(f"\r  {percent:6.2f}%", end="")

    urllib.request.urlretrieve(url, tmp_path, reporthook=progress)
    print()
    tmp_path.rename(out_path)


def extract_images(tgz_path: Path, data_root: Path) -> None:
    jpg_dir = data_root / "jpg"
    if jpg_dir.exists() and any(jpg_dir.glob("*.jpg")):
        print(f"[skip] {jpg_dir} already exists")
        return

    print(f"[extract] {tgz_path}")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_root)
    print(f"[done] images extracted to {jpg_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/flowers102")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        download_file(url, data_root / filename)

    extract_images(data_root / "102flowers.tgz", data_root)

    print("\nPrepared files:")
    print(f"  {data_root / 'jpg'}")
    print(f"  {data_root / 'imagelabels.mat'}")
    print(f"  {data_root / 'setid.mat'}")


if __name__ == "__main__":
    main()
