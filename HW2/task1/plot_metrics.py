#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_one(df: pd.DataFrame, y_cols, title: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(7, 4))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[save] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="e.g. outputs/resnet18_imagenet_ep30_bb0.0001_head0.001")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "metrics.csv")

    plot_one(df, ["train_loss", "val_loss"], "Training and Validation Loss", "Loss", run_dir / "loss_curve.png")
    plot_one(df, ["val_acc", "val_mAP"], "Validation Accuracy / mAP", "Score", run_dir / "val_acc_map_curve.png")


if __name__ == "__main__":
    main()
