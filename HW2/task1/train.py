#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import Flower102Dataset, build_transforms
from src.engine import evaluate, train_one_epoch
from src.models import build_model, split_backbone_head_params
from src.utils import ExperimentLogger, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ImageNet-pretrained ResNet on Oxford Flowers 102")

    parser.add_argument("--data_root", type=str, default="./data/flowers102")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "se_resnet18", "cbam_resnet18"])
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="Use ImageNet pretrained weights. Use --no-pretrained for ablation.")
    parser.add_argument("--num_classes", type=int, default=102)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--logger", type=str, default="none", choices=["none", "wandb", "swanlab"])
    parser.add_argument("--project", type=str, default="flower102-finetune")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name
    if run_name is None:
        pre = "imagenet" if args.pretrained else "random"
        run_name = f"{args.model}_{pre}_ep{args.epochs}_bb{args.lr_backbone:g}_head{args.lr_head:g}"

    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device"] = str(device)
    config["run_name"] = run_name
    save_json(config, out_dir / "config.json")

    train_set = Flower102Dataset(
        args.data_root, split="train",
        transform=build_transforms("train", img_size=args.img_size),
    )
    val_set = Flower102Dataset(
        args.data_root, split="val",
        transform=build_transforms("val", img_size=args.img_size),
    )
    test_set = Flower102Dataset(
        args.data_root, split="test",
        transform=build_transforms("test", img_size=args.img_size),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained).to(device)

    backbone_params, head_params = split_backbone_head_params(model)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    logger = ExperimentLogger(args.logger, args.project, run_name, config)

    metrics_csv = out_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc", "val_mAP",
                "lr_backbone", "lr_head",
            ],
        )
        writer.writeheader()

    best_val_acc = -1.0
    best_epoch = -1

    print(f"Device: {device}")
    print(f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"Output dir: {out_dir}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp=args.amp)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes=args.num_classes, desc="val")
        scheduler.step()

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_mAP": val_metrics["mAP"],
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
        }

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        logger.log({
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/acc": train_metrics["acc"],
            "val/loss": val_metrics["loss"],
            "val/acc": val_metrics["acc"],
            "val/mAP": val_metrics["mAP"],
            "lr/backbone": lr_backbone,
            "lr/head": lr_head,
        }, step=epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_mAP={val_metrics['mAP']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": config,
                "best_val_acc": best_val_acc,
                "val_metrics": val_metrics,
            }, out_dir / "best.pt")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "args": config,
    }, out_dir / "last.pt")

    # Test using the best checkpoint.
    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes=args.num_classes, desc="test")
    save_json({
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_mAP": test_metrics["mAP"],
    }, out_dir / "test_metrics.json")

    logger.log({
        "test/loss": test_metrics["loss"],
        "test/acc": test_metrics["acc"],
        "test/mAP": test_metrics["mAP"],
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }, step=args.epochs + 1)
    logger.finish()

    print("\nFinal result:")
    print(f"  best_epoch  = {best_epoch}")
    print(f"  best_val_acc= {best_val_acc:.4f}")
    print(f"  test_acc    = {test_metrics['acc']:.4f}")
    print(f"  test_mAP    = {test_metrics['mAP']:.4f}")
    print(f"  saved to    = {out_dir}")


if __name__ == "__main__":
    main()
