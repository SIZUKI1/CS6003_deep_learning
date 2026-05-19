"""
YOLOv8s Fine-tuning on Road Vehicle Images Dataset.

Training script with SwanLab integration for experiment tracking.
Logs: loss curves, mAP metrics, learning rate, etc.

Usage:
    python train.py [--data DATA_YAML] [--epochs EPOCHS] [--batch BATCH] [--device DEVICE]
"""

import argparse
import os
import sys
from pathlib import Path

# ── SwanLab callback for Ultralytics ──────────────────────────────────────
# We implement a lightweight SwanLab callback that hooks into
# ultralytics' training loop to log metrics at every epoch.

import swanlab


def setup_swanlab_callback(model):
    """Register SwanLab callbacks with the YOLO model."""

    def on_train_start(trainer):
        """Initialize SwanLab run when training starts."""
        hyp = trainer.args
        config = {
            "model": str(hyp.model),
            "data": str(hyp.data),
            "epochs": hyp.epochs,
            "batch_size": hyp.batch,
            "imgsz": hyp.imgsz,
            "optimizer": hyp.optimizer,
            "lr0": hyp.lr0,
            "lrf": hyp.lrf,
            "momentum": hyp.momentum,
            "weight_decay": hyp.weight_decay,
            "patience": hyp.patience,
            "device": str(hyp.device),
        }
        swanlab.init(
            project="yolov8-road-vehicle",
            experiment_name=f"yolov8s-finetune-ep{hyp.epochs}-bs{hyp.batch}",
            config=config,
        )
        print("✅ SwanLab initialized")

    def on_train_epoch_end(trainer):
        """Log training metrics at each epoch end."""
        epoch = trainer.epoch
        metrics = trainer.label_loss_items(trainer.tloss, prefix="train")
        metrics["train/lr"] = trainer.optimizer.param_groups[0]["lr"]
        metrics["epoch"] = epoch
        swanlab.log(metrics, step=epoch)

    def on_val_end(validator):
        """Log validation metrics."""
        # This is called from the trainer's validation step
        pass

    def on_fit_epoch_end(trainer):
        """Log combined train + val metrics at each epoch end."""
        epoch = trainer.epoch
        metrics = {}

        # Validation metrics
        if hasattr(trainer, "metrics") and trainer.metrics:
            m = trainer.metrics
            if hasattr(m, "results_dict"):
                for k, v in m.results_dict.items():
                    metrics[f"val/{k}"] = v

        # Losses from validator
        if hasattr(trainer, "validator") and trainer.validator and hasattr(trainer.validator, "loss"):
            val_loss = trainer.validator.loss
            if val_loss is not None:
                loss_names = ["box_loss", "cls_loss", "dfl_loss"]
                for i, name in enumerate(loss_names):
                    if i < len(val_loss):
                        metrics[f"val/{name}"] = float(val_loss[i])

        if metrics:
            swanlab.log(metrics, step=epoch)

    def on_train_end(trainer):
        """Finish SwanLab run."""
        # Log final best metrics
        if hasattr(trainer, "best_fitness"):
            swanlab.log({"best_fitness": trainer.best_fitness})
        swanlab.finish()
        print("✅ SwanLab run finished")

    # Register callbacks
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8s on Road Vehicle Dataset")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml config file")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Pretrained model to start from (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (default: 0)")
    parser.add_argument("--optimizer", type=str, default="SGD",
                        help="Optimizer: SGD, Adam, AdamW (default: SGD)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (default: 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate factor (default: 0.01)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--no-swanlab", action="store_true",
                        help="Disable SwanLab logging")
    args = parser.parse_args()

    # Auto-discover data.yaml
    base_dir = Path(__file__).parent
    if args.data is None:
        candidates = [
            base_dir / "data" / "road_vehicle" / "data.yaml",
            base_dir / "data.yaml",
        ]
        # Also search recursively
        for yaml_file in (base_dir / "data").rglob("data.yaml"):
            candidates.append(yaml_file)
        
        for c in candidates:
            if c.exists():
                args.data = str(c)
                break
        
        if args.data is None:
            print("❌ Could not find data.yaml! Please specify with --data")
            print("   Run download_dataset.py first to prepare the dataset.")
            sys.exit(1)

    print(f"📁 Data config: {args.data}")
    print(f"🏋️ Model: {args.model}")
    print(f"📊 Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    print(f"🔧 Optimizer: {args.optimizer}, LR: {args.lr0} → {args.lr0 * args.lrf}")
    print(f"🎮 Device: {args.device}")

    # ── Load model ──
    from ultralytics import YOLO

    if args.resume:
        # Resume from last checkpoint
        last_pt = base_dir / "runs" / "detect" / "yolov8s_road_vehicle" / "weights" / "last.pt"
        if last_pt.exists():
            model = YOLO(str(last_pt))
            print(f"▶️ Resuming from {last_pt}")
        else:
            print(f"❌ No checkpoint found at {last_pt}")
            sys.exit(1)
    else:
        model = YOLO(args.model)
        print(f"▶️ Loaded pretrained model: {args.model}")

    # ── Setup SwanLab ──
    if not args.no_swanlab:
        try:
            setup_swanlab_callback(model)
            print("📈 SwanLab logging enabled")
        except Exception as e:
            print(f"⚠️ SwanLab setup failed: {e}")
            print("   Training will continue without SwanLab.")

    # ── Train ──
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        patience=args.patience,
        device=args.device,
        project=str(base_dir / "runs" / "detect"),
        name="yolov8s_road_vehicle",
        exist_ok=True,
        pretrained=True,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Save settings
        save=True,
        save_period=-1,  # Save best only
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"📁 Results saved to: {base_dir / 'runs' / 'detect' / 'yolov8s_road_vehicle'}")
    print(f"🏆 Best model: {base_dir / 'runs' / 'detect' / 'yolov8s_road_vehicle' / 'weights' / 'best.pt'}")
    print("=" * 60)

    # ── Validate ──
    print("\n🔍 Running final validation...")
    best_model = YOLO(str(base_dir / "runs" / "detect" / "yolov8s_road_vehicle" / "weights" / "best.pt"))
    val_results = best_model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    
    print(f"\n📊 Final Validation Results:")
    print(f"   mAP@0.5:      {val_results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {val_results.box.map:.4f}")


if __name__ == "__main__":
    main()
