#!/bin/bash
# ============================================================================
# Quick Setup Script for Task 2
# Run this script when you have network access to download all required files.
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Task 2: Quick Setup"
echo "========================================"

# Step 1: Download YOLOv8s pretrained model
echo ""
echo "Step 1: Downloading YOLOv8s pretrained model..."
if [ -f "yolov8s.pt" ]; then
    echo "  ✅ yolov8s.pt already exists."
else
    python3 -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); print('  ✅ yolov8s.pt downloaded')"
fi

# Step 2: Download dataset from Kaggle
echo ""
echo "Step 2: Downloading Road Vehicle Images Dataset..."
if [ -d "data/road_vehicle/train" ]; then
    echo "  ✅ Dataset already exists."
else
    if [ -f ~/.kaggle/kaggle.json ]; then
        mkdir -p data/road_vehicle
        kaggle datasets download -d ashfakyeafi/road-vehicle-images-dataset -p data/road_vehicle --unzip
        echo "  ✅ Dataset downloaded."
    else
        echo "  ⚠️  Kaggle API not configured."
        echo "     Please download manually from:"
        echo "     https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset"
        echo "     Extract to: $SCRIPT_DIR/data/road_vehicle/"
    fi
fi

# Step 3: Verify dataset and create data.yaml
echo ""
echo "Step 3: Verifying dataset..."
if [ -d "data/road_vehicle" ]; then
    python3 download_dataset.py
fi

# Step 4: Check test video
echo ""
echo "Step 4: Checking test video..."
if [ -f "data/test_video/test.mp4" ]; then
    SIZE=$(stat --format=%s "data/test_video/test.mp4" 2>/dev/null || stat -f%z "data/test_video/test.mp4" 2>/dev/null)
    if [ "$SIZE" -gt 100000 ]; then
        echo "  ✅ Test video exists ($SIZE bytes)."
    else
        echo "  ⚠️  Test video too small. Please provide a real traffic video."
    fi
else
    echo "  ⚠️  No test video found."
    echo "     Please place a 10-30s traffic video at:"
    echo "     $SCRIPT_DIR/data/test_video/test.mp4"
fi

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. python train.py --epochs 100 --batch 32"
echo "  2. python inference_pipeline.py --source data/test_video/test.mp4"
echo ""
