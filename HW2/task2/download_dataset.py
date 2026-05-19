"""
Download and prepare the Road Vehicle Images Dataset for YOLOv8 training.
Dataset source: https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset

This script supports two download methods:
1. Kaggle API (if configured)
2. Manual download - provides instructions

After download, it verifies the dataset structure and creates a data.yaml config.
"""

import os
import sys
import yaml
import glob
import shutil
from pathlib import Path


def check_kaggle_api():
    """Check if Kaggle API is configured."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    return os.path.exists(kaggle_json)


def download_with_kaggle(dest_dir):
    """Download dataset using Kaggle API."""
    os.makedirs(dest_dir, exist_ok=True)
    cmd = f"kaggle datasets download -d ashfakyeafi/road-vehicle-images-dataset -p {dest_dir} --unzip"
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print("Kaggle download failed!")
        return False
    return True


def discover_classes(labels_dir):
    """Discover all class IDs from label files and return sorted list."""
    class_ids = set()
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for lf in label_files:
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_ids.add(int(parts[0]))
    return sorted(class_ids)


def verify_and_prepare(data_root):
    """Verify dataset structure and create data.yaml."""
    data_root = Path(data_root)
    
    # Try to find the actual data directories
    # The dataset might have a nested structure after extraction
    possible_roots = [
        data_root,
        data_root / "Road Vehicle Images Dataset",
        data_root / "road-vehicle-images-dataset",
    ]
    
    actual_root = None
    for pr in possible_roots:
        # Check for train and valid directories
        has_train = (pr / "train" / "images").exists() or (pr / "train").exists()
        has_valid = (pr / "valid" / "images").exists() or (pr / "val" / "images").exists()
        if has_train and has_valid:
            actual_root = pr
            break
    
    if actual_root is None:
        # List what we have
        print(f"Contents of {data_root}:")
        for item in sorted(data_root.rglob("*")):
            if item.is_dir():
                print(f"  [DIR]  {item.relative_to(data_root)}")
            elif str(item).endswith(('.jpg', '.png', '.txt', '.yaml')):
                print(f"  [FILE] {item.relative_to(data_root)}")
        print("\nCould not find expected train/valid structure. Please check manually.")
        return False
    
    print(f"Found dataset root: {actual_root}")
    
    # Determine the split directories
    train_img_dir = None
    val_img_dir = None
    train_lbl_dir = None
    val_lbl_dir = None
    
    for img_dir_name in ["images", ""]:
        for train_name in ["train", "training"]:
            candidate = actual_root / train_name / img_dir_name if img_dir_name else actual_root / train_name
            if candidate.exists() and any(candidate.glob("*.jpg")) or any(candidate.glob("*.png")):
                train_img_dir = candidate
                break
        for val_name in ["valid", "val", "validation", "test"]:
            candidate = actual_root / val_name / img_dir_name if img_dir_name else actual_root / val_name
            if candidate.exists() and (any(candidate.glob("*.jpg")) or any(candidate.glob("*.png"))):
                val_img_dir = candidate
                break
    
    if train_img_dir is None:
        # Try deeper search
        for d in actual_root.rglob("images"):
            parent = d.parent
            if "train" in str(parent).lower():
                train_img_dir = d
            elif "val" in str(parent).lower() or "valid" in str(parent).lower():
                val_img_dir = d
    
    if train_img_dir is None:
        print("Could not find training images directory!")
        return False
    
    # Find corresponding label directories
    train_lbl_dir = train_img_dir.parent / "labels"
    if not train_lbl_dir.exists():
        train_lbl_dir = Path(str(train_img_dir).replace("images", "labels"))
    
    if val_img_dir:
        val_lbl_dir = val_img_dir.parent / "labels"
        if not val_lbl_dir.exists():
            val_lbl_dir = Path(str(val_img_dir).replace("images", "labels"))
    
    # Count files
    train_imgs = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    train_lbls = list(train_lbl_dir.glob("*.txt")) if train_lbl_dir and train_lbl_dir.exists() else []
    
    print(f"\nTrain images: {len(train_imgs)} in {train_img_dir}")
    print(f"Train labels: {len(train_lbls)} in {train_lbl_dir}")
    
    if val_img_dir:
        val_imgs = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
        val_lbls = list(val_lbl_dir.glob("*.txt")) if val_lbl_dir and val_lbl_dir.exists() else []
        print(f"Valid images: {len(val_imgs)} in {val_img_dir}")
        print(f"Valid labels: {len(val_lbls)} in {val_lbl_dir}")
    
    # Discover classes from labels
    all_label_dirs = [d for d in [train_lbl_dir, val_lbl_dir] if d and d.exists()]
    all_classes = set()
    for ld in all_label_dirs:
        all_classes.update(discover_classes(str(ld)))
    
    print(f"\nDiscovered class IDs: {sorted(all_classes)}")
    print(f"Number of classes: {len(all_classes)}")
    
    # Check for existing data.yaml
    existing_yaml = None
    for yaml_file in actual_root.glob("*.yaml"):
        existing_yaml = yaml_file
        break
    
    if existing_yaml:
        print(f"\nFound existing config: {existing_yaml}")
        with open(existing_yaml, 'r') as f:
            existing_config = yaml.safe_load(f)
        print(f"Existing config: {existing_config}")
        class_names = existing_config.get('names', {})
    else:
        # Default class names for this dataset (common Bangladeshi road vehicles)
        # These will be verified against the actual labels
        default_names = {
            0: "ambulance",
            1: "auto rickshaw",
            2: "bicycle",
            3: "bus",
            4: "car",
            5: "garbagevan",
            6: "human hauler",
            7: "minibus",
            8: "motorbike",
            9: "pickup",
            10: "army vehicle",
            11: "policecar",
            12: "rickshaw",
            13: "scooter",
            14: "suv",
            15: "taxi",
            16: "three wheelers",
            17: "truck",
            18: "van",
            19: "wheelbarrow",
        }
        class_names = {k: default_names.get(k, f"class_{k}") for k in sorted(all_classes)}
    
    # Create data.yaml in the dataset root
    # Use relative paths from the yaml file location
    data_yaml_path = actual_root / "data.yaml"
    
    data_config = {
        "path": str(actual_root.resolve()),
        "train": str(train_img_dir.relative_to(actual_root)),
        "val": str(val_img_dir.relative_to(actual_root)) if val_img_dir else str(train_img_dir.relative_to(actual_root)),
        "nc": len(all_classes),
        "names": class_names if isinstance(class_names, dict) else {i: n for i, n in enumerate(class_names)} if isinstance(class_names, list) else class_names,
    }
    
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\nCreated data.yaml at: {data_yaml_path}")
    print(f"Config:\n{yaml.dump(data_config, default_flow_style=False, allow_unicode=True)}")
    
    return str(data_yaml_path)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "road_vehicle")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if check_kaggle_api():
        print("Kaggle API found. Downloading dataset...")
        success = download_with_kaggle(data_dir)
        if not success:
            sys.exit(1)
    else:
        print("=" * 60)
        print("Kaggle API not configured.")
        print("Please download the dataset manually:")
        print()
        print("Option 1: Configure Kaggle API")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' to download kaggle.json")
        print("  3. Place it at ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  5. Re-run this script")
        print()
        print("Option 2: Manual download")
        print("  1. Download from: https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset")
        print(f"  2. Extract to: {data_dir}")
        print("  3. Re-run this script to verify and create data.yaml")
        print("=" * 60)
        
        # Check if data already exists
        if any(Path(data_dir).rglob("*.jpg")) or any(Path(data_dir).rglob("*.png")):
            print("\nFound existing image files. Proceeding with verification...")
        else:
            sys.exit(1)
    
    # Verify and prepare
    yaml_path = verify_and_prepare(data_dir)
    if yaml_path:
        print(f"\n✅ Dataset ready! data.yaml: {yaml_path}")
    else:
        print("\n❌ Dataset verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
