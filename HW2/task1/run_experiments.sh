#!/usr/bin/env bash
set -e

# 1. Install environment:
#    pip install -r requirements.txt
#
# 2. Prepare data:
#    python download_data.py --data_root ./data/flowers102
#
# 3. Choose logger:
#    LOGGER=wandb   # or LOGGER=swanlab, or LOGGER=none
LOGGER=${LOGGER:-wandb}
DATA_ROOT=${DATA_ROOT:-./data/flowers102}
OUT_DIR=${OUT_DIR:-./outputs}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}

# Baseline: ImageNet pretrained ResNet-18, tune lr combinations.
python train.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" --model resnet18 --pretrained --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr_backbone 1e-4 --lr_head 1e-3 --logger "$LOGGER" --run_name baseline_resnet18_ep${EPOCHS}_bb1e-4_head1e-3

python train.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" --model resnet18 --pretrained --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr_backbone 3e-5 --lr_head 3e-4 --logger "$LOGGER" --run_name baseline_resnet18_ep${EPOCHS}_bb3e-5_head3e-4

python train.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" --model resnet18 --pretrained --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr_backbone 1e-5 --lr_head 1e-3 --logger "$LOGGER" --run_name baseline_resnet18_ep${EPOCHS}_bb1e-5_head1e-3

# Pretraining ablation: same architecture, random initialization.
python train.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" --model resnet18 --no-pretrained --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr_backbone 1e-3 --lr_head 1e-3 --logger "$LOGGER" --run_name ablation_resnet18_random_ep${EPOCHS}

# Attention experiments.
python train.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" --model se_resnet18 --pretrained --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr_backbone 1e-4 --lr_head 1e-3 --logger "$LOGGER" --run_name attention_se_resnet18_ep${EPOCHS}

python train.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" --model cbam_resnet18 --pretrained --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr_backbone 1e-4 --lr_head 1e-3 --logger "$LOGGER" --run_name attention_cbam_resnet18_ep${EPOCHS}
