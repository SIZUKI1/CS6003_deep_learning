# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def compute_accuracy_and_map(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    acc = float((preds == targets_np).mean())

    # Multiclass mAP: one-vs-rest macro average precision over 102 classes.
    one_hot = np.eye(num_classes, dtype=np.float32)[targets_np]
    try:
        mAP = float(average_precision_score(one_hot, probs, average="macro"))
    except ValueError:
        mAP = float("nan")
    return acc, mAP


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, amp: bool = False) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_num += batch_size
        pbar.set_postfix(loss=total_loss / max(total_num, 1), acc=total_correct / max(total_num, 1))

    return {
        "loss": total_loss / total_num,
        "acc": total_correct / total_num,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int = 102, desc: str = "val") -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_num = 0
    all_logits = []
    all_targets = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_num += batch_size

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    acc, mAP = compute_accuracy_and_map(logits, targets, num_classes=num_classes)

    return {
        "loss": total_loss / total_num,
        "acc": acc,
        "mAP": mAP,
    }
