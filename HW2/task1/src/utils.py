# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep benchmark=True for speed on fixed-size images.
    torch.backends.cudnn.benchmark = True


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class ExperimentLogger:
    """
    Thin wrapper for no logger / wandb / swanlab.

    Use:
      logger = ExperimentLogger("wandb", project, run_name, config)
      logger.log({"train/loss": 1.0}, step=1)
      logger.finish()
    """
    def __init__(self, logger_type: str, project: str, run_name: str, config: dict):
        self.logger_type = logger_type.lower()
        self.run = None

        if self.logger_type == "none":
            return

        if self.logger_type == "wandb":
            import wandb
            self.run = wandb.init(project=project, name=run_name, config=config)
            self.backend = wandb
            return

        if self.logger_type == "swanlab":
            import swanlab
            self.run = swanlab.init(project=project, experiment_name=run_name, config=config)
            self.backend = swanlab
            return

        raise ValueError("logger must be one of: none, wandb, swanlab")

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.logger_type == "none":
            return
        if self.logger_type == "wandb":
            self.backend.log(metrics, step=step)
        elif self.logger_type == "swanlab":
            self.backend.log(metrics, step=step)

    def finish(self) -> None:
        if self.logger_type == "none":
            return
        if self.logger_type == "wandb":
            self.backend.finish()
        elif self.logger_type == "swanlab":
            self.backend.finish()
