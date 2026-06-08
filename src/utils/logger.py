"""src/utils/logger.py — WandB + TensorBoard 듀얼 로거"""

from __future__ import annotations
import os
from typing import Any


class Logger:
    def __init__(
        self,
        run_name:   str,
        use_wandb:  bool = False,
        use_tb:     bool = True,
        log_dir:    str  = "runs",
        config:     dict | None = None,
    ):
        self.run_name = run_name
        self._wandb   = None
        self._writer  = None

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(project="attention-drlca", name=run_name,
                           config=config or {})
            except ImportError:
                print("[Logger] wandb not installed, skipping.")

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_path = os.path.join(log_dir, run_name)
                self._writer = SummaryWriter(tb_path)
            except ImportError:
                print("[Logger] tensorboard not installed, skipping.")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._wandb:
            self._wandb.log(metrics, step=step)
        if self._writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, step)

    def close(self) -> None:
        if self._wandb:
            self._wandb.finish()
        if self._writer:
            self._writer.close()
