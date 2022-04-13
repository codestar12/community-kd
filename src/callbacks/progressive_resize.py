import torch
import numpy as np
import pytorch_lightning as pl

from composer import functional as cf

from pytorch_lightning import Callback, Trainer
from typing import Any, Optional
from collections import Counter

class ProgressiveResize(Callback):
    "resize images during training"

    def __init__(self, rescale_factors=[0.5, 0.75, 1.00], rescale_schedule=[0, 100, 150]):
        self.rescale = dict(zip(rescale_schedule, rescale_factors))
        self.epoch = 0
        self.curr_factor = rescale_factors[0]
    
    def on_train_batch_start(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule, 
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0) -> None:

        X, y = batch
        batch = cf.resize_batch(X, y, scale_factor=self.curr_factor, mode="resize")

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch += 1
        self.update_factor()
    
    def update_factor(self):
        if self.epoch in self.rescale:
            print(f"changing factor from {self.curr_factor}")
            self.curr_factor = self.rescale[self.epoch]
            print(f"to {self.curr_factor}")
    
        