from os import sched_getscheduler
from re import I, T
from typing import Any, List, Tuple

import timm
import torch

import torch.nn as nn

from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from .modules.kd_loss import DistillKL


class CommKD(LightningModule):

    def __init__(
        self,
        teacher_model: str,
        student_model: str,
        num_students: int,
        kd_weights: List[List[float]],
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ) -> None:

        super().__init__()
        self.num_students: int = num_students

        __community = [timm.create_model(teacher_model)] + [timm.create_model(student_model) for i in range(num_students)]

        self.community = nn.ModuleList(__community)

        self.criterion = nn.CrossEntropyLoss()
        self.kd_weights = kd_weights
        self.kd_loss = DistillKL()

        self.train_acc = nn.ModuleList(
            [Accuracy() for i in range(self.num_students + 1)]
        )
        self.val_acc = nn.ModuleList(
            [Accuracy() for i in range(self.num_students + 1)]
        )
        self.test_acc = nn.ModuleList(
            [Accuracy() for i in range(self.num_students + 1)]
        )

        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor):
        outs = []
        for m in self.community:
            outs.append(m(x))

        return torch.stack(outs)

    def step(self, batch: Any):

        x, y = batch
        outputs = self.forward(x)

        losses = []
        preds = []

        for i, weights in enumerate(self.kd_weights):
            # calcualte the loss per model in community
            model_loss = []
            logits = outputs[i]
            hard_loss = self.criterion(logits, y)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred)
            if i == 0:
                model_loss.append(hard_loss)
            else:
                model_loss.append(hard_loss * 0.1)

            for j, weight in enumerate(weights):
                s_logits = outputs[j]
                soft_loss = 0.0
                if j != i:
                    soft_loss = self.kd_loss(logits, s_logits.detach())

                model_loss.append(soft_loss * weight)

            losses.append(sum(model_loss))

        return losses, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        losses, preds, targets = self.step(batch)

        # log train metrics
        for i, pred in enumerate(preds):

            acc = self.train_acc[i](pred, targets)

            if i == 0:
                self.log("teacher/train/loss", losses[i], on_step=False, on_epoch=True, prog_bar=False)
                self.log("teacher/train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log(f"student_{i}/train/loss", losses[i], on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"student_{i}/train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        loss = sum(losses)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        losses, preds, targets = self.step(batch)

        # log val metrics
        for i, pred in enumerate(preds):

            acc = self.val_acc[i](pred, targets)

            if i == 0:
                self.log("teacher/val/loss", losses[i], on_step=False, on_epoch=True, prog_bar=False)
                self.log("teacher/val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log(f"student_{i}/val/loss", losses[i], on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"student_{i}/val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        
        loss = sum(losses)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        losses, preds, targets = self.step(batch)

        # log val metrics
        for i, pred in enumerate(preds):

            acc = self.test_acc[i](pred, targets)

            if i == 0:
                self.log("teacher/test/loss", losses[i], on_step=False, on_epoch=True, prog_bar=False)
                self.log("teacher/test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log(f"student_{i}/test/loss", losses[i], on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"student_{i}/test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        loss = sum(losses)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        for i in range(self.num_students + 1):
            self.train_acc[i].reset()
            self.test_acc[i].reset()
            self.val_acc[i].reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        sched = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=1, max_epochs=50)

        return (
            {
                "optimizer" : optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "teacher/val/acc"
                }
            }
        )


