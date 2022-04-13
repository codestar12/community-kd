from os import sched_getscheduler
from re import I, T
from typing import Any, List, Tuple, Union

import timm
import torch
import numpy as np

import torch.nn as nn

from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from .modules.hydra import Hydra29, Hydra50, Hydra41, Hydra35
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch_optimizer import Lamb


from .modules.kd_loss import DistillKL


def return_kd_weight(delay, trans, epoch, weight):

    rates = np.linspace(0.0, weight, num=trans)
    if epoch < delay:
        return 0
    elif epoch - delay < trans:
        return rates[epoch - delay]
    else:
        return weight


def return_hard_weight(delay, trans, epoch, weight_end, weight_start):
    rates = np.linspace(weight_start, weight_end, num=trans)
    if epoch < delay:
        return weight_start
    elif epoch - delay < trans:
        return rates[epoch - delay]
    else:
        return weight_end


class CommKD(LightningModule):
    def __init__(
        self,
        num_students: int,
        kd_weights: List[List[float]],
        backbone: str = "Hydra50",
        kd_trans_epochs: Union[int, List[int]] = 0,
        kd_delay: Union[int, List[int]] = 0,
        hard_label_start: Union[float, List[float]] = 0.1,
        hard_label_end: Union[float, List[float]] = 0.1,
        lr: float = 0.001,
        student_layers: List[List[int]] = [],
        weight_decay: float = 0.0005,
        num_classes: int = 1000,
        lr_milestones: List[int] = [50, 80],
        periods: int = 1,
        warmup: int = 5,
    ) -> None:

        super().__init__()
        self.num_students: int = num_students
        self.num_classes = num_classes
        self.periods = periods
        self.warmup = warmup

        if backbone == "Hydra50":
            self.model = Hydra50(
                num_classes,
                channels=3,
                student_layers=student_layers,
                num_heads=num_students + 1,
            )
        elif backbone == "Hyrdra41":
            self.model = Hydra41(
                num_classes,
                channels=3,
                student_layers=student_layers,
                num_heads=num_students + 1,
            )
        elif backbone == "Hydra35":
            self.model = Hydra35(
                num_classes,
                channels=3,
                student_layers=student_layers,
                num_heads=num_students + 1,
            )
        else:
            self.model = Hydra29(
                num_classes,
                channels=3,
                student_layers=student_layers,
                num_heads=num_students + 1,
            )

        self.criterion = nn.CrossEntropyLoss()
        self.kd_weights = kd_weights
        self.kd_loss = DistillKL()

        self.hard_label_start = hard_label_start
        self.hard_label_end = hard_label_end
        self.kd_delay = kd_delay
        self.kd_trans_epochs = kd_trans_epochs

        self.student_hard_label = self.hard_label_start

        self.train_acc = nn.ModuleList(
            [Accuracy() for i in range(self.num_students + 1)]
        )
        self.val_acc = nn.ModuleList([Accuracy() for i in range(self.num_students + 1)])
        self.test_acc = nn.ModuleList(
            [Accuracy() for i in range(self.num_students + 1)]
        )

        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor):

        return self.model(x)

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
                if isinstance(self.student_hard_label, float):
                    model_loss.append(hard_loss * self.student_hard_label)
                else:
                    model_loss.append(hard_loss * self.student_hard_label[i - 1])

            for j, weight in enumerate(weights):
                s_logits = outputs[j]
                soft_loss = 0.0
                if j != i:
                    soft_loss = self.kd_loss(logits, s_logits.detach())

                if isinstance(self.kd_delay, int):
                    final_weight = return_kd_weight(
                        self.kd_delay, self.kd_trans_epochs, self.current_epoch, weight
                    )
                else:
                    final_weight = return_kd_weight(
                        self.kd_delay[i],
                        self.kd_trans_epochs,
                        self.current_epoch,
                        weight,
                    )

                model_loss.append(soft_loss * final_weight)

            losses.append(sum(model_loss))

        return losses, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        losses, preds, targets = self.step(batch)

        # log train metrics
        for i, pred in enumerate(preds):

            acc = self.train_acc[i](pred, targets)

            if i == 0:
                self.log(
                    "teacher/train/loss",
                    losses[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "teacher/train/acc",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            else:
                self.log(
                    f"student_{i}/train/loss",
                    losses[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"student_{i}/train/acc",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

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
                self.log(
                    "teacher/val/loss",
                    losses[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "teacher/val/acc", acc, on_step=False, on_epoch=True, prog_bar=True
                )
            else:
                self.log(
                    f"student_{i}/val/loss",
                    losses[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"student_{i}/val/acc",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        loss = sum(losses)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        losses, preds, targets = self.step(batch)

        # log val metrics
        for i, pred in enumerate(preds):

            acc = self.test_acc[i](pred, targets)

            if i == 0:
                self.log(
                    "teacher/test/loss",
                    losses[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "teacher/test/acc", acc, on_step=False, on_epoch=True, prog_bar=True
                )
            else:
                self.log(
                    f"student_{i}/test/loss",
                    losses[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    f"student_{i}/test/acc",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        loss = sum(losses)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_start(self):
        if isinstance(self.kd_delay, int):
            self.student_hard_label = return_hard_weight(
                self.kd_delay,
                self.kd_trans_epochs,
                self.current_epoch,
                self.hard_label_end,
                self.hard_label_start,
            )
        else:
            for i in range(self.num_students + 1):
                self.student_hard_label = return_hard_weight(
                    self.kd_delay[i],
                    self.kd_trans_epochs,
                    self.current_epoch,
                    self.hard_label_end,
                    self.hard_label_start,
                )

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
        optimizer = Lamb(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        sched = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=self.trainer.max_epochs / self.periods,
            warmup_t=self.warmup,
            warmup_lr_init=1e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": "teacher/val/acc"},
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx: int, metric) -> None:
        return scheduler.step(epoch=self.current_epoch)
