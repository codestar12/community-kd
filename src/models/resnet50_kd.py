from re import T
from typing import Any, List, Tuple

import timm
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from .modules.kd_loss import DistillKL
from collections import OrderedDict
from timm.scheduler.cosine_lr import CosineLRScheduler


class ResNet(LightningModule):

    def __init__(
        self,
        teacher_checkpoint: str,
        lr: float = 0.001,
        warmup: int = 5,
        weight_decay: float = 0.0005,
        num_classes: int = 1000,
        teacher_model: str = 'resnet34',
        student_model: str = 'resnet18',
        hard_weight: float = 0.1,
        soft_weight: float = 0.9,
    )->None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes

        self.teacher = timm.create_model('resnet34', num_classes=self.num_classes)
        checkpoint = torch.load(teacher_checkpoint)
        corrected_state_dict = self.fix_checkpoint(checkpoint['state_dict'])
        self.teacher.load_state_dict(corrected_state_dict)
        self.teacher.eval()

        self.student = timm.create_model('resnet18', num_classes=self.num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.kl_loss = DistillKL()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight

        self.train_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc = Accuracy()

    def fix_checkpoint(self, old_state_dict):
    
        new_state_dict = OrderedDict()
    
        for key, value in old_state_dict.items():
            new_state_dict[key.replace('model.', '')] = value

        return new_state_dict


    def forward(self, x: torch.Tensor):

        with torch.no_grad():
            logits_t = self.teacher(x)

        logits_s = self.student(x)

        return logits_t, logits_s

    def step(self, batch: Any):
        x, y = batch
        logits_t, logits_s = self.forward(x)
        hard_loss = self.criterion(logits_s, y)
        soft_loss = self.kl_loss(logits_s, logits_t.detach())
        preds = torch.argmax(logits_s, dim=1)
        loss = self.soft_weight * soft_loss + self.hard_weight * hard_loss 
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    # def test_epoch_end(self, outputs: List[Any]):
    #     pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_acc.reset()
        #self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer =  torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        sched = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": "val/acc"},
        }
    
    def lr_scheduler_step(self, scheduler, optimizer_idx: int, metric) -> None:
        return scheduler.step(epoch=self.current_epoch)
