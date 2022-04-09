from re import T
from typing import Any, List, Tuple

import timm
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
import bitsandbytes
from composer.optim import scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from composer.optim.pytorch_future import WarmUpLR
from .modules.at_loss import Attention
from .modules.model_wrapper import ModelWrapper
from .modules.wrapped_resnet import WrappedResnet
from .modules.kd_loss import DistillKL

from .torchdistill.densenet import densenet_bc_k12_depth100
from .torchdistill.resnet import resnet20
from .torchdistill.custom_loss import GeneralizedCustomLoss

class ResNet(LightningModule):

    def __init__(
        self,
        lr: float = 0.1,
        weight_decay: float = 0.0001,
        num_classes: int = 10,
    )->None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.teacher_model = densenet_bc_k12_depth100()
        self.teacher_model.load_state_dict(torch.load('/home/nathaniel/community-kd/src/models/torchdistill/checkpoints/cifar10-densenet_bc_k12_depth100.pt'))
        self.teacher_model.train(False)

        self.student_model = resnet20()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.distill_loss = DistillKL()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    
    def forward(self, x: torch.Tensor):
        logit_s = self.student_model(x)
        logit_t = self.teacher_model(x)
        return logit_s, logit_t

    def step(self, batch: Any):
        x, y = batch
        #logits = self.forward(x)
        logit_s, logit_t = self.forward(x)
        hard_loss = self.criterion(logit_s, y) #loss_cls
        soft_loss = self.distill_loss(logit_s, logit_t) #loss_kd

        preds = torch.argmax(logit_s, dim=1)
        loss = 1 * hard_loss +  0.9 * soft_loss
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

    # def test_step(self, batch: Any, batch_idx: int):
    #     loss, preds, targets = self.step(batch)

    #     # log test metrics
    #     acc = self.test_acc(preds, targets)
    #     self.log("test/loss", loss, on_step=False, on_epoch=True)
    #     self.log("test/acc", acc, on_step=False, on_epoch=True)

    #     return {"loss": loss, "preds": preds, "targets": targets}

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
        optimizer = torch.optim.SGD(
            params=self.student_model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9
        )

        lr_milestones: List[int] = [91, 136]
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_milestones, gamma=0.1)

        return (
            {
                "optimizer" : optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "teacher/val/acc"
                }
            }
        )
