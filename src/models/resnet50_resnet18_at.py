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

class ResNet(LightningModule):

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        num_classes: int = 1000,
    )->None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.teacher_model = timm.create_model('resnet50', num_classes=self.num_classes, pretrained=True)
        self.teacher_model = WrappedResnet(self.teacher_model)
        self.teacher_model.model.train(False)

        self.student_model = timm.create_model('resnet18', num_classes=self.num_classes, pretrained=False)
        self.student_model = WrappedResnet(self.student_model)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.distill_loss = Attention()
        self.div_loss = DistillKL()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    
    def forward(self, x: torch.Tensor):
        feat_s, logit_s = self.student_model(x)
        feat_t, logit_t = self.teacher_model(x)
        feat_t = [f.detach() for f in feat_t]
        return feat_s, feat_t, logit_s, logit_t

    def step(self, batch: Any):
        x, y = batch
        #logits = self.forward(x)
        feat_s, feat_t, logit_s, logit_t = self.forward(x)
        hard_loss = self.criterion(logit_s, y) #loss_cls
        soft_loss = self.distill_loss(feat_s, feat_t) #loss_kd
        loss_div = self.div_loss(logit_s, logit_t)   #loss_div
        
        preds = torch.argmax(logit_s, dim=1)
        loss = 1 * hard_loss + 1 * loss_div + 1 * soft_loss
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
        # return torch.optim.Adam(
        #     params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        # )
        # optimizer_list = []
        # scheduler_list = []
        trainable_list = torch.nn.ModuleList([])
        trainable_list.append(self.student_model)
        
        optimizer = torch.optim.Adam(
            params=self.student_model.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler1 = CosineAnnealingLR(optimizer=optimizer, T_max=82, eta_min=0)
        
        scheduler1_config = {"scheduler": scheduler1,
                             "interval": "step"}

        # scheduler2 = WarmUpLR(optimizer=optimizer, warmup_factor=0 ,warmup_iters=8)

        # scheduler2_config = {"scheduler": scheduler2,
        #                      "interval": "step"}

        

        return optimizer
