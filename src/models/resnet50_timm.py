from re import T
from typing import Any, List, Tuple, Optional

import timm
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from composer import functional as cf
from timm.scheduler.cosine_lr import CosineLRScheduler
from omegaconf import OmegaConf
import torch_pruning as tp

class ResNet(LightningModule):

    def __init__(
        self,
        lr: float = 0.001,
        warmup: int = 5,
        weight_decay: float = 0.0005,
        model: str = 'resnet34',
        num_classes: int = 1000,
        in_place_prune: Optional[bool] = None,
        in_place_prune_ratio: Optional[float] = None
    )->None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes

        self.model = timm.create_model(model, num_classes=self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc = Accuracy()
        
        prune_list = []
        for name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                if ('conv1' in name or "conv2" in name) and 'layer' in name:
                    prune_list.append((layer, "weight"))

        self.prune_list = prune_list

        resolver_name = "model"
        OmegaConf.register_new_resolver(
            resolver_name,
            lambda name: getattr(self, name),
            use_cache=False
        )

        if in_place_prune:
            self.prune_model(self.model, self.prune_list, in_place_prune_ratio)

    def prune_model(self, model, prune_list, amount=0.2):
        model.cpu()
        DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 224, 224) )
        def prune_conv(conv, amount=0.2):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
            strategy = tp.strategy.L1Strategy()
            pruning_index = strategy(conv.weight, amount=amount, round_to=2)
            plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
            plan.exec()
    
        block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
        blk_id = 0
        for m, n in prune_list:
            prune_conv(m,amount)
        
        return model    
    
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
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

        # if self.current_epoch in self.rescale_values:
        #     self.current_scale = self.rescale_values[self.current_epoch]

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
