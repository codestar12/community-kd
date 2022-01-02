from typing import Optional, Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder


class ImagenetteModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kw
    ) -> None:
        super().__init__(self)

        self.save_hyperparameters(logger=False)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    @property
    def num_classes(self) -> int:
        return 10

    def make_transform(self, mode: str = "train"):
        if mode == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        elif mode == "val":
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )

    def make_loader(self, mode: str = "train"):

        transform = self.make_transform(mode=mode)

        dataset = ImageFolder(self.hparams.data_dir + f"/{mode}/", transform=transform)

        shuffle = mode == "train"

        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle
        )

    def train_dataloader(self):
        return self.make_loader(mode="train")

    def val_dataloader(self):
        return self.make_loader(mode="val")
