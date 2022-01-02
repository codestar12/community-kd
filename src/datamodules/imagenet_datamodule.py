import os.path

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import webdataset as wds


def identity(x):
    return x


class ShardImagenetData(pl.LightningDataModule):
    def __init__(
        self,
        shards=None,
        valshards=None,
        batch_size=64,
        workers=4,
        bucket=None,
        pin_memory=False,
        **kw
    ) -> None:
        super().__init__(self)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.training_urls = os.path.join(bucket, shards)
        print("training_urls = ", self.training_urls)
        self.val_urls = os.path.join(bucket, valshards)
        print("val_urls = ", self.val_urls)
        self.batch_size = batch_size
        self.num_workers = workers
        self.pin_memory = pin_memory
        print("batch_size", self.batch_size, "num_workers", self.num_workers)

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

    def make_loader(self, urls: str, mode: str = "train"):

        if isinstance(urls, str) and urls.startswith("fake:"):
            xs = torch.randn((self.batch_size, 3, 224, 224))
            ys = torch.zeros(self.batch_size, dtype=torch.int64)
            return wds.MockDataset((xs, ys), 10000)

        if mode == "train":
            dataset_size = 1281167
            shuffle = 10000
        elif mode == "val":
            dataset_size = 50000
            shuffle = 0

        transform = self.make_transform(mode=mode)

        dataset = (
            wds.WebDataset(urls)
            .shuffle(shuffle)
            .decode("pil")
            .to_tuple("jpg;png;jpeg cls")
            .map_tuple(transform, identity)
            .batched(self.batch_size, partial=False)
        )

        #dataset.length = dataset // self.batch_size

        loader = wds.WebLoader(
            wds.extradatasets.FakeLength(dataset, dataset_size // self.batch_size),
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        loader.length = dataset_size // self.batch_size

        if mode == "train":
            # ensure same number of batches in all clients
            loader = loader.ddp_equalize(dataset_size // self.batch_size, with_length=True)
            print("# loader length", len(loader))

        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")

    # @staticmethod
    # def add_loader_specific_args(parser):
    #     parser.add_argument("-b", "--batch-size", type=int, default=128)
    #     parser.add_argument("--workers", type=int, default=6)
    #     parser.add_argument("--bucket", default="./shards")
    #     parser.add_argument("--shards", default="imagenet-train-{000000..001281}.tar")
    #     parser.add_argument("--valshards", default="imagenet-val-{000000..000006}.tar")
    #     return parser
