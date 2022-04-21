import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torchvision 

import numpy as np
from typing import List
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

class FfcvImagenet(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: str,
        val_dataset: str,
        batch_size: int = 64,
        workers: int = 4,
        in_memory: bool = False,
        train_size: int = 224,
        test_size: int = 224,
        **kw,
    ) -> None:
        super().__init__(self)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = workers
        self.in_memory = in_memory
        self.train_size = train_size
        self.test_size = test_size

    def train_dataloader(self,):
        this_device = f'cuda:0'
        train_path = Path(self.train_dataset)
        assert train_path.is_file()

        self.decoder = RandomResizedCropRGBImageDecoder((self.train_size, self.train_size))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
        ]

        order = OrderOption.RANDOM #if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(self.train_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        order=order,
                        os_cache=self.in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=True)

        return loader


    def val_dataloader(self,):
        #this_device = f'cuda:{self.gpu}'
        this_device = f'cuda:0'
        val_path = Path(self.val_dataset)
        assert val_path.is_file()
        res_tuple = (self.test_size, self.test_size)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
        ]

        loader = Loader(self.val_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=False)
        return loader