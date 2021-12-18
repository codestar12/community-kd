from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader

class AsyncLoader(ImagenetDataModule):
    def train_dataloader(self) -> AsynchronousLoader:
        return AsynchronousLoader(ImagenetDataModule.train_dataloader(self))

class AsyncLoader(ImagenetDataModule):
    def test_dataloader(self) -> AsynchronousLoader:
        return AsynchronousLoader(ImagenetDataModule.test_dataloader(self))

class AsyncLoader(ImagenetDataModule):
    def val_dataloader(self) -> AsynchronousLoader:
        return AsynchronousLoader(ImagenetDataModule.val_dataloader(self))