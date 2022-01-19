from pl_bolts.datamodules import ImagenetDataModule
from typing import Any, Callable, Optional
from composer.algorithms.randaugment import RandAugmentTransform
from torchvision import transforms as transform_lib
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization

class ImagenetMosaicAugmented(ImagenetDataModule):
    def __init__(self, data_dir: str, meta_dir: Optional[str] = None, num_imgs_per_val_class: int = 50, image_size: int = 224, num_workers: int = 0, batch_size: int = 32, shuffle: bool = True, pin_memory: bool = True, drop_last: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(data_dir, meta_dir, num_imgs_per_val_class, image_size, num_workers, batch_size, shuffle, pin_memory, drop_last, *args, **kwargs)

    def train_transform(self) -> Callable:
        preprocessing = transform_lib.Compose(
            [
                RandAugmentTransform(),
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing
