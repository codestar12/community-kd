import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 6,
        train_size: int = 32,
        data_dir: str = "./data/",
        pin_memory: bool = False):
        super().__init__()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.train_size = train_size
        self.data_dir = data_dir
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        if train_size == 32:
            self.train_transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
            
            self.test_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
        else:
            self.train_transform = transforms.Compose([
                                                transforms.Resize(self.train_size),
                                                transforms.RandomCrop(self.train_size, padding=int(self.train_size * 0.125)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
            
            self.test_transform = transforms.Compose([
                                                transforms.Resize(self.train_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])


    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir + 'cifar10',train=True,download=True, transform=self.train_transform)
        datasets.CIFAR10(root=self.data_dir + 'cifar10',train=False,download=True, transform=self.test_transform)

    def setup(self, stage):
        cifar_train = datasets.CIFAR10(root=self.data_dir + 'cifar10',train=True,download=True, transform=self.train_transform)
        self.cifar_test = datasets.CIFAR10(root=self.data_dir + 'cifar10',train=False,download=True, transform=self.test_transform)
        self.cifar_train = cifar_train

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=self.pin_memory)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=self.pin_memory)
        return cifar_val

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=self.pin_memory)



class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 6,
        train_size: int = 32,
        data_dir: str = "./data/",
        pin_memory: bool = False):
        super().__init__()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.train_size = train_size
        self.data_dir = data_dir
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        if train_size == 32:
            self.train_transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
            
            self.test_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
        else:
            self.train_transform = transforms.Compose([
                                                transforms.Resize(self.train_size),
                                                 transforms.RandomCrop(self.train_size, padding=int(self.train_size * 0.125)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
            
            self.test_transform = transforms.Compose([
                                                transforms.Resize(self.train_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])


    def prepare_data(self):
        datasets.CIFAR100(root=self.data_dir + 'cifar100',train=True,download=True, transform=self.train_transform)
        datasets.CIFAR100(root=self.data_dir + 'cifar100',train=False,download=True, transform=self.test_transform)

    def setup(self, stage):
        cifar_train = datasets.CIFAR100(root=self.data_dir + 'cifar100',train=True,download=True, transform=self.train_transform)
        self.cifar_test = datasets.CIFAR100(root=self.data_dir + 'cifar100',train=False,download=True, transform=self.test_transform)
        self.cifar_train = cifar_train

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=self.pin_memory)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=self.pin_memory)
        return cifar_val

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=self.pin_memory)