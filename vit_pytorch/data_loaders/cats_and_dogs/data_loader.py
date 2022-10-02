# Cats and Dogs dataset lightning data module

import os
import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import ipdb

class CatsAndDogsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        self.ds = torchvision.datasets.ImageFolder(self.data_dir)
        # only 10 percent of the data
        full = len(self.ds)
        # split
        self.train_ds, self.val_ds = random_split(
            self.ds, [int(0.9 * full), int(0.1 * full)]
        )
        

    def setup(self, stage=None):
        # transforms
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        # ipdb.set_trace()
        self.train_ds.dataset.transform = self.train_transforms
        self.val_ds.dataset.transform = self.val_transforms
        # self.test_ds.dataset.transforms = test_transforms

        # self.train_ds.transform = train_transforms
        # self.val_ds.transform = val_transforms
        # self.test_ds.transform = test_transforms

        # data

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

