import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from .config import DataloaderConfig, DatasetConfig

from .dataset import VVCDataset, FrameDataset
from .dataset_npz import VVCDatasetNPZ, get_vvc_dataset


class LoaderWrapper:
    """LoaderWrapper."""

    def __init__(self, dataloader: DataLoader, n_step: int):
        """__init__.

        :param dataloader:
        :type dataloader: DataLoader
        :param n_step:
        :type n_step: int
        """
        self.n_step = n_step
        self.idx = 0
        self.dataloader = dataloader
        self.iter_loader = iter(dataloader)

    def __iter__(self) -> "LoaderWrapper":
        """__iter__.

        :rtype: "LoaderWrapper"
        """
        return self

    def __len__(self) -> int:
        """__len__.

        :rtype: int
        """
        return self.n_step

    def __next__(self):
        """__next__."""
        if self.idx == self.n_step:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1

        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.dataloader)
            return next(self.iter_loader)


class VVCDataModule(pl.LightningDataModule):
    """VVCDataModule."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        dataloader_config: DataloaderConfig,
        test_full_frames: bool = False,
        fused_maps_dir: str = None,  # NEW: Path to fused maps
    ):
        """__init__.

        :param chunk_folder:
        :type chunk_folder: str
        :param orig_chunk_folder:
        :type orig_chunk_folder: str
        :param chunk_height:
        :type chunk_height: int
        :param chunk_width:
        :type chunk_width: int
        :param batch_size:
        :type batch_size: int
        :param fused_maps_dir:
        :type fused_maps_dir: str
        """
        super().__init__()
        self.config = dataloader_config
        self.dataset_config = dataset_config
        self.fused_maps_dir = fused_maps_dir  # NEW

        self.dataset_val = None
        self.dataset_test = None
        self.dataset_train = None
        self.test_full_frames = test_full_frames

    def setup(self, stage=None):
        """setup.

        :param stage:
        """
        if stage == "fit":
            self.dataset_train = get_vvc_dataset(
                settings=self.dataset_config.train,
                chunk_transform=self.chunk_transform(),
                metadata_transform=self.metadata_transform(),
                fused_maps_dir=self.fused_maps_dir,  # NEW
                split='train',  # NEW: train split
            )

            epochs_for_real_one = len(self.dataset_train) / self.config.n_step
            print(f"it takes {epochs_for_real_one} of training to reach one real epoch")

            self.dataset_val = get_vvc_dataset(
                settings=self.dataset_config.val,
                chunk_transform=self.chunk_transform(),
                metadata_transform=self.metadata_transform(),
                fused_maps_dir=self.fused_maps_dir,  # NEW
                split='val',  # NEW: validation split
            )

            epochs_for_real_one = len(self.dataset_val) / self.config.val_n_step
            print(
                f"it takes {epochs_for_real_one} of validation to reach one real epoch"
            )

        if stage in ("test", "predict"):
            if self.test_full_frames:
                self.dataset_test = FrameDataset(
                    settings=self.dataset_config.test,
                    chunk_transform=self.chunk_transform(),
                    metadata_transform=self.metadata_transform(),
                )
            else:
                self.dataset_test = get_vvc_dataset(
                    settings=self.dataset_config.test,
                    chunk_transform=self.chunk_transform(),
                    metadata_transform=self.metadata_transform(),
                    fused_maps_dir=self.fused_maps_dir,  # NEW
                    split='test',  # NEW: test split
                )

    def train_dataloader(self):
        """train_dataloader - FIXED: Remove LoaderWrapper to use full dataset."""
        data_loader = DataLoader(
            self.dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
        )
        # REMOVED LoaderWrapper - now uses entire train split per epoch!
        return data_loader

    def test_dataloader(self, shuffle=False):
        """test_dataloader."""
        data_loader = DataLoader(
            self.dataset_test,
            batch_size=self.config.test_batch_size if not self.test_full_frames else 1,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=2,  # Reduced from os.cpu_count()
        )
        return data_loader

    def predict_dataloader(self):
        return self.test_dataloader(True)

    def val_dataloader(self):
        """val_dataloader."""
        data_loader = DataLoader(
            self.dataset_val,
            batch_size=self.config.val_batch_size,
            shuffle=True,
            pin_memory=False,  # DISABLED - causes VRAM bloat during sanity check
            num_workers=2,  # Less for val
            persistent_workers=True,
            prefetch_factor=2,
        )
        return LoaderWrapper(
            data_loader,
            self.config.val_n_step,
        )

    def chunk_transform(self):
        """chunk_transform."""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return transform

    def metadata_transform(self):
        """metadata_transform."""

        def transform(metadata):
            return torch.as_tensor(metadata).float().view(len(metadata), 1, 1)

        return transform
