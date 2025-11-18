import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from .config import DataloaderConfig, DatasetConfig
from .dataset import FrameDataset          # do ew. testów na pełnych klatkach
from .dataset_pt import VVCChunksPTDataset  # NOWY dataset .pt


def custom_collate(batch):
    """
    Custom collate function for VVCChunksPTDataset.
    Dataset returns: (dec_chunk, orig_chunk, metadata, chunk_obj, vvc_features)
    where chunk_obj can be None during training (not needed).
    """
    # Unpack batch items
    dec_chunks = []
    orig_chunks = []
    metadata = []
    chunk_objs = []
    vvc_features = []
    
    for item in batch:
        dec_chunks.append(item[0])
        orig_chunks.append(item[1])
        metadata.append(item[2])
        chunk_objs.append(item[3])  # Keep it even if None
        vvc_features.append(item[4])
    
    # Stack tensors
    dec_chunks = torch.stack(dec_chunks, dim=0)
    orig_chunks = torch.stack(orig_chunks, dim=0)
    metadata = torch.stack(metadata, dim=0)
    vvc_features = torch.stack(vvc_features, dim=0)
    
    # chunk_objs is a list (not stacked - can contain None or dicts)
    return dec_chunks, orig_chunks, metadata, chunk_objs, vvc_features


class LoaderWrapper:
    """Ogranicza licznik kroków na epokę (tak jak u Piotra)."""

    def __init__(self, dataloader: DataLoader, n_step: int):
        self.n_step = n_step
        self.idx = 0
        self.dataloader = dataloader
        self.iter_loader = iter(dataloader)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_step

    def __next__(self):
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
    def __init__(
        self,
        dataset_config: DatasetConfig,
        dataloader_config: DataloaderConfig,
        test_full_frames: bool = False,
        fused_maps_dir: str | None = None,  # ← DODANE, żeby nie gryzło się z __main__.py
    ):
        super().__init__()
        self.config = dataloader_config
        self.dataset_config = dataset_config

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.test_full_frames = test_full_frames

        # fused_maps_dir tu w zasadzie ignorujemy,
        # bo i tak korzystamy z dataset_config.*.fused_maps_root
        self._legacy_fused_maps_dir = fused_maps_dir

    # ------------- pomocnicze „fabryki” datasetów .pt -------------

    def _make_pt_dataset(self, split: str):
        """Tworzy VVCChunksPTDataset dla danego splitu."""
        ds_cfg = getattr(self.dataset_config, split)

        # Use orig_chunks_pt_root if available, fallback to orig_pt_root
        orig_root = ds_cfg.orig_chunks_pt_root or ds_cfg.orig_pt_root

        return VVCChunksPTDataset(
            decoded_root=ds_cfg.chunks_pt_root,
            orig_root=orig_root,
            fused_root=ds_cfg.fused_maps_root,
            chunk_h=ds_cfg.chunk_height,
            chunk_w=ds_cfg.chunk_width,
            border=ds_cfg.chunk_border,
        )

    # ------------- Lightning hooks -------------

    def setup(self, stage=None):
        # TRAIN + VAL
        if stage == "fit" or stage is None:
            # train
            self.dataset_train = self._make_pt_dataset("train")
            epochs_for_real_one = len(self.dataset_train) / self.config.n_step
            print(f"[DataModule] it takes {epochs_for_real_one} pseudo-epochs (train) to see all data once")

            # val
            self.dataset_val = self._make_pt_dataset("val")
            epochs_for_real_one_val = len(self.dataset_val) / self.config.val_n_step
            print(f"[DataModule] it takes {epochs_for_real_one_val} pseudo-epochs (val) to see all data once")

        # TEST / PREDICT
        if stage in ("test", "predict"):
            if self.test_full_frames:
                # jeśli kiedyś będziesz chciał test na pełnych klatkach (PNG) – nadal zostawiam
                self.dataset_test = FrameDataset(
                    settings=self.dataset_config.test,
                    chunk_transform=self.chunk_transform(),
                    metadata_transform=self.metadata_transform(),
                )
            else:
                # standardowo: test na chunkach .pt
                self.dataset_test = self._make_pt_dataset("test")

    # ------------- dataloadery -------------

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            collate_fn=custom_collate,
        )

    def val_dataloader(self):
        dl = DataLoader(
            self.dataset_val,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            collate_fn=custom_collate,
        )
        return LoaderWrapper(dl, self.config.val_n_step)

    def test_dataloader(self, shuffle: bool = False):
        # Use custom_collate only for chunks_pt dataset, not for FrameDataset
        collate_fn = None if self.test_full_frames else custom_collate
        
        dl = DataLoader(
            self.dataset_test,
            batch_size=self.config.test_batch_size if not self.test_full_frames else 1,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=2,
            collate_fn=collate_fn,
        )
        return dl

    def predict_dataloader(self):
        return self.test_dataloader(shuffle=True)

    # ------------- transformy (używane tylko przez FrameDataset) -------------

    def chunk_transform(self):
        # dla FrameDataset (PNG); VVCChunksPTDataset już zwraca tensory
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return transform

    def metadata_transform(self):
        def transform(metadata):
            return torch.as_tensor(metadata).float().view(len(metadata), 1, 1)
        return transform
