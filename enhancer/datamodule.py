import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from .config import DataloaderConfig, DatasetConfig
from .dataset import FrameDataset, VVCChunksPTDataset


def custom_collate(batch):
    """
    Pakuje dane z datasetu w batche tensorów.
    """
    dec_chunks = []
    orig_chunks = []
    metadata = []
    chunk_objs = []
    vvc_features = []
    
    for item in batch:
        dec_chunks.append(item[0])
        orig_chunks.append(item[1])
        metadata.append(item[2])
        chunk_objs.append(item[3])
        vvc_features.append(item[4])
    
    return (
        torch.stack(dec_chunks, dim=0),
        torch.stack(orig_chunks, dim=0),
        torch.stack(metadata, dim=0),
        chunk_objs,
        torch.stack(vvc_features, dim=0)
    )


class LoaderWrapper:
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
        if self.idx >= self.n_step:
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
        fused_maps_dir: str | None = None,
    ):
        super().__init__()
        self.config = dataloader_config
        self.dataset_config = dataset_config
        self.test_full_frames = test_full_frames
        
        # --- Tu definiujemy zmienne ---
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_pt_dataset(self, split: str):
        ds_cfg = getattr(self.dataset_config, split)
        orig_root = ds_cfg.orig_chunks_pt_root or ds_cfg.orig_pt_root
        
        if not ds_cfg.chunks_pt_root:
            return None

        return VVCChunksPTDataset(
            decoded_root=ds_cfg.chunks_pt_root,
            orig_root=orig_root,
            fused_root=ds_cfg.fused_maps_root,
            chunk_h=ds_cfg.chunk_height,
            chunk_w=ds_cfg.chunk_width,
            border=ds_cfg.chunk_border,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_dataset = self._make_pt_dataset("train")
            
            val_path = self.dataset_config.val.chunks_pt_root
            train_path = self.dataset_config.train.chunks_pt_root
            
            if val_path and val_path != train_path:
                print(f"[DataModule] Loading separate validation set from {val_path}")
                self.train_dataset = full_dataset
                self.val_dataset = self._make_pt_dataset("val")
            else:
                print(f"[DataModule] Splitting training set (95% train, 5% val).")
                total_len = len(full_dataset)
                val_len = int(total_len * 0.05)
                train_len = total_len - val_len
                
                # Tutaj przypisujemy self.val_dataset
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset, [train_len, val_len], 
                    generator=torch.Generator().manual_seed(42)
                )

        if stage in ("test", "predict"):
            if self.test_full_frames:
                self.test_dataset = FrameDataset(
                    settings=self.dataset_config.test,
                    chunk_transform=self.chunk_transform(),
                    metadata_transform=self.metadata_transform(),
                )
            else:
                self.test_dataset = self._make_pt_dataset("test")

    def train_dataloader(self):
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=8,  
            pin_memory=True,
            collate_fn=custom_collate,
            persistent_workers=True,
            prefetch_factor=4
        )
        return LoaderWrapper(dl, self.config.n_step)

    def val_dataloader(self):
        dl = DataLoader(
            self.val_dataset,  # <--- POPRAWKA: było self.dataset_val
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=custom_collate,
            persistent_workers=True,
            prefetch_factor=4
        )
        return LoaderWrapper(dl, self.config.val_n_step)

    def test_dataloader(self, shuffle: bool = False):
        collate_fn = None if self.test_full_frames else custom_collate
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size if not self.test_full_frames else 1,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=8,
            collate_fn=collate_fn,
            prefetch_factor=4
        )

    def predict_dataloader(self):
        return self.test_dataloader(shuffle=True)

    def chunk_transform(self):
        return transforms.Compose([transforms.ToTensor()])

    def metadata_transform(self):
        def transform(metadata):
            return torch.as_tensor(metadata).float().view(len(metadata), 1, 1)
        return transform