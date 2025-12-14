import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from .config import DataloaderConfig, DatasetConfig
from .dataset import VVCChunksPTDataset, VVCFullFramePTDataset

# === KONFIGURACJA QP ===
# Wybierz QP, na których chcesz trenować i testować.
# Usunięcie niskich QP (22, 27) znacząco podniesie zysk (Delta PSNR).
TARGET_QPS = [32, 37, 42, 47] 
# =======================

def custom_collate(batch):
    dec_chunks, orig_chunks, metadata, chunk_objs, vvc_features = zip(*batch)
    return (
        torch.stack(dec_chunks, 0),
        torch.stack(orig_chunks, 0),
        torch.stack(metadata, 0),
        list(chunk_objs),
        torch.stack(vvc_features, 0)
    )

def test_collate(batch):
    return custom_collate(batch)

class LoaderWrapper:
    def __init__(self, dataloader: DataLoader, n_step: int):
        self.n_step = n_step
        self.idx = 0
        self.dataloader = dataloader
        self.iter_loader = iter(dataloader)
    def __iter__(self): return self
    def __len__(self): return self.n_step
    def __next__(self):
        if self.idx >= self.n_step:
            self.idx = 0
            raise StopIteration
        self.idx += 1
        try: return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.dataloader)
            return next(self.iter_loader)

class VVCDataModule(pl.LightningDataModule):
    def __init__(self, dataset_config, dataloader_config, test_full_frames=False, fused_maps_dir=None):
        super().__init__()
        self.config = dataloader_config
        self.dataset_config = dataset_config
        self.test_full_frames = test_full_frames
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            ds_cfg = self.dataset_config.train
            full_dataset = VVCChunksPTDataset(
                decoded_root=ds_cfg.chunks_pt_root,
                orig_root=ds_cfg.orig_chunks_pt_root,
                chunk_h=ds_cfg.chunk_height, chunk_w=ds_cfg.chunk_width,
                allowed_qps=TARGET_QPS  # <--- Filtr
            )
            
            total_len = len(full_dataset)
            val_len = int(total_len * 0.05)
            train_len = total_len - val_len
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, val_len], 
                generator=torch.Generator().manual_seed(42)
            )

        if stage in ("test", "predict"):
            ds_cfg = self.dataset_config.test
            if self.test_full_frames:
                self.test_dataset = VVCFullFramePTDataset(
                    decoded_root=ds_cfg.chunks_pt_root,
                    orig_root=ds_cfg.orig_chunks_pt_root,
                    allowed_qps=TARGET_QPS # <--- Filtr
                )
            else:
                self.test_dataset = VVCChunksPTDataset(
                    decoded_root=ds_cfg.chunks_pt_root,
                    orig_root=ds_cfg.orig_chunks_pt_root,
                    allowed_qps=TARGET_QPS # <--- Filtr
                )

    def train_dataloader(self):
        if not self.train_dataset:
            return None
            
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
        # TU BYŁ BŁĄD - brakowało Wrappera. Teraz epoka będzie trwać n_step (5000)
        return LoaderWrapper(dl, self.config.n_step)

    def val_dataloader(self):
        dl = DataLoader(self.val_dataset, batch_size=self.config.val_batch_size, shuffle=False, 
                        num_workers=8, pin_memory=True, collate_fn=custom_collate, persistent_workers=True, prefetch_factor=4)
        return LoaderWrapper(dl, self.config.val_n_step)

    def test_dataloader(self):
        batch_size = 1 if self.test_full_frames else self.config.test_batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True, collate_fn=test_collate)

    def predict_dataloader(self):
        return self.test_dataloader()