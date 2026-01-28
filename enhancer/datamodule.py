import os
import re
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from .dataset import VVCChunksPTDataset, VVCFullFramePTDataset


def custom_collate(batch):
    dec_chunks, orig_chunks, metadata, chunk_objs, vvc_features = zip(*batch)
    return (
        torch.stack(dec_chunks, 0),
        torch.stack(orig_chunks, 0),
        torch.stack(metadata, 0),
        list(chunk_objs),
        torch.stack(vvc_features, 0),
    )


def test_collate(batch):
    return custom_collate(batch)


class LoaderWrapper:
    """
    Wrapper jak u Ciebie: umożliwia "epokę" o stałej liczbie kroków n_step,
    niezależnie od długości datasetu.
    """
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
        self.idx += 1
        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.dataloader)
            return next(self.iter_loader)


def _seq_key_from_dec_path(dec_path: str) -> str:
    """
    Grupowanie po sekwencji na podstawie folderu decoded.
    U Ciebie folder wygląda np.:
      720p50_mobcal_ter_AI_QP28_ALF0_DB0_SAO0
    Klucz sekwencji bierzemy jako część przed _AI_ / _RA_.
    """
    folder = os.path.basename(os.path.dirname(dec_path))
    if "_AI_" in folder:
        return folder.split("_AI_")[0]
    if "_RA_" in folder:
        return folder.split("_RA_")[0]
    return folder


class VVCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_config,
        dataloader_config,
        test_full_frames: bool = False,
        fused_maps_dir=None,
        val_ratio: float = 0.05,
        seed: int = 42,
        num_workers_train: int = 4, # było 8
        num_workers_test: int = 2, # było 4
    ):
        super().__init__()
        self.config = dataloader_config
        self.dataset_config = dataset_config
        self.test_full_frames = test_full_frames

        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        self.num_workers_train = int(num_workers_train)
        self.num_workers_test = int(num_workers_test)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        print(f"\n[DataModule] setup(stage={stage})")

        if stage == "fit" or stage is None:
            ds_cfg = self.dataset_config.train

            full_dataset = VVCChunksPTDataset(
                decoded_root=ds_cfg.chunks_pt_root,
                orig_root=ds_cfg.orig_chunks_pt_root,
                chunk_h=ds_cfg.chunk_height,
                chunk_w=ds_cfg.chunk_width,
                border=getattr(ds_cfg, "chunk_border", 2),
                allowed_qps=[32, 37, 42, 47],   # ← JEDYNE allowed_qps
                strict_pairs=getattr(ds_cfg, "strict_pairs", True),
            )

            total_len = len(full_dataset)
            print(f"[DataModule] Train root indexed samples: {total_len}")
            if total_len == 0:
                raise RuntimeError("Empty training dataset. Check chunks_pt_root / filters.")

            # --- split po sekwencjach, żeby nie mieszać chunków z tej samej sekwencji ---
            seq_to_indices = {}
            for i, dec_path in enumerate(full_dataset.dec_files):
                k = _seq_key_from_dec_path(dec_path)
                seq_to_indices.setdefault(k, []).append(i)

            seq_keys = sorted(seq_to_indices.keys())
            g = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(seq_keys), generator=g).tolist()
            seq_keys = [seq_keys[i] for i in perm]

            n_val_seq = max(1, int(round(len(seq_keys) * self.val_ratio)))
            val_seqs = set(seq_keys[:n_val_seq])
            train_seqs = set(seq_keys[n_val_seq:])

            train_indices = []
            val_indices = []
            for k, idxs in seq_to_indices.items():
                if k in val_seqs:
                    val_indices.extend(idxs)
                else:
                    train_indices.extend(idxs)

            if len(val_indices) == 0:
                # awaryjnie: weź chociaż 1 próbkę
                val_indices = [train_indices.pop()]

            self.train_dataset = Subset(full_dataset, train_indices)
            self.val_dataset = Subset(full_dataset, val_indices)

            print(f"[DataModule] Split by sequences:")
            print(f"  sequences total: {len(seq_to_indices)}")
            print(f"  val sequences:   {len(val_seqs)}")
            print(f"  train samples:   {len(self.train_dataset)}")
            print(f"  val samples:     {len(self.val_dataset)}")

        if stage in ("test", "predict") or stage is None:
            ds_cfg = self.dataset_config.test

            if self.test_full_frames:
                self.test_dataset = VVCFullFramePTDataset(
                    decoded_root=ds_cfg.chunks_pt_root,
                    orig_root=ds_cfg.orig_chunks_pt_root,
                    allowed_qps=getattr(ds_cfg, "allowed_qps", None),
                    strict_pairs=getattr(ds_cfg, "strict_pairs", True),
                )
            else:
                self.test_dataset = VVCChunksPTDataset(
                    decoded_root=ds_cfg.chunks_pt_root,
                    orig_root=ds_cfg.orig_chunks_pt_root,
                    chunk_h=ds_cfg.chunk_height,
                    chunk_w=ds_cfg.chunk_width,
                    border=getattr(ds_cfg, "chunk_border", 2),
                    allowed_qps=getattr(ds_cfg, "allowed_qps", None),
                    strict_pairs=getattr(ds_cfg, "strict_pairs", True),
                )

            print(f"[DataModule] Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers_train,
            pin_memory=True,
            collate_fn=custom_collate,
        )
        return LoaderWrapper(dl, self.config.n_step)

    def val_dataloader(self):
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers_train,
            pin_memory=True,
            collate_fn=custom_collate,
        )
        return LoaderWrapper(dl, self.config.val_n_step)

    def test_dataloader(self):
        batch_size = 1 if self.test_full_frames else self.config.test_batch_size
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers_test,
            pin_memory=True,
            collate_fn=test_collate,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
