import os
import numpy as np
import torch
from pathlib import Path
from glob import glob
from .config import SubDatasetConfig


class FrameDataset(torch.utils.data.Dataset):
    """
    Dataset do testów na pełnych klatkach (PNG/YUV → Tensor).
    Używany tylko w trybie --test z parametrem test_full_frames=True.
    """

    FRAME_GLOB = "{folder}/*__*__*/*/*.png"
    FRAME_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{is_intra}.png"
    ORIG_FRAME_NAME = "{file}/{frame}.png"

    def __init__(
        self,
        settings: SubDatasetConfig,
        chunk_transform,
        metadata_transform,
    ):
        super().__init__()

        self.chunk_folder = settings.chunk_folder
        self.orig_chunk_folder = settings.orig_chunk_folder
        self.chunk_height = getattr(settings, "chunk_height", 132)
        self.chunk_width = getattr(settings, "chunk_width", 132)

        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform

        self.frame_files = glob(self.FRAME_GLOB.format(folder=self.chunk_folder))

    def _metadata_to_np(self, qp, alf, sao, db):
        """Zamienia proste metadane na tensor numeryczny."""
        return np.array(
            (qp / 64, alf, sao, db),
            dtype=np.float32
        )

    def _parse_metadata(self, path):
        """
        Parsuje ścieżkę typu:
        /path/file/profile_QP32_ALF1_DB0_SAO1/12_True.png
        """
        fname, profiles, frame = path.split("/")[-3:]
        profile, qp, alf, db, sao = profiles.split("_")
        frame, is_intra = frame.split("_")
        return dict(
            file=fname,
            profile=profile,
            qp=int(qp[2:]),
            alf=int(alf[3:]),
            db=int(db[2:]),
            sao=int(sao[3:]),
            frame=int(frame),
            is_intra=is_intra.split(".")[0] == "True",
        )

    def __getitem__(self, idx):
        meta = self._parse_metadata(self.frame_files[idx])
        frame_path = os.path.join(self.chunk_folder, self.FRAME_NAME.format_map(meta))
        orig_path = os.path.join(self.orig_chunk_folder, self.ORIG_FRAME_NAME.format_map(meta))

        # Ładowanie
        with open(frame_path, "rb") as f:
            frame = np.frombuffer(f.read(), dtype=np.uint8)
            frame = np.resize(frame, (self.chunk_height, self.chunk_width, 3))

        with open(orig_path, "rb") as f:
            orig = np.frombuffer(f.read(), dtype=np.uint8)
            orig = np.resize(orig, (self.chunk_height, self.chunk_width, 3))

        meta_tensor = self._metadata_to_np(meta["qp"], meta["alf"], meta["sao"], meta["db"])

        return (
            self.chunk_transform(frame),
            self.chunk_transform(orig),
            self.metadata_transform(meta_tensor),
            meta,  # można logować / zapisać
        )

    def __len__(self):
        return len(self.frame_files)
