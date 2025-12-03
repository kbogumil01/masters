import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, Any
from .config import SubDatasetConfig

def _norm_metadata(seq_meta: Dict) -> torch.Tensor:
    """
    Normalizuje metadane do zakresu akceptowalnego przez sieć (0-1).
    Zgodne z logiką z pracy magisterskiej.
    """
    qp = float(seq_meta.get("qp", 32))
    qp_n = qp / 64.0  
    alf = float(seq_meta.get("alf", 0))
    sao = float(seq_meta.get("sao", 0))
    db = float(seq_meta.get("db", 0))
    return torch.tensor([qp_n, alf, sao, db], dtype=torch.float32).view(4, 1, 1)


class VVCChunksPTDataset(Dataset):
    """
    Główny dataset. Ładuje chunki .pt (obraz + cechy) bezpośrednio z dysku.
    """
    def __init__(
        self,
        decoded_root: str,
        orig_root: str,
        fused_root: Optional[str] = None, # Ignorowane (cechy są w chunkach), ale zostawiamy dla zgodności
        chunk_h: int = 132,
        chunk_w: int = 132,
        border: int = 2,
    ):
        self.decoded_root = decoded_root
        self.orig_root = orig_root
        self.chunk_h = chunk_h
        self.chunk_w = chunk_w
        
        if not os.path.exists(decoded_root):
            # Jeśli ścieżka nie istnieje, wypisz warning ale nie crashuj od razu (może to walidacja)
            print(f"[Dataset] WARNING: Path does not exist: {decoded_root}")
            self.dec_files = []
        else:
            print(f"[Dataset] Indexing chunks in {decoded_root}...")
            # Szukamy rekurencyjnie wszystkich plików chunk_*.pt
            self.dec_files = sorted(glob.glob(os.path.join(decoded_root, "*", "chunk_*.pt")))
            print(f"[Dataset] Found {len(self.dec_files)} chunks.")

    def __len__(self):
        return len(self.dec_files)

    def __getitem__(self, idx):
        dec_path = self.dec_files[idx]
        
        try:
            # 1. Ładowanie chunka z dysku (USB SSD)
            d_dec = torch.load(dec_path, map_location="cpu")
        except Exception as e:
            print(f"[ERR] Corrupt file {dec_path}: {e}")
            # W razie błędu zwróćmy losowy inny element (prosta strategia fallback)
            return self.__getitem__((idx + 1) % len(self))
        
        # 2. Znalezienie ścieżki do oryginału
        # dec_path: .../chunks_pt/SeqName_Profil_QP.../chunk_poc...
        seq_dir_dec = os.path.dirname(dec_path)
        chunk_name = os.path.basename(dec_path)
        
        # Parsowanie nazwy folderu (usuwamy sufiksy _AI_QP...)
        seq_folder_name = os.path.basename(seq_dir_dec)
        
        if "_AI_" in seq_folder_name:
            seq_pure = seq_folder_name.split("_AI_")[0]
        elif "_RA_" in seq_folder_name:
            seq_pure = seq_folder_name.split("_RA_")[0]
        else:
            seq_pure = seq_folder_name
            
        orig_path = os.path.join(self.orig_root, seq_pure, chunk_name)
        
        # 3. Ładowanie oryginału
        try:
            d_orig = torch.load(orig_path, map_location="cpu")
        except FileNotFoundError:
             # Jeśli nie ma oryginału (błąd generowania), użyj zdekodowanego jako placeholder (żeby trening nie padł)
             d_orig = d_dec 

        # 4. Konwersja na float [0, 1]
        dec_tensor = d_dec["chunk"].float() / 255.0
        orig_tensor = d_orig["chunk"].float() / 255.0
        
        # Metadane
        meta_tensor = _norm_metadata(d_dec["seq_meta"])
        
        # 5. Cechy VVC (Fused Maps)
        if "vvc_features" in d_dec:
            # Są w pliku (float16 -> float32)
            vvc_feat = d_dec["vvc_features"].float()
        else:
            # Brak cech -> same zera
            vvc_feat = torch.zeros(6, self.chunk_h, self.chunk_w, dtype=torch.float32)

        # Zwracamy: (input, target, metadata, extra_info, features)
        return dec_tensor, orig_tensor, meta_tensor, None, vvc_feat


class FrameDataset(torch.utils.data.Dataset):
    """
    Legacy dataset do testów na pełnych klatkach PNG (opcjonalny).
    """
    FRAME_GLOB = "{folder}/*__*__*/*/*.png"
    FRAME_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{is_intra}.png"
    ORIG_FRAME_NAME = "{file}/{frame}.png"

    def __init__(self, settings: SubDatasetConfig, chunk_transform, metadata_transform):
        super().__init__()
        self.chunk_folder = settings.chunk_folder
        self.orig_chunk_folder = settings.orig_chunk_folder
        self.chunk_height = getattr(settings, "chunk_height", 132)
        self.chunk_width = getattr(settings, "chunk_width", 132)
        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform
        self.frame_files = glob(self.FRAME_GLOB.format(folder=self.chunk_folder))

    def _metadata_to_np(self, qp, alf, sao, db):
        return np.array((qp / 64, alf, sao, db), dtype=np.float32)

    def _parse_metadata(self, path):
        fname, profiles, frame = path.split("/")[-3:]
        profile, qp, alf, db, sao = profiles.split("_")
        frame, is_intra = frame.split("_")
        return dict(
            file=fname, profile=profile, qp=int(qp[2:]), alf=int(alf[3:]),
            db=int(db[2:]), sao=int(sao[3:]), frame=int(frame),
            is_intra=is_intra.split(".")[0] == "True",
        )

    def __getitem__(self, idx):
        meta = self._parse_metadata(self.frame_files[idx])
        frame_path = os.path.join(self.chunk_folder, self.FRAME_NAME.format_map(meta))
        orig_path = os.path.join(self.orig_chunk_folder, self.ORIG_FRAME_NAME.format_map(meta))

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
            meta,
            torch.zeros(6, self.chunk_height, self.chunk_width)
        )

    def __len__(self):
        return len(self.frame_files)