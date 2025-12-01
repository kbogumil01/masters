# enhancer/dataset_pt.py
import os
import glob
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

def _norm_metadata(seq_meta: Dict) -> torch.Tensor:
    # Normalizacja metadanych (zgodna z Twoim configiem)
    qp = float(seq_meta.get("qp", 32))
    qp_n = qp / 64.0  # Normalizacja QP
    alf = float(seq_meta.get("alf", 0))
    sao = float(seq_meta.get("sao", 0))
    db = float(seq_meta.get("db", 0))
    # Zwracamy tensor [4, 1, 1]
    return torch.tensor([qp_n, alf, sao, db], dtype=torch.float32).view(4, 1, 1)

class VVCChunksPTDataset(Dataset):
    def __init__(
        self,
        decoded_root: str,
        orig_root: str,
        fused_root: Optional[str] = None, # Ignorowane, bo dane są w chunkach
        chunk_h: int = 132,
        chunk_w: int = 132,
        border: int = 2,
    ):
        self.decoded_root = decoded_root
        self.orig_root = orig_root
        self.chunk_h = chunk_h
        self.chunk_w = chunk_w
        
        print(f"[Dataset] Indexing chunks in {decoded_root}...")
        # Szukamy rekurencyjnie wszystkich plików .pt
        self.dec_files = sorted(glob.glob(os.path.join(decoded_root, "*", "chunk_*.pt")))
        print(f"[Dataset] Found {len(self.dec_files)} chunks.")
        
    def __len__(self):
        return len(self.dec_files)

    def __getitem__(self, idx):
        dec_path = self.dec_files[idx]
        
        # 1. Ładowanie chunka (mały plik z NVMe)
        #    Ten plik zawiera już: obraz, metadane ORAZ vvc_features (jeśli zostały wygenerowane)
        try:
            d_dec = torch.load(dec_path, map_location="cpu")
        except Exception as e:
            print(f"[ERR] Corrupt file {dec_path}: {e}")
            raise e
        
        # 2. Znalezienie ścieżki do oryginału
        #    dec_path: .../chunks_pt/SeqName_Profil_QP.../chunk_poc...
        seq_dir_dec = os.path.dirname(dec_path)
        chunk_name = os.path.basename(dec_path)
        
        # Parsowanie nazwy folderu zdekodowanego, aby znaleźć folder z oryginałami
        # Np. "FourPeople_AI_QP32..." -> "FourPeople"
        seq_folder_name = os.path.basename(seq_dir_dec)
        
        if "_AI_" in seq_folder_name:
            seq_pure = seq_folder_name.split("_AI_")[0]
        elif "_RA_" in seq_folder_name:
            seq_pure = seq_folder_name.split("_RA_")[0]
        else:
            # Fallback dla folderów bez profilu (np. same nazwy sekwencji)
            seq_pure = seq_folder_name
            
        orig_path = os.path.join(self.orig_root, seq_pure, chunk_name)
        
        # 3. Ładowanie oryginału
        d_orig = torch.load(orig_path, map_location="cpu")

        # 4. Przygotowanie tensorów
        # Obrazy są zapisane jako uint8 [0-255], rzutujemy na float [0-1]
        dec_tensor = d_dec["chunk"].float() / 255.0
        orig_tensor = d_orig["chunk"].float() / 255.0
        
        # Metadane z chunka
        meta_tensor = _norm_metadata(d_dec["seq_meta"])
        
        # 5. Obsługa Fused Maps (VVC Features) [TO JEST KLUCZOWA ZMIANA]
        if "vvc_features" in d_dec:
            # Mamy cechy w pliku! (float16 -> float32)
            vvc_feat = d_dec["vvc_features"].float()
        else:
            # Brak cech (np. oryginał lub stary format) -> same zera
            vvc_feat = torch.zeros(6, self.chunk_h, self.chunk_w, dtype=torch.float32)

        # Zwracamy krotkę (input, target, metadata, extra_info, features)
        # extra_info=None (nie używamy obiektu Chunk w treningu dla wydajności)
        return dec_tensor, orig_tensor, meta_tensor, None, vvc_feat