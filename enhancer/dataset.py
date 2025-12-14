import os
import glob
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, List

def _norm_metadata(seq_meta: Dict) -> torch.Tensor:
    qp = float(seq_meta.get("qp", 32))
    qp_n = qp / 64.0  
    alf = float(seq_meta.get("alf", 0))
    sao = float(seq_meta.get("sao", 0))
    db = float(seq_meta.get("db", 0))
    return torch.tensor([qp_n, alf, sao, db], dtype=torch.float32).view(4, 1, 1)

class VVCChunksPTDataset(Dataset):
    """Dataset do TRENINGU: małe chunki."""
    def __init__(
        self,
        decoded_root: str,
        orig_root: str,
        fused_root: Optional[str] = None, 
        chunk_h: int = 132,
        chunk_w: int = 132,
        border: int = 2,
        # Nowy parametr: lista dozwolonych QP. Jeśli None -> bierze wszystko.
        allowed_qps: Optional[List[int]] = None 
    ):
        self.decoded_root = decoded_root
        self.orig_root = orig_root
        self.chunk_h = chunk_h
        self.chunk_w = chunk_w
        
        if not os.path.exists(decoded_root):
            print(f"[Dataset] WARNING: Path does not exist: {decoded_root}")
            self.dec_files = []
        else:
            print(f"[Dataset] Indexing chunks in {decoded_root}...")
            all_files = sorted(glob.glob(os.path.join(decoded_root, "*", "chunk_*.pt")))
            
            # --- FILTROWANIE QP ---
            if allowed_qps is not None and len(allowed_qps) > 0:
                print(f"[Dataset] Filtering for QPs: {allowed_qps}")
                self.dec_files = []
                # Regex szukający QP w ścieżce, np. "..._QP37_..."
                qp_pattern = re.compile(r"_QP(\d+)_")
                
                for f in all_files:
                    # Szukamy QP w nazwie folderu nadrzędnego (szybciej) lub pliku
                    match = qp_pattern.search(f)
                    if match:
                        qp_val = int(match.group(1))
                        if qp_val in allowed_qps:
                            self.dec_files.append(f)
            else:
                self.dec_files = all_files
                
            print(f"[Dataset] Found {len(self.dec_files)} chunks (after filtering).")

    def __len__(self):
        return len(self.dec_files)

    def __getitem__(self, idx):
        dec_path = self.dec_files[idx]
        try:
            d_dec = torch.load(dec_path, map_location="cpu")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
        
        seq_dir_dec = os.path.dirname(dec_path)
        chunk_name = os.path.basename(dec_path)
        seq_folder_name = os.path.basename(seq_dir_dec)
        
        if "_AI_" in seq_folder_name: seq_pure = seq_folder_name.split("_AI_")[0]
        elif "_RA_" in seq_folder_name: seq_pure = seq_folder_name.split("_RA_")[0]
        else: seq_pure = seq_folder_name
            
        orig_path = os.path.join(self.orig_root, seq_pure, chunk_name)
        try:
            d_orig = torch.load(orig_path, map_location="cpu")
        except FileNotFoundError:
             d_orig = d_dec 

        dec_tensor = d_dec["chunk"].float() / 255.0
        orig_tensor = d_orig["chunk"].float() / 255.0
        meta_tensor = _norm_metadata(d_dec["seq_meta"])
        
        if "vvc_features" in d_dec:
            vvc_feat = d_dec["vvc_features"].float()
        else:
            vvc_feat = torch.zeros(6, self.chunk_h, self.chunk_w, dtype=torch.float32)

        return dec_tensor, orig_tensor, meta_tensor, None, vvc_feat


class VVCFullFramePTDataset(Dataset):
    """Dataset do TESTÓW: pełne klatki."""
    def __init__(self, decoded_root, orig_root, allowed_qps: Optional[List[int]] = None):
        self.decoded_root = decoded_root
        self.orig_root = orig_root
        
        if not os.path.exists(decoded_root):
             print(f"[TestDataset] WARNING: Path not found: {decoded_root}")
             self.dec_files = []
        else:
             print(f"[TestDataset] Indexing full frames in {decoded_root}...")
             all_files = sorted(glob.glob(os.path.join(decoded_root, "*", "frame_*.pt")))
             
             # --- FILTROWANIE QP (Testy) ---
             if allowed_qps is not None and len(allowed_qps) > 0:
                print(f"[TestDataset] Filtering for QPs: {allowed_qps}")
                self.dec_files = []
                qp_pattern = re.compile(r"_QP(\d+)_")
                for f in all_files:
                    match = qp_pattern.search(f)
                    if match:
                        qp_val = int(match.group(1))
                        if qp_val in allowed_qps:
                            self.dec_files.append(f)
             else:
                self.dec_files = all_files
                
             print(f"[TestDataset] Found {len(self.dec_files)} full test frames (after filtering).")

    def __len__(self):
        return len(self.dec_files)

    def __getitem__(self, idx):
        dec_path = self.dec_files[idx]
        d_dec = torch.load(dec_path, map_location="cpu")
        
        seq_dir_dec = os.path.dirname(dec_path)
        fname = os.path.basename(dec_path)
        seq_folder_name = os.path.basename(seq_dir_dec)
        
        if "_AI_" in seq_folder_name: seq_pure = seq_folder_name.split("_AI_")[0]
        elif "_RA_" in seq_folder_name: seq_pure = seq_folder_name.split("_RA_")[0]
        else: seq_pure = seq_folder_name
        
        orig_path = os.path.join(self.orig_root, seq_pure, fname)
        try:
            d_orig = torch.load(orig_path, map_location="cpu")
        except FileNotFoundError:
            d_orig = d_dec

        dec_tensor = d_dec["chunk"].float() / 255.0
        orig_tensor = d_orig["chunk"].float() / 255.0
        meta_tensor = _norm_metadata(d_dec["seq_meta"])

        if "vvc_features" in d_dec:
            vvc_feat = d_dec["vvc_features"].float()
        else:
            vvc_feat = torch.zeros(6, dec_tensor.shape[1], dec_tensor.shape[2], dtype=torch.float32)

        return dec_tensor, orig_tensor, meta_tensor, d_dec["seq_meta"], vvc_feat

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs): pass
    def __len__(self): return 0
    def __getitem__(self, idx): raise NotImplementedError