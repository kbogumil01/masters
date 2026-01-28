import os
import glob
import re
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Tuple


def _norm_metadata(seq_meta: Dict, is_intra: bool) -> torch.Tensor:
    """
    6 kanałów metadanych (zgodnie z implementacją Piotra - dataset.py _metadata_to_np):
      0: profile_ai (AI=1, RA=0)
      1: qp/64
      2: alf
      3: sao  ← UWAGA: Piotr ma sao przed db!
      4: db
      5: is_intra
    Shape: (6, 1, 1)
    """
    prof = str(seq_meta.get("profile", "RA")).upper()
    profile_ai = 1.0 if "AI" in prof else 0.0

    qp = float(seq_meta.get("qp", 32))
    qp_n = qp / 64.0

    alf = float(seq_meta.get("alf", 0))
    sao = float(seq_meta.get("sao", 0))
    db = float(seq_meta.get("db", 0))

    intra = 1.0 if bool(is_intra) else 0.0

    # Kolejność zgodna z Piotrem: [profile, qp, alf, SAO, DB, intra]
    return torch.tensor([profile_ai, qp_n, alf, sao, db, intra], dtype=torch.float32).view(6, 1, 1)


def _infer_qp_from_path(path: str) -> Optional[int]:
    """
    Wyciąga QP z nazwy folderu/ścieżki typu: ..._QP28_...
    Zwraca None jeśli nie znaleziono.
    """
    m = re.search(r"_QP(\d+)_", path)
    if not m:
        return None
    return int(m.group(1))


def _build_orig_chunk_path(orig_root: str, d_dec: Dict, dec_path: str) -> str:
    """
    Twoje dane:
      decoded: .../<something>_AI_QPxx_.../chunk_*.pt
      orig:    .../<seq_meta["name"]>/chunk_*.pt
    """
    chunk_name = os.path.basename(dec_path)

    seq_name = d_dec.get("seq_meta", {}).get("name", None)
    if seq_name is None:
        raise KeyError(f"Missing seq_meta.name in: {dec_path}")

    # główna (poprawna dla Twojej struktury) ścieżka
    orig_path = os.path.join(orig_root, str(seq_name), chunk_name)

    # awaryjnie: czasem ktoś robi orig_root/<folder_decoded>/<chunk>
    if not os.path.exists(orig_path):
        seq_folder_name = os.path.basename(os.path.dirname(dec_path))
        cand = os.path.join(orig_root, seq_folder_name, chunk_name)
        if os.path.exists(cand):
            orig_path = cand

    return orig_path


class VVCChunksPTDataset(Dataset):
    """
    Dataset do TRENINGU na chunkach 132x132.
    - QP filter opcjonalny (allowed_qps)
    - mapowanie orig po seq_meta["name"]
    - strict_pairs: jeśli True -> brak orig = twardy błąd (zalecane)
    """
    def __init__(
        self,
        decoded_root: str,
        orig_root: str,
        fused_root: Optional[str] = None,
        chunk_h: int = 132,
        chunk_w: int = 132,
        border: int = 2,
        allowed_qps: Optional[List[int]] = None,
        min_std: float = 0.0,
        strict_pairs: bool = True,
    ):
        self.decoded_root = decoded_root
        self.orig_root = orig_root
        self.chunk_h = chunk_h
        self.chunk_w = chunk_w
        self.allowed_qps = allowed_qps
        self.strict_pairs = strict_pairs

        if not os.path.exists(decoded_root):
            print(f"[Dataset] WARNING: Path does not exist: {decoded_root}")
            self.dec_files = []
            return

        print(f"[Dataset] Indexing chunks in {decoded_root} ...")
        all_files = sorted(glob.glob(os.path.join(decoded_root, "*", "chunk_*.pt")))
        print(f"[Dataset] Found {len(all_files)} chunk files total.")

        # QP filter
        if allowed_qps is None:
            print("[Dataset] QP filter: OFF (using all QPs).")
            self.dec_files = all_files
        else:
            print(f"[Dataset] QP filter: {allowed_qps}")
            kept = []
            ignored = 0
            for f in all_files:
                qp_val = _infer_qp_from_path(f)
                if qp_val is None:
                    ignored += 1
                    continue
                if qp_val in allowed_qps:
                    kept.append(f)
                else:
                    ignored += 1
            self.dec_files = kept
            print(f"[Dataset] Selected {len(self.dec_files)} chunks after QP filter.")
            print(f"[Dataset] Ignored {ignored} chunks (QP mismatch or missing QP).")

    def __len__(self) -> int:
        return len(self.dec_files)

    def __getitem__(self, idx: int):
        if len(self.dec_files) == 0:
            raise IndexError("Empty dataset: no files indexed.")

        dec_path = self.dec_files[idx]

        try:
            d_dec = torch.load(dec_path, map_location="cpu")
        except Exception as e:
            # jeśli plik uszkodzony, spróbuj kolejny
            return self.__getitem__((idx + 1) % len(self))

        # zbuduj ścieżkę do orig na podstawie seq_meta["name"]
        orig_path = _build_orig_chunk_path(self.orig_root, d_dec, dec_path)

        if not os.path.exists(orig_path):
            msg = f"[Dataset] Missing orig for decoded={dec_path} -> expected orig={orig_path}"
            if self.strict_pairs:
                raise FileNotFoundError(msg)
            # tryb nie-strict: pomiń próbkę
            return self.__getitem__((idx + 1) % len(self))

        d_orig = torch.load(orig_path, map_location="cpu")

        # tensors
        dec_tensor = d_dec["chunk"].float() / 255.0
        orig_tensor = d_orig["chunk"].float() / 255.0

        meta_tensor = _norm_metadata(d_dec["seq_meta"], bool(d_dec.get("is_intra", False)))

        # vvc_features (na baseline możesz ignorować, ale zwracamy dla kompatybilności)
        if "vvc_features" in d_dec and d_dec["vvc_features"] is not None:
            vvc_feat = d_dec["vvc_features"].float()
        else:
            vvc_feat = torch.zeros(6, self.chunk_h, self.chunk_w, dtype=torch.float32)

        return dec_tensor, orig_tensor, meta_tensor, None, vvc_feat


class VVCFullFramePTDataset(Dataset):
    """
    Dataset do TESTÓW (full frames zapisane jako frame_*.pt).
    Struktura mapowania orig: orig_root/<seq_meta["name"]>/frame_*.pt
    """
    def __init__(
        self,
        decoded_root: str,
        orig_root: str,
        allowed_qps: Optional[List[int]] = None,
        strict_pairs: bool = True,
    ):
        self.decoded_root = decoded_root
        self.orig_root = orig_root
        self.allowed_qps = allowed_qps
        self.strict_pairs = strict_pairs

        if not os.path.exists(decoded_root):
            print(f"[TestDataset] WARNING: Path does not exist: {decoded_root}")
            self.dec_files = []
            return

        print(f"[TestDataset] Indexing full frames in {decoded_root} ...")
        all_files = sorted(glob.glob(os.path.join(decoded_root, "*", "frame_*.pt")))
        print(f"[TestDataset] Found {len(all_files)} full-frame files total.")

        if allowed_qps is None:
            print("[TestDataset] QP filter: OFF (using all QPs).")
            self.dec_files = all_files
        else:
            print(f"[TestDataset] Filtering for QPs: {allowed_qps}")
            kept = []
            for f in all_files:
                qp_val = _infer_qp_from_path(f)
                if qp_val is not None and qp_val in allowed_qps:
                    kept.append(f)
            self.dec_files = kept
            print(f"[TestDataset] Kept {len(self.dec_files)} full frames after QP filter.")

    def __len__(self) -> int:
        return len(self.dec_files)

    def __getitem__(self, idx: int):
        if len(self.dec_files) == 0:
            raise IndexError("Empty dataset: no files indexed.")

        dec_path = self.dec_files[idx]
        d_dec = torch.load(dec_path, map_location="cpu")

        fname = os.path.basename(dec_path)

        seq_name = d_dec.get("seq_meta", {}).get("name", None)
        if seq_name is None:
            raise KeyError(f"Missing seq_meta.name in: {dec_path}")

        orig_path = os.path.join(self.orig_root, str(seq_name), fname)

        if not os.path.exists(orig_path):
            # awaryjnie: orig_root/<folder_decoded>/<frame>
            seq_folder_name = os.path.basename(os.path.dirname(dec_path))
            cand = os.path.join(self.orig_root, seq_folder_name, fname)
            if os.path.exists(cand):
                orig_path = cand

        if not os.path.exists(orig_path):
            msg = f"[TestDataset] Missing orig for decoded={dec_path} -> expected orig={orig_path}"
            if self.strict_pairs:
                raise FileNotFoundError(msg)
            return self.__getitem__((idx + 1) % len(self))

        d_orig = torch.load(orig_path, map_location="cpu")

        dec_tensor = d_dec["chunk"].float() / 255.0
        orig_tensor = d_orig["chunk"].float() / 255.0

        meta_tensor = _norm_metadata(d_dec["seq_meta"], bool(d_dec.get("is_intra", False)))

        if "vvc_features" in d_dec and d_dec["vvc_features"] is not None:
            vvc_feat = d_dec["vvc_features"].float()
        else:
            vvc_feat = torch.zeros(6, dec_tensor.shape[1], dec_tensor.shape[2], dtype=torch.float32)

        # zwracamy seq_meta (przydaje się w ewaluacji)
        return dec_tensor, orig_tensor, meta_tensor, d_dec["seq_meta"], vvc_feat


class FrameDataset(Dataset):
    """
    Placeholder – w Twoim trainerze jest odwołanie do FrameDataset.save_frame,
    ale w tym repo FrameDataset jest pusty. Zostawiamy jak było.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError
