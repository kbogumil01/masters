# enhancer/dataset_pt.py
import os
import re
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


DECODED_DIR_RE = re.compile(
    r"^(?P<name>.+)_(?P<profile>AI|RA)_QP(?P<qp>\d+)_ALF(?P<alf>[01])_DB(?P<db>[01])_SAO(?P<sao>[01])$"
)


def _norm_metadata(seq_meta: Dict) -> torch.Tensor:
    """
    QP + flagi filtrów -> tensor [4, 1, 1] (tak jak oczekuje sieć)
    """
    qp = float(seq_meta.get("qp", 0))
    qp_n = qp / 51.0  # QP w [0,51]
    alf = float(seq_meta.get("alf", 0))
    sao = float(seq_meta.get("sao", 0))
    db = float(seq_meta.get("db", 0))

    return torch.tensor([qp_n, alf, sao, db], dtype=torch.float32).view(4, 1, 1)


def _load_pt(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def _upsample_fmaps_to_padded_frame(
    fm: np.ndarray,
    H: int,
    W: int,
    stride_y: int,
    stride_x: int,
    border: int,
    interp: int,
) -> np.ndarray:
    """
    Skalowanie map cech do rozmiaru ramki + ten sam bordurek,
    którego użyliśmy w split_to_chunks_pt.
    """
    # 1) do rozmiaru ramki
    fm = cv2.resize(fm, (W, H), interpolation=interp)
    # 2) bordur + dopad jak w split_to_chunks_pt
    fm = cv2.copyMakeBorder(
        fm,
        border,
        2 * border + ((-H) % stride_y),
        border,
        2 * border + ((-W) % stride_x),
        cv2.BORDER_REPLICATE if interp != cv2.INTER_NEAREST else cv2.BORDER_CONSTANT,
        value=0.0,
    )
    return fm


class VVCChunksPTDataset(Dataset):
    """
    Zestaw danych dla chunków zapisanych jako .pt:

      decoded_root: <...>/chunks_pt/<seq>/chunks_pocXXX.pt  (zdekodowane)
      orig_root:    <...>/orig_chunks_pt/<seq>/chunks_pocXXX.pt  (oryginał)
      fused_root:   <...>/decoded/<seq>/fused_maps/fused_maps_pocX.pt  (opcjonalnie)

    VVC features – 6 kanałów (Y-channel tylko):
      - y_ac_energy
      - y_nz_density
      - y_dc
      - boundary_bin
      - boundary_weight
      - size_map_norm

    Zwracany sample:
      (dec_chunk, orig_chunk, metadata_tensor, None, vvc_tensor)
    """

    def __init__(
        self,
        decoded_root: str,
        orig_root: str,
        fused_root: Optional[str] = None,
        chunk_h: int = 132,
        chunk_w: int = 132,
        border: int = 2,
    ):
        super().__init__()
        self.decoded_root = decoded_root
        self.orig_root = orig_root
        self.fused_root = fused_root
        self.chunk_h = int(chunk_h)
        self.chunk_w = int(chunk_w)
        self.border = int(border)

        # indeks: (seq, poc, idx_chunk)
        self.items: List[Tuple[str, int, int]] = []
        # cache metadanych per sekwencja
        self.seq_meta_cache: Dict[str, Dict] = {}
        # mapowanie (seq, poc) -> ścieżka do pliku .pt
        self.dec_index: Dict[Tuple[str, int], str] = {}
        self.orig_index: Dict[Tuple[str, int], str] = {}

        self._build_index()

    def _build_index(self):
        print(f"[VVCChunksPTDataset] Building index from {self.decoded_root}...")
        seqs = sorted(
            [
                d
                for d in os.listdir(self.decoded_root)
                if os.path.isdir(os.path.join(self.decoded_root, d))
            ]
        )
        print(f"[VVCChunksPTDataset] Found {len(seqs)} sequences")

        n_total = 0

        for seq_idx, seq in enumerate(seqs):
            seq_dec = os.path.join(self.decoded_root, seq)

            # Wszystkie POC-e dla tej sekwencji
            dec_pocs = sorted(glob.glob(os.path.join(seq_dec, "chunks_poc*.pt")))
            if not dec_pocs:
                continue

            print(
                f"[{seq_idx+1}/{len(seqs)}] Processing {seq}: {len(dec_pocs)} POCs..."
            )

            # Dopasuj folder z oryginałami
            # np. FourPeople_1280x720_60_AI_QP28...  -> FourPeople_1280x720_60
            seq_base = seq.split("_AI_")[0].split("_RA_")[0]
            seq_orig = os.path.join(self.orig_root, seq_base)
            if not os.path.isdir(seq_orig):
                # fallback – jeśli kiedyś nazwy będą 1:1
                seq_orig = os.path.join(self.orig_root, seq)
                if not os.path.isdir(seq_orig):
                    print(f"[WARN] Brak odpowiadającego folderu w orig_root dla {seq}")
                    continue

            for poc_idx, dec_path in enumerate(dec_pocs):
                # z nazwy pliku: chunks_poc00X.pt -> numer POC
                poc_str = os.path.basename(dec_path).replace("chunks_poc", "").replace(
                    ".pt", ""
                )
                poc = int(poc_str)

                orig_path = os.path.join(seq_orig, f"chunks_poc{poc_str}.pt")
                if not os.path.isfile(orig_path):
                    print(
                        f"[WARN] Brak orig_chunks {orig_path} – pomijam POC {poc} ({seq})"
                    )
                    continue

                # Jednorazowo wczytaj metadane sekwencji
                if seq not in self.seq_meta_cache:
                    d_dec = _load_pt(dec_path)
                    self.seq_meta_cache[seq] = {
                        "width": int(d_dec["seq_meta"]["width"]),
                        "height": int(d_dec["seq_meta"]["height"]),
                        "qp": int(d_dec["seq_meta"].get("qp", 0)),
                        "alf": int(d_dec["seq_meta"].get("alf", 0)),
                        "sao": int(d_dec["seq_meta"].get("sao", 0)),
                        "db": int(d_dec["seq_meta"].get("db", 0)),
                        "profile": d_dec["seq_meta"].get("profile", "AI"),
                        "num_chunks_per_poc": int(d_dec["chunks"].shape[0]),
                    }

                n_chunks = self.seq_meta_cache[seq]["num_chunks_per_poc"]

                self.dec_index[(seq, poc)] = dec_path
                self.orig_index[(seq, poc)] = orig_path

                for idx in range(n_chunks):
                    self.items.append((seq, poc, idx))
                    n_total += 1

        print(
            f"[VVCChunksPTDataset] Index built: {n_total} chunks from {len(seqs)} sequences"
        )
        if n_total == 0:
            raise RuntimeError(
                f"Brak danych w {self.decoded_root} vs {self.orig_root} – sprawdź ścieżki"
            )

    def __len__(self):
        return len(self.items)

    def _load_fused_features(
        self, seq: str, poc: int, coords: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Ładuje cechy VVC z fused_maps_pocX.pt i wycina fragment
        odpowiadający chunkowi.

        Zwraca: tensor (6, chunk_h, chunk_w) float32
        """
        if self.fused_root is None:
            return torch.zeros(
                6, self.chunk_h, self.chunk_w, dtype=torch.float32
            )

        fmap_path = os.path.join(
            self.fused_root, seq, "fused_maps", f"fused_maps_poc{poc}.pt"
        )
        
        if not os.path.isfile(fmap_path):
            return torch.zeros(
                6, self.chunk_h, self.chunk_w, dtype=torch.float32
            )

        d = torch.load(fmap_path, map_location="cpu")
        
        # rozmiar ramki z pierwszego klucza
        sample_key = next(iter(d.keys()))
        h, w = d[sample_key].shape[-2:]

        # stride jak w split_to_chunks_pt
        stride_x = self.chunk_w - 2 * self.border
        stride_y = self.chunk_h - 2 * self.border

        chs: List[np.ndarray] = []

        # 1) y_ac_energy (zakres spory – możemy lekko normalizować np. / 2048)
        if "y_ac_energy" in d:
            fm = d["y_ac_energy"].cpu().numpy().astype(np.float32) / 2048.0
            fm = _upsample_fmaps_to_padded_frame(
                fm, h, w, stride_y, stride_x, self.border, cv2.INTER_LINEAR
            )
            chs.append(fm)

        # 2) y_nz_density (0..1)
        if "y_nz_density" in d:
            fm = d["y_nz_density"].cpu().numpy().astype(np.float32)
            fm = _upsample_fmaps_to_padded_frame(
                fm, h, w, stride_y, stride_x, self.border, cv2.INTER_LINEAR
            )
            chs.append(fm)

        # 3) y_dc – składowa stała, normalizacja np. / 4096
        if "y_dc" in d:
            fm = d["y_dc"].cpu().numpy().astype(np.float32) / 4096.0
            fm = _upsample_fmaps_to_padded_frame(
                fm, h, w, stride_y, stride_x, self.border, cv2.INTER_LINEAR
            )
            chs.append(fm)

        # 4) boundary_bin – 0/1, bez dodatkowej normalizacji
        if "boundary_bin" in d:
            fm = d["boundary_bin"].cpu().numpy().astype(np.float32)
            fm = _upsample_fmaps_to_padded_frame(
                fm, h, w, stride_y, stride_x, self.border, cv2.INTER_NEAREST
            )
            chs.append(fm)

        # 5) boundary_weight (0..6) → /6
        if "boundary_weight" in d:
            fm = d["boundary_weight"].cpu().numpy().astype(np.float32) / 6.0
            fm = _upsample_fmaps_to_padded_frame(
                fm, h, w, stride_y, stride_x, self.border, cv2.INTER_LINEAR
            )
            chs.append(fm)

        # 6) size_map_norm (0.031..0.5) – już znormalizowane
        if "size_map_norm" in d:
            fm = d["size_map_norm"].cpu().numpy().astype(np.float32)
            fm = _upsample_fmaps_to_padded_frame(
                fm, h, w, stride_y, stride_x, self.border, cv2.INTER_NEAREST
            )
            chs.append(fm)

        if not chs:
            return torch.zeros(
                6, self.chunk_h, self.chunk_w, dtype=torch.float32
            )

        y, x = coords
        Hc, Wc = self.chunk_h, self.chunk_w
        crop = [fm[y : y + Hc, x : x + Wc] for fm in chs]

        feat = np.stack(crop, axis=0).astype(np.float32)
        return torch.from_numpy(feat)

    def __getitem__(self, idx: int):
        seq, poc, idx_chunk = self.items[idx]

        dec_path = self.dec_index[(seq, poc)]
        orig_path = self.orig_index[(seq, poc)]

        d_dec = _load_pt(dec_path)
        d_orig = _load_pt(orig_path)

        # (N, 3, H, W) uint8 -> float32 [0,1]
        dec = d_dec["chunks"][idx_chunk].to(torch.float32) / 255.0
        orig = d_orig["chunks"][idx_chunk].to(torch.float32) / 255.0

        # metadane sekwencji (nie chunk-specyficzne)
        meta = self.seq_meta_cache[seq]
        meta_t = _norm_metadata(meta)

        # koordynaty chunku w ramce z bordurem
        coords = tuple(
            map(int, d_dec["coords"][idx_chunk].tolist())
        )  # (y, x)

        if self.fused_root:
            vvc_feat = self._load_fused_features(seq, poc, coords)
        else:
            vvc_feat = torch.zeros(
                6, self.chunk_h, self.chunk_w, dtype=torch.float32
            )

        # Zgodność z TrainerModule/custom_collate:
        # (chunks, orig_chunks, metadata, chunk_objs, vvc_features)
        return dec, orig, meta_t, None, vvc_feat
