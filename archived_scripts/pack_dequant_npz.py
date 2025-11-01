#!/usr/bin/env python3
# bin/pack_dequant_npz.py
# Konwersja par (dequant_poc_XX.csv, dequant_poc_XX.bin) -> dequant_poc_XX.npz
# Opcjonalnie kasuje oryginały po udanym zapisie (--delete).

import argparse, csv, os, sys, glob
from pathlib import Path
import numpy as np

def load_meta(csv_path: Path) -> np.ndarray:
    rows = []
    with open(csv_path, "r", newline="") as fh:
        r = csv.reader(fh)
        header = next(r)
        # Akceptuj nagłówek: poc,comp,x,y,w,h,offset,length
        for line in r:
            if not line or line[0].startswith("#"):
                continue
            poc, comp, x, y, w, h, off, ln = map(int, line)
            rows.append([poc, comp, x, y, w, h, off, ln])
    if not rows:
        return np.empty((0,8), dtype=np.int32)
    return np.array(rows, dtype=np.int32)

def try_downcast_int16(arr: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(arr)) if arr.size else 0
    if max_abs <= 32767:
        return arr.astype(np.int16, copy=False)
    return arr  # zostaw int32

def pack_one(poc_csv: Path, delete: bool=False, downcast: bool=True) -> None:
    poc = poc_csv.stem.split("_")[-1]  # 'dequant_poc_XX' -> 'XX'
    poc_bin = poc_csv.with_suffix(".bin")
    out_npz = poc_csv.with_suffix(".npz")

    if not poc_bin.exists():
        print(f"[WARN] Brak BIN dla {poc_csv.name}, pomijam.")
        return

    meta = load_meta(poc_csv)
    # wczytaj cały bin jako int32 (little-endian)
    coeffs = np.fromfile(poc_bin, dtype="<i4")

    if meta.size:
        if np.any(meta[:, 6] % 4) or np.any(meta[:, 7] % 4):
            print(f"[WARN] {poc_csv.name}: offset/length niepodzielne przez 4 – zaokrąglę w dół.")
        meta[:, 6] //= 4
        meta[:, 7] //= 4

    if downcast:
        coeffs = try_downcast_int16(coeffs)

    np.savez_compressed(out_npz, coeffs=coeffs, meta=meta)
    out_size = out_npz.stat().st_size
    bin_size = poc_bin.stat().st_size
    csv_size = poc_csv.stat().st_size
    print(f"[OK] {out_npz.name}  ({(bin_size+csv_size)/1e6:.1f}MB -> {out_size/1e6:.1f}MB)")

    if delete:
        try:
            poc_bin.unlink()
            poc_csv.unlink()
        except Exception as e:
            print(f"[WARN] Nie udało się usunąć oryginałów: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=".", help="Folder z dequant_poc_*.csv/bin")
    ap.add_argument("--keep-int32", action="store_true", help="Nie próbuj rzutować do int16")
    ap.add_argument("--delete", action="store_true", help="Usuń .csv i .bin po konwersji")
    args = ap.parse_args()

    root = Path(args.folder)
    csvs = sorted(root.glob("dequant_poc_*.csv"))
    if not csvs:
        print("Brak plików dequant_poc_*.csv w:", root)
        sys.exit(1)

    for poc_csv in csvs:
        pack_one(poc_csv, delete=args.delete, downcast=(not args.keep_int32))

if __name__ == "__main__":
    main()
