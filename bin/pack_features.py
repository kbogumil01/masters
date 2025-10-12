#!/usr/bin/env python3
"""
pack_features.py

Zbiera:
- metadata z .info (rozmiar, format),
- mapy Depth/QT/BTMT z block_stats.csv,
- indeksy współczynników z dequant_poc_*.csv (+ wskazania do .bin),

i zapisuje per-POC paczki .npz do katalogu out/.

Użycie:
python pack_features.py \
  --decoded-dir videos/decoded \
  --recon videos/decoded/NAME.yuv \
  --info videos/data/NAME.info \
  --out out_npz
"""

'''
Użycie (przykład dla jednej sekwencji):
cd /home/karol/mgr/new_PC

# nazwa bazowa BEZ rozszerzenia – dopasuj do swoich plików
NAME="VTM_test"   # przykład: jeśli masz videos/decoded/VTM_test.yuv

python pack_features.py \
  --decoded-dir videos/decoded \
  --recon videos/decoded/${NAME}.yuv \
  --info videos/data/${NAME}.info \
  --out videos/npz/${NAME}
'''

import re, argparse, numpy as np
from pathlib import Path
import json

_rx_meta = re.compile(r"(\w+)\s*:\s*(.+)")

def read_info(info_path: Path):
    meta = {}
    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = _rx_meta.search(ln)
            if not m: continue
            k, v = m.groups()
            meta[k] = v.strip()
    # Spróbuj wyłuskać rozmiar
    # Szukamy np. "Width: 1280" i "Height: 720" lub "1280x720"
    W = None; H = None
    for k in ("Width", "width"):
        if k in meta:
            try: W = int(re.findall(r"\d+", meta[k])[0])
            except: pass
    for k in ("Height", "height"):
        if k in meta:
            try: H = int(re.findall(r"\d+", meta[k])[0])
            except: pass
    if (W is None or H is None) and "DisplayAspectRatio" in meta:
        # nic
        pass
    # awaryjnie poszukaj w całym pliku wpisu 1234x567
    if W is None or H is None:
        with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        m = re.search(r"(\d+)\s*[xX]\s*(\d+)", txt)
        if m:
            W, H = int(m.group(1)), int(m.group(2))
    # domyślnie 8-bit 420
    chroma = meta.get("ChromaSubsampling", "4:2:0")
    bitdepth = int(re.findall(r"\d+", meta.get("BitDepth", "8"))[0])
    if W is None or H is None:
        raise RuntimeError(f"Nie udało się ustalić rozmiaru z {info_path}")
    return {"width": W, "height": H, "bitdepth": bitdepth, "chroma": chroma}

line_rx = re.compile(
    r"BlockStat;(\d+); *(\d+); *(\d+); *(\d+); *(\d+); *([^;]+); *([^\n\r]+)"
)
def parse_block_line(line):
    m = line_rx.match(line)
    if not m:
        return None
    poc, x, y, w, h, name, val = m.groups()
    return (int(poc), int(x), int(y), int(w), int(h), name.strip(), val.strip())

def build_maps(block_csv: Path, poc: int, H: int, W: int):
    depth   = np.full((H, W), -1, np.int8)
    qt_only = np.zeros_like(depth, dtype=np.uint8)
    bt_mt   = np.zeros_like(depth, dtype=np.uint8)

    with open(block_csv, "r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            rec = parse_block_line(ln)
            if not rec or rec[0] != poc:
                continue
            _, x, y, w, h, name, val = rec
            ys, xs = slice(y, y + h), slice(x, x + w)

            if name == "Depth":
                depth[ys, xs] = int(val)
            elif name == "QT_Depth" and int(val) > 0:
                qt_only[ys, xs] = 1
            elif name in ("BT_Depth", "MT_Depth") and int(val) > 0:
                bt_mt[ys, xs] = 1
    return depth, qt_only, bt_mt

def read_dequant_index(csv_path: Path):
    """Wczytuje dequant_poc_*.csv -> lista rekordów (comp,x,y,w,h,offset,length)"""
    out = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()  # poc,comp,x,y,w,h,offset,length
        for ln in f:
            poc, comp, x, y, w, h, off, leng = ln.strip().split(",")
            out.append({
                "comp": int(comp),
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "offset": int(off), "length": int(leng)
            })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoded-dir", required=True, help="katalog z danymi po dekodowaniu (block_stats.csv, dequant_*, recon)")
    ap.add_argument("--recon", required=True, help="ścieżka do recon.yuv (w tym samym katalogu co block_stats.csv)")
    ap.add_argument("--info", required=True, help="ścieżka do pliku .info odpowiadającego tej sekwencji")
    ap.add_argument("--out", required=True, help="katalog wyjściowy na paczki .npz")
    args = ap.parse_args()

    decoded = Path(args.decoded_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    info = read_info(Path(args.info))
    W, H = info["width"], info["height"]

    block_csv = decoded / "block_stats.csv"
    # Zbierz dostępne POC z plików dequant
    dq_csvs = sorted(decoded.glob("dequant_poc_*.csv"))
    present_pocs = []
    for p in dq_csvs:
        m = re.search(r"dequant_poc_(\d+)\.csv", p.name)
        if m:
            present_pocs.append(int(m.group(1)))
    present_pocs = sorted(set(present_pocs))

    # UWAGA: współczynniki są tylko dla POC z CBF=1 (to normalne)
    for poc in present_pocs:
        depth, qt, bt = build_maps(block_csv, poc, H, W)
        dq_index_csv = decoded / f"dequant_poc_{poc}.csv"
        dq_bin       = decoded / f"dequant_poc_{poc}.bin"
        idx = read_dequant_index(dq_index_csv)

        payload = {
            "poc": poc,
            "width": W,
            "height": H,
            "bitdepth": info["bitdepth"],
            "chroma": info["chroma"],
            "recon_path": str(Path(args.recon).resolve()),  # globalna ścieżka do YUV
            "dequant_bin_path": str(dq_bin.resolve()),
            # uwaga: depth/qt/bt zapisujemy jako tablice
        }

        np.savez_compressed(
            outdir / f"frame_{poc:04d}.npz",
            **payload,
            depth=depth.astype(np.int8),
            qt_mask=qt.astype(np.uint8),
            btmt_mask=bt.astype(np.uint8),
            dequant_index=np.array(
                [[r["comp"], r["x"], r["y"], r["w"], r["h"], r["offset"], r["length"]]
                 for r in idx],
                dtype=np.int32
            )
        )
        print(f"✔ Zapisano {outdir / f'frame_{poc:04d}.npz'}")

if __name__ == "__main__":
    main()
