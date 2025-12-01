#!/usr/bin/env python3
import os
import re
import math
import argparse
from pathlib import Path
import glob

import numpy as np
import cv2
import torch
from tqdm import tqdm

# ====== parse Piotrowe nazwy (zdekodowane) ======
DECODED_DIR_RE = re.compile(
    r"^(?P<name>.+)_(?P<profile>AI|RA)_QP(?P<qp>\d+)_ALF(?P<alf>[01])_DB(?P<db>[01])_SAO(?P<sao>[01])$"
)

# ====== parse wiersza z decode.log ======
DECODE_LOG_SIZE_RE = re.compile(r"CS\.lumaSize=(\d+)x(\d+)")

# ====== parse Piotrowe nazwy oryginałów ======
ORIG_INFO_RE = re.compile(r"^\s*(?P<key>[^:]+)\s*:\s*(?P<val>.+)\s*$")


def read_info_file(info_path: str):
    """czyta .info Piotra i wyciąga width/height/frame count"""
    width = height = frames = None
    def _to_int(s: str):
        m = re.search(r"(\d+)", s.replace(" ", ""))
        return int(m.group(1)) if m else None

    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ORIG_INFO_RE.match(line)
            if not m: continue
            k, v = m.group("key").lower(), m.group("val").strip()
            if k.startswith("width"): width = _to_int(v)
            elif k.startswith("height"): height = _to_int(v)
            elif k.startswith("frame count"): frames = _to_int(v)
    return width, height, frames

def parse_decode_log(log_path: str):
    """Parsuje plik decode.log w stylu VTM"""
    width = height = None
    frames = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = DECODE_LOG_SIZE_RE.search(line)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
                frames += 1
    return width, height, frames

def upsample_uv(frame_buffer: np.ndarray, width: int, height: int) -> np.ndarray:
    """YUV420 -> 3 kanały (Y, U, V)"""
    i = width * height
    Y = frame_buffer[0:i].reshape((height, width))
    uv_size = width * height // 4
    V = frame_buffer[i : i + uv_size].reshape((height // 2, width // 2))
    V = cv2.resize(V, (width, height))
    U = frame_buffer[i + uv_size : i + 2 * uv_size].reshape((height // 2, width // 2))
    U = cv2.resize(U, (width, height))
    return np.dstack([Y, U, V])

def iter_decoded_sequences(decoded_root: str):
    if not os.path.isdir(decoded_root): return
    for entry in sorted(os.listdir(decoded_root)):
        full = os.path.join(decoded_root, entry)
        if not os.path.isdir(full): continue
        m = DECODED_DIR_RE.match(entry)
        if not m: continue
        recon_path = os.path.join(full, "recon.yuv")
        if not os.path.isfile(recon_path): continue
        decode_log = os.path.join(full, "decode.log")
        yield {
            "dir_name": entry, "dir_path": full, "recon_path": recon_path,
            "decode_log": decode_log if os.path.isfile(decode_log) else None,
            **m.groupdict(),
        }

def iter_original_sequences(data_root: str):
    if not os.path.isdir(data_root): return
    for entry in sorted(os.listdir(data_root)):
        if not entry.endswith(".yuv"): continue
        base = entry[:-4]
        info1 = os.path.join(data_root, base + ".y4m.info")
        info2 = os.path.join(data_root, base + ".info")
        info_path = info1 if os.path.isfile(info1) else (info2 if os.path.isfile(info2) else None)
        if not info_path: continue
        yield {"name": base, "yuv_path": os.path.join(data_root, entry), "info_path": info_path}

def find_original_info_by_name(data_root: str, name: str):
    cand1 = os.path.join(data_root, name + ".y4m.info")
    cand2 = os.path.join(data_root, name + ".info")
    return cand1 if os.path.isfile(cand1) else (cand2 if os.path.isfile(cand2) else None)

def load_and_pad_fused_maps(seq_path, poc, border, chunk_h, chunk_w):
    """
    Ładuje fused_maps_pocX.pt ze wskazanego folderu sekwencji.
    """
    # Ścieżka: seq_path/fused_maps/fused_maps_poc{poc}.pt
    map_path = os.path.join(seq_path, "fused_maps", f"fused_maps_poc{poc}.pt")
    
    if not os.path.isfile(map_path):
        return None

    try:
        d = torch.load(map_path, map_location="cpu")
    except Exception as e:
        print(f"[ERR] Failed to load {map_path}: {e}")
        return None

    # Pobierz wymiary z dowolnej mapy
    k0 = next(iter(d.keys()))
    H, W = d[k0].shape[-2:]
    
    stride_x = chunk_w - 2 * border
    stride_y = chunk_h - 2 * border

    padded_maps = {}
    keys_to_keep = ["y_ac_energy", "y_nz_density", "y_dc", "boundary_bin", "boundary_weight", "size_map_norm"]

    for k in keys_to_keep:
        if k not in d: continue
        
        arr = d[k].float().numpy() # float32
        
        # Padding BORDER_CONSTANT=0
        pad_h = 2 * border + ((-H) % stride_y)
        pad_w = 2 * border + ((-W) % stride_x)
        
        arr_padded = cv2.copyMakeBorder(
            arr,
            border, pad_h,
            border, pad_w,
            cv2.BORDER_CONSTANT, value=0.0
        )
        padded_maps[k] = arr_padded

    return padded_maps

def save_chunks_for_decoded(seq, data_root, out_root, fused_override_root=None, chunk_w=132, chunk_h=132, border=2):
    seq_dir_name = seq["dir_name"]
    # === NOWOŚĆ: Pominąć, jeśli już zrobione ===
    out_seq_dir = os.path.join(out_root, seq_dir_name)
    if os.path.isdir(out_seq_dir):
        # Sprawdzamy, czy są tam jakiekolwiek chunki .pt
        existing_chunks = glob.glob(os.path.join(out_seq_dir, "chunk_*.pt"))
        if len(existing_chunks) > 10: # Zakładamy, że jak jest >10 plików, to sekwencja jest gotowa
            # print(f"⏩ Skipping {seq_dir_name} (already exists)")
            return
    # ===========================================
    seq_full_path = seq["dir_path"]
    recon_path = seq["recon_path"]

    # 1. Ustalanie rozdzielczości
    width = height = frames_log = None
    if seq["decode_log"] is not None:
        try:
            width, height, frames_log = parse_decode_log(seq["decode_log"])
        except: pass
    if width is None:
        orig_info = find_original_info_by_name(data_root, seq["name"])
        if orig_info:
            width, height, frames_log = read_info_file(orig_info)
    if width is None:
         m = re.search(r"_(\d+)x(\d+)_", seq_dir_name)
         if m: width, height = int(m.group(1)), int(m.group(2))

    # 2. Wykrywanie bitdepth
    file_size = os.path.getsize(recon_path)
    wh = width * height
    bytes_per_frame_8  = (wh * 3) // 2
    bytes_per_frame_16 =  wh * 3
    frames_8  = file_size // bytes_per_frame_8  if bytes_per_frame_8  > 0 else 0
    frames_16 = file_size // bytes_per_frame_16 if bytes_per_frame_16 > 0 else 0

    if frames_log and frames_log > 0:
        use_16bit = abs(frames_16 - frames_log) <= abs(frames_8 - frames_log)
    else:
        use_16bit = frames_16 >= frames_8

    frames = min(frames_log, frames_16 if use_16bit else frames_8) if frames_log else (frames_16 if use_16bit else frames_8)
    
    # 3. Wczytanie bufora
    nh = (height * 3) // 2
    if use_16bit:
        with open(recon_path, "rb") as f:
            buf16 = np.frombuffer(f.read(frames * bytes_per_frame_16), dtype=np.uint16)
        buf16 = buf16[:frames * nh * width].reshape((frames, nh * width))
        buff = np.round(buf16 / 4).astype(np.uint8)
    else:
        with open(recon_path, "rb") as f:
            buf8 = np.frombuffer(f.read(frames * bytes_per_frame_8), dtype=np.uint8)
        buff = buf8[:frames * nh * width].reshape((frames, nh * width))

    # 4. Chunkowanie
    stride_x = chunk_w - 2 * border
    stride_y = chunk_h - 2 * border
    hor_chunks = math.ceil(width / stride_x)
    ver_chunks = math.ceil(height / stride_y)

    out_seq_dir = os.path.join(out_root, seq_dir_name)
    os.makedirs(out_seq_dir, exist_ok=True)

    # Gdzie szukać map? Jeśli podano override (np. USB), to tam. Jeśli nie, to w folderze sekwencji (SSD).
    if fused_override_root:
        maps_search_path = os.path.join(fused_override_root, seq_dir_name)
    else:
        maps_search_path = seq_full_path

    print(f"Processing {seq_dir_name} ({frames} frames) -> Individual Chunks...")
    
    # Sprawdźmy czy mamy mapy
    has_maps = os.path.isdir(os.path.join(maps_search_path, "fused_maps"))
    if not has_maps:
        print(f"  [WARN] No fused_maps found in {maps_search_path}")

    for poc in range(frames):
        # a) Przetwarzanie obrazu
        frame = buff[poc]
        frame = upsample_uv(frame, width, height)
        frame_padded = cv2.copyMakeBorder(
            frame,
            border, 2 * border + ((-height) % stride_y),
            border, 2 * border + ((-width) % stride_x),
            cv2.BORDER_CONSTANT, value=0
        )

        # b) Ładowanie map cech (jeśli są)
        fused_data = None
        if has_maps:
            fused_data = load_and_pad_fused_maps(maps_search_path, poc, border, chunk_h, chunk_w)
        
        seq_meta = {
            "name": seq["name"],
            "profile": seq["profile"],
            "qp": int(seq["qp"]),
            "alf": int(seq["alf"]),
            "db": int(seq["db"]),
            "sao": int(seq["sao"]),
            "width": width,
            "height": height,
        }

        for hy in range(ver_chunks):
            y = hy * stride_y
            for hx in range(hor_chunks):
                x = hx * stride_x
                
                # Wycięcie obrazu
                ch_img = frame_padded[y:y+chunk_h, x:x+chunk_w, :]
                if ch_img.shape[:2] != (chunk_h, chunk_w):
                     pad = np.zeros((chunk_h, chunk_w, 3), dtype=np.uint8)
                     pad[:ch_img.shape[0], :ch_img.shape[1], :] = ch_img
                     ch_img = pad

                # Wycięcie cech
                chunk_feats = None
                if fused_data:
                    feat_stack = []
                    for k in ["y_ac_energy", "y_nz_density", "y_dc", "boundary_bin", "boundary_weight", "size_map_norm"]:
                        if k in fused_data:
                            fmap = fused_data[k]
                            ch_f = fmap[y:y+chunk_h, x:x+chunk_w]
                            if ch_f.shape != (chunk_h, chunk_w):
                                pad = np.zeros((chunk_h, chunk_w), dtype=np.float32)
                                pad[:ch_f.shape[0], :ch_f.shape[1]] = ch_f
                                ch_f = pad
                            feat_stack.append(ch_f)
                        else:
                            feat_stack.append(np.zeros((chunk_h, chunk_w), dtype=np.float32))
                    # Zapisujemy jako float16 (half) aby oszczędzić miejsce na NVMe
                    chunk_feats = torch.from_numpy(np.stack(feat_stack, axis=0)).half()

                # Flagi
                flags = 0
                if hx == 0: flags |= 1
                if hx == hor_chunks - 1: flags |= 2
                if hy == 0: flags |= 4
                if hy == ver_chunks - 1: flags |= 8
                
                chunk_tensor = torch.from_numpy(ch_img.transpose(2, 0, 1).copy()) # (3, H, W)
                
                data_dict = {
                    "seq_meta": seq_meta,
                    "poc": poc,
                    "coords": (y, x),
                    "corner_flags": flags,
                    "is_intra": (1 if seq["profile"] == "AI" else 0),
                    "chunk": chunk_tensor
                }
                if chunk_feats is not None:
                    data_dict["vvc_features"] = chunk_feats

                out_path = os.path.join(out_seq_dir, f"chunk_poc{poc:03d}_y{y:04d}_x{x:04d}.pt")
                torch.save(data_dict, out_path)

def save_chunks_for_original(orig, out_root, chunk_w=132, chunk_h=132, border=2):
    """
    orig: dict z iter_original_sequences
    zapisze: out_root/<name>/chunks_pocXXX.pt
    """
    name = orig["name"]
    yuv_path = orig["yuv_path"]
    info_path = orig["info_path"]

    width, height, frames = read_info_file(info_path)
    if frames is None: frames = 64
    nh = height * 3 // 2

    with open(yuv_path, "rb") as f:
        buff = np.frombuffer(f.read(), dtype=np.uint8)

    vals_per_frame = nh * width
    total_frames_in_file = buff.size // vals_per_frame
    frames = min(frames, total_frames_in_file)

    buff = np.resize(buff, (frames, nh * width))

    stride_x = chunk_w - 2 * border
    stride_y = chunk_h - 2 * border
    hor_chunks = math.ceil(width / stride_x)
    ver_chunks = math.ceil(height / stride_y)

    out_seq_dir = os.path.join(out_root, name)
    os.makedirs(out_seq_dir, exist_ok=True)
    
    print(f"Processing Original {name} ({frames} frames) -> Individual Chunks...")

    for poc in range(frames):
        frame = buff[poc]
        frame = upsample_uv(frame, width, height)
        frame_padded = cv2.copyMakeBorder(
            frame,
            border, 2 * border + ((-height) % stride_y),
            border, 2 * border + ((-width) % stride_x),
            cv2.BORDER_CONSTANT, value=0
        )

        seq_meta = {
            "name": name,
            "profile": "orig",
            "qp": 0, "alf": 0, "db": 0, "sao": 0,
            "width": width, "height": height
        }

        for hy in range(ver_chunks):
            y = hy * stride_y
            for hx in range(hor_chunks):
                x = hx * stride_x
                
                ch_img = frame_padded[y : y + chunk_h, x : x + chunk_w, :]
                if ch_img.shape[:2] != (chunk_h, chunk_w):
                    pad = np.zeros((chunk_h, chunk_w, 3), dtype=np.uint8)
                    pad[: ch_img.shape[0], : ch_img.shape[1], :] = ch_img
                    ch_img = pad

                flags = 0
                if hx == 0: flags |= 1
                if hx == hor_chunks - 1: flags |= 2
                if hy == 0: flags |= 4
                if hy == ver_chunks - 1: flags |= 8

                chunk_tensor = torch.from_numpy(ch_img.transpose(2, 0, 1).copy())
                
                data_dict = {
                    "seq_meta": seq_meta,
                    "poc": poc,
                    "coords": (y, x),
                    "corner_flags": flags,
                    "is_intra": 0,
                    "chunk": chunk_tensor
                }
                
                out_path = os.path.join(out_seq_dir, f"chunk_poc{poc:03d}_y{y:04d}_x{x:04d}.pt")
                torch.save(data_dict, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decoded_or_orig_root", help="Folder z sekwencjami (decoded_new lub data)")
    ap.add_argument("data_root", help="Folder z plikami .info")
    ap.add_argument("out_root", help="Folder wynikowy dla chunków .pt (NVMe)")
    ap.add_argument("--fused_maps_root", help="OPCJONALNIE: Ścieżka do map na USB. Jeśli puste, szuka wewnątrz decoded_or_orig_root", default=None)
    ap.add_argument("--chunk-w", type=int, default=132)
    ap.add_argument("--chunk-h", type=int, default=132)
    ap.add_argument("--border", type=int, default=2)
    args = ap.parse_args()

    src = args.decoded_or_orig_root
    data_root = args.data_root
    out_root = args.out_root
    fused_root = args.fused_maps_root

    os.makedirs(out_root, exist_ok=True)

    decoded = list(iter_decoded_sequences(src))
    if decoded:
        print(f"📂 Detected decoded layout: {len(decoded)} sequences")
        for seq in decoded:
            save_chunks_for_decoded(
                seq, data_root, out_root,
                fused_override_root=fused_root,
                chunk_w=args.chunk_w, chunk_h=args.chunk_h, border=args.border
            )
        return

    originals = list(iter_original_sequences(src))
    if not originals:
        print("❌ No sequences found.")
        return

    print(f"📂 Detected original layout: {len(originals)} sequences")
    for orig in originals:
        save_chunks_for_original(
            orig, out_root,
            chunk_w=args.chunk_w, chunk_h=args.chunk_h, border=args.border
        )

if __name__ == "__main__":
    main()