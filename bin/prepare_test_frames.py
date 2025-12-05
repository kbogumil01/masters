#!/usr/bin/env python3
import os
import re
import math
import argparse
from pathlib import Path
import glob
import multiprocessing
import traceback

import numpy as np
import cv2
import torch
from tqdm import tqdm


''''
python3 bin/prepare_test_frames.py \
    /ścieżka/do/test_decoded \
    /ścieżka/do/test_data \
    /mnt/d/data_mgr/test_frames_pt \
    --workers 4
'''

# ====== Regexy (te same co w split_to_chunks) ======
DECODED_DIR_RE = re.compile(
    r"^(?P<name>.+)_(?P<profile>AI|RA)_QP(?P<qp>\d+)_ALF(?P<alf>[01])_DB(?P<db>[01])_SAO(?P<sao>[01])$"
)
DECODE_LOG_SIZE_RE = re.compile(r"CS\.lumaSize=(\d+)x(\d+)")
ORIG_INFO_RE = re.compile(r"^\s*(?P<key>[^:]+)\s*:\s*(?P<val>.+)\s*$")

# ====== Funkcje pomocnicze ======

def read_info_file(info_path: str):
    width = height = frames = None
    def _to_int(s: str):
        m = re.search(r"(\d+)", s.replace(" ", ""))
        return int(m.group(1)) if m else None

    try:
        with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = ORIG_INFO_RE.match(line)
                if not m: continue
                k, v = m.group("key").lower(), m.group("val").strip()
                if k.startswith("width"): width = _to_int(v)
                elif k.startswith("height"): height = _to_int(v)
                elif k.startswith("frame count"): frames = _to_int(v)
    except Exception:
        pass
    return width, height, frames

def parse_decode_log(log_path: str):
    width = height = None
    frames = 0
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = DECODE_LOG_SIZE_RE.search(line)
                if m:
                    width, height = int(m.group(1)), int(m.group(2))
                    frames += 1
    except Exception:
        pass
    return width, height, frames

def upsample_uv(frame_buffer: np.ndarray, width: int, height: int) -> np.ndarray:
    i = width * height
    Y = frame_buffer[0:i].reshape((height, width))
    uv_size = width * height // 4
    V = frame_buffer[i : i + uv_size].reshape((height // 2, width // 2))
    V = cv2.resize(V, (width, height))
    U = frame_buffer[i + uv_size : i + 2 * uv_size].reshape((height // 2, width // 2))
    U = cv2.resize(U, (width, height))
    return np.dstack([Y, U, V])

def find_original_info_by_name(data_root: str, name: str):
    cand1 = os.path.join(data_root, name + ".y4m.info")
    cand2 = os.path.join(data_root, name + ".info")
    return cand1 if os.path.isfile(cand1) else (cand2 if os.path.isfile(cand2) else None)

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

def get_pad_size(dim, divisor=64):
    """Zwraca nowy wymiar będący najbliższą (w górę) wielokrotnością divisor."""
    return math.ceil(dim / divisor) * divisor

# === Logika przetwarzania ===

def save_full_frames_for_decoded(seq, data_root, out_root, fused_override_root=None):
    seq_dir_name = seq["dir_name"]
    
    # Skip if exists check
    out_seq_dir = os.path.join(out_root, seq_dir_name)
    if os.path.isdir(out_seq_dir):
        existing = glob.glob(os.path.join(out_seq_dir, "frame_*.pt"))
        if len(existing) > 10:
            # print(f"⏩ Skipping {seq_dir_name} (already exists)")
            return

    seq_full_path = seq["dir_path"]
    recon_path = seq["recon_path"]

    # 1. Rozdzielczość
    width = height = frames_log = None
    if seq["decode_log"]:
        width, height, frames_log = parse_decode_log(seq["decode_log"])
    if width is None:
        orig_info = find_original_info_by_name(data_root, seq["name"])
        if orig_info:
            width, height, frames_log = read_info_file(orig_info)
    if width is None:
         m = re.search(r"_(\d+)x(\d+)_", seq_dir_name)
         if m: width, height = int(m.group(1)), int(m.group(2))
    
    if width is None or height is None:
        raise ValueError(f"Unknown resolution for {seq_dir_name}")

    # 2. Bitdepth
    file_size = os.path.getsize(recon_path)
    wh = width * height
    bytes_per_frame_8  = (wh * 3) // 2
    bytes_per_frame_16 =  wh * 3
    frames_8  = file_size // bytes_per_frame_8 if bytes_per_frame_8 > 0 else 0
    frames_16 = file_size // bytes_per_frame_16 if bytes_per_frame_16 > 0 else 0
    
    if frames_log and frames_log > 0:
        use_16bit = abs(frames_16 - frames_log) <= abs(frames_8 - frames_log)
    else:
        use_16bit = frames_16 >= frames_8

    frames = min(frames_log, frames_16 if use_16bit else frames_8) if frames_log else (frames_16 if use_16bit else frames_8)

    # 3. Load buffer
    nh = (height * 3) // 2
    with open(recon_path, "rb") as f:
        if use_16bit:
            raw = np.frombuffer(f.read(frames * bytes_per_frame_16), dtype=np.uint16)
            raw = raw[:frames*nh*width].reshape((frames, nh*width))
            buff = np.round(raw / 4).astype(np.uint8)
        else:
            raw = np.frombuffer(f.read(frames * bytes_per_frame_8), dtype=np.uint8)
            buff = raw[:frames*nh*width].reshape((frames, nh*width))

    os.makedirs(out_seq_dir, exist_ok=True)

    # 4. Fused maps location
    if fused_override_root:
        maps_search_path = os.path.join(fused_override_root, seq_dir_name)
    else:
        maps_search_path = seq_full_path
    
    has_maps = os.path.isdir(os.path.join(maps_search_path, "fused_maps"))

    # 5. Calculate Padding
    pad_w = get_pad_size(width, 64)
    pad_h = get_pad_size(height, 64)

    seq_meta = {
        "name": seq["name"],
        "profile": seq["profile"],
        "qp": int(seq["qp"]),
        "alf": int(seq["alf"]),
        "db": int(seq["db"]),
        "sao": int(seq["sao"]),
        "orig_width": width,
        "orig_height": height,
        "padded_width": pad_w,
        "padded_height": pad_h
    }

    # 6. Process frames
    for poc in range(frames):
        # a) Image padding
        frame = buff[poc]
        frame = upsample_uv(frame, width, height) # H, W, 3
        frame_padded = cv2.copyMakeBorder(
            frame,
            0, pad_h - height,
            0, pad_w - width,
            cv2.BORDER_CONSTANT, value=0
        )

        # b) Features padding
        feats_padded = None
        if has_maps:
            map_path = os.path.join(maps_search_path, "fused_maps", f"fused_maps_poc{poc}.pt")
            if os.path.isfile(map_path):
                try:
                    d = torch.load(map_path, map_location="cpu")
                    feat_stack = []
                    # Ważne: Stała kolejność cech!
                    for k in ["y_ac_energy", "y_nz_density", "y_dc", "boundary_bin", "boundary_weight", "size_map_norm"]:
                        if k in d:
                            arr = d[k].float().numpy()
                            arr_pad = cv2.copyMakeBorder(
                                arr, 0, pad_h - height, 0, pad_w - width,
                                cv2.BORDER_CONSTANT, value=0.0
                            )
                            feat_stack.append(arr_pad)
                        else:
                            feat_stack.append(np.zeros((pad_h, pad_w), dtype=np.float32))
                    
                    feats_padded = torch.from_numpy(np.stack(feat_stack, axis=0)).half()
                except Exception:
                    pass

        # c) Save
        chunk_tensor = torch.from_numpy(frame_padded.transpose(2, 0, 1).copy()) # CHW

        data_dict = {
            "seq_meta": seq_meta,
            "poc": poc,
            "chunk": chunk_tensor,
            "is_intra": (1 if seq["profile"] == "AI" else 0)
        }
        if feats_padded is not None:
            data_dict["vvc_features"] = feats_padded
        
        torch.save(data_dict, os.path.join(out_seq_dir, f"frame_poc{poc:03d}.pt"))


def save_full_frames_for_original(orig, out_root):
    name = orig["name"]
    
    out_seq_dir = os.path.join(out_root, name)
    if os.path.isdir(out_seq_dir):
        existing = glob.glob(os.path.join(out_seq_dir, "frame_*.pt"))
        if len(existing) > 10:
            return

    yuv_path = orig["yuv_path"]
    info_path = orig["info_path"]

    width, height, frames = read_info_file(info_path)
    if frames is None: frames = 64
    
    # Detekcja paddingu dla oryginałów (taka sama logika jak dla decoded)
    pad_w = get_pad_size(width, 64)
    pad_h = get_pad_size(height, 64)

    nh = height * 3 // 2
    with open(yuv_path, "rb") as f:
        buff = np.frombuffer(f.read(), dtype=np.uint8)
    
    vals_per_frame = nh * width
    total = buff.size // vals_per_frame
    frames = min(frames, total)
    buff = np.resize(buff, (frames, nh * width))

    os.makedirs(out_seq_dir, exist_ok=True)

    seq_meta = {
        "name": name,
        "profile": "orig",
        "qp": 0, "alf": 0, "db": 0, "sao": 0,
        "orig_width": width,
        "orig_height": height,
        "padded_width": pad_w,
        "padded_height": pad_h
    }

    for poc in range(frames):
        frame = buff[poc]
        frame = upsample_uv(frame, width, height)
        frame_padded = cv2.copyMakeBorder(
            frame,
            0, pad_h - height,
            0, pad_w - width,
            cv2.BORDER_CONSTANT, value=0
        )
        
        chunk_tensor = torch.from_numpy(frame_padded.transpose(2, 0, 1).copy())
        
        data_dict = {
            "seq_meta": seq_meta,
            "poc": poc,
            "chunk": chunk_tensor,
            "is_intra": 0
        }
        torch.save(data_dict, os.path.join(out_seq_dir, f"frame_poc{poc:03d}.pt"))

# === Workery ===

def _worker_decoded(args):
    seq, data_root, out_root, fused_override_root = args
    try:
        save_full_frames_for_decoded(seq, data_root, out_root, fused_override_root)
        return True
    except Exception as e:
        print(f"[ERR] Failed {seq['dir_name']}: {e}")
        traceback.print_exc()
        return False

def _worker_original(args):
    orig, out_root = args
    try:
        save_full_frames_for_original(orig, out_root)
        return True
    except Exception as e:
        print(f"[ERR] Failed {orig['name']}: {e}")
        traceback.print_exc()
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decoded_or_orig_root", help="Folder z sekwencjami")
    ap.add_argument("data_root", help="Folder z .info")
    ap.add_argument("out_root", help="Folder wynikowy dla pełnych klatek .pt")
    ap.add_argument("--fused_maps_root", help="Opcjonalny override dla map", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    src = args.decoded_or_orig_root
    data_root = args.data_root
    out_root = args.out_root
    fused_root = args.fused_maps_root

    os.makedirs(out_root, exist_ok=True)

    if args.workers is None:
        args.workers = max(1, multiprocessing.cpu_count() // 2)

    decoded = list(iter_decoded_sequences(src))
    if decoded:
        print(f"📂 Detected decoded layout: {len(decoded)} sequences")
        print(f"🚀 Processing full frames with {args.workers} workers...")
        tasks = [(seq, data_root, out_root, fused_root) for seq in decoded]
        with multiprocessing.Pool(args.workers) as p:
            list(tqdm(p.imap_unordered(_worker_decoded, tasks), total=len(tasks)))
        print("✅ Done decoded.")
        return

    originals = list(iter_original_sequences(src))
    if not originals:
        print("❌ No sequences found.")
        return

    print(f"📂 Detected original layout: {len(originals)} sequences")
    print(f"🚀 Processing full frames with {args.workers} workers...")
    tasks = [(orig, out_root) for orig in originals]
    with multiprocessing.Pool(args.workers) as p:
        list(tqdm(p.imap_unordered(_worker_original, tasks), total=len(tasks)))
    print("✅ Done originals.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()