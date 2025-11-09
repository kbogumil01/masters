#!/usr/bin/env python3
import os
import re
import math
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
'''
python3 bin/split_to_chunks_pt.py \
    /mnt/d/data_mgr/decoded_new \ - folder z sekwencjami
    /mnt/d/data_mgr/data \ - folder with .info files
    /mnt/d/data_mgr/orig_chunks_pt - folder wyniokowy

'''

# ====== parse Piotrowe nazwy (zdekodowane) ======
# np. FourPeople_1280x720_60_AI_QP28_ALF0_DB0_SAO0
DECODED_DIR_RE = re.compile(
    r"^(?P<name>.+)_(?P<profile>AI|RA)_QP(?P<qp>\d+)_ALF(?P<alf>[01])_DB(?P<db>[01])_SAO(?P<sao>[01])$"
)

# ====== parse wiersza z decode.log ======
# [DEQDBG] POC=0  CS.lumaPos=(0,0)  CS.lumaSize=1920x1080
DECODE_LOG_SIZE_RE = re.compile(r"CS\.lumaSize=(\d+)x(\d+)")


# ====== parse Piotrowe nazwy oryginałów ======
# np. blue_sky_1080p25.yuv  + blue_sky_1080p25.y4m.info
ORIG_INFO_RE = re.compile(r"^\s*(?P<key>[^:]+)\s*:\s*(?P<val>.+)\s*$")


def read_info_file(info_path: str):
    """
    czyta .info Piotra i wyciąga width/height/frame count
    toleruje formaty typu '1 280 pixels'
    """
    width = height = frames = None

    # helper: wyciągnij pierwszą liczbę z napisu
    def _to_int(s: str):
        m = re.search(r"(\d+)", s.replace(" ", ""))
        return int(m.group(1)) if m else None

    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ORIG_INFO_RE.match(line)
            if not m:
                continue
            k = m.group("key").lower()
            v = m.group("val").strip()

            if k.startswith("width"):
                w = _to_int(v)
                if w is not None:
                    width = w
            elif k.startswith("height"):
                h = _to_int(v)
                if h is not None:
                    height = h
            elif k.startswith("frame count"):
                fr = _to_int(v)
                if fr is not None:
                    frames = fr

    return width, height, frames


def parse_decode_log(log_path: str):
    """
    Parsuje plik decode.log w stylu VTM:
    [DEQDBG] POC=0  ... CS.lumaSize=1920x1080
    Zwraca: (width, height, frames)
    """
    width = height = None
    frames = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = DECODE_LOG_SIZE_RE.search(line)
            if m:
                width = int(m.group(1))
                height = int(m.group(2))
                frames += 1
    if width is None or height is None:
        raise ValueError(f"Nie udało się odczytać rozdzielczości z {log_path}")
    return width, height, frames


def upsample_uv(frame_buffer: np.ndarray, width: int, height: int) -> np.ndarray:
    """Piotrowe YUV420 -> 3 kanały (Y, U, V)"""
    i = width * height
    Y = frame_buffer[0:i].reshape((height, width))

    uv_size = width * height // 4
    V = frame_buffer[i : i + uv_size].reshape((height // 2, width // 2))
    V = cv2.resize(V, (width, height))

    U = frame_buffer[i + uv_size : i + 2 * uv_size].reshape((height // 2, width // 2))
    U = cv2.resize(U, (width, height))

    return np.dstack([Y, U, V])  # (H, W, 3), uint8


def iter_decoded_sequences(decoded_root: str):
    """
    Znajdź sekwencje w stylu:
    decoded_root/<name>_AI_QP32_ALF1_DB0_SAO0/recon.yuv
    """
    if not os.path.isdir(decoded_root):
        return
    for entry in sorted(os.listdir(decoded_root)):
        full = os.path.join(decoded_root, entry)
        if not os.path.isdir(full):
            continue
        m = DECODED_DIR_RE.match(entry)
        if not m:
            continue
        recon_path = os.path.join(full, "recon.yuv")
        if not os.path.isfile(recon_path):
            continue
        decode_log = os.path.join(full, "decode.log")
        yield {
            "dir_name": entry,
            "dir_path": full,
            "recon_path": recon_path,
            "decode_log": decode_log if os.path.isfile(decode_log) else None,
            **m.groupdict(),
        }


def iter_original_sequences(data_root: str):
    """
    Znajdź oryginały w stylu:
    data_root/blue_sky_1080p25.yuv  + data_root/blue_sky_1080p25.y4m.info
    """
    if not os.path.isdir(data_root):
        return
    for entry in sorted(os.listdir(data_root)):
        if not entry.endswith(".yuv"):
            continue
        base = entry[:-4]  # bez .yuv
        info1 = os.path.join(data_root, base + ".y4m.info")
        info2 = os.path.join(data_root, base + ".info")
        info_path = None
        if os.path.isfile(info1):
            info_path = info1
        elif os.path.isfile(info2):
            info_path = info2
        if not info_path:
            # brak info -> pomijamy
            continue
        yield {
            "name": base,
            "yuv_path": os.path.join(data_root, entry),
            "info_path": info_path,
        }


def find_original_info_by_name(data_root: str, name: str):
    """
    Jeśli mamy decoded_new/FourPeople_1280x720_60_AI_QP...
    to name='FourPeople_1280x720_60'
    i tu próbujemy znaleźć:
    data_root/FourPeople_1280x720_60.y4m.info
    albo
    data_root/FourPeople_1280x720_60.info
    """
    cand1 = os.path.join(data_root, name + ".y4m.info")
    cand2 = os.path.join(data_root, name + ".info")
    if os.path.isfile(cand1):
        return cand1
    if os.path.isfile(cand2):
        return cand2
    return None


def save_chunks_for_decoded(seq, data_root, out_root, chunk_w=132, chunk_h=132, border=2):
    """
    seq: dict z iter_decoded_sequences
    zapisze: out_root/<dir_name>/chunks_pocXXX.pt
    Teraz robust: wykrywa 8-bit vs 16-bit (10/12-bit packed) po rozmiarze pliku.
    """
    seq_dir = seq["dir_name"]
    recon_path = seq["recon_path"]

    # 1) spróbuj decode.log
    width = height = frames_log = None
    if seq["decode_log"] is not None:
        try:
            width, height, frames_log = parse_decode_log(seq["decode_log"])
        except Exception as e:
            print(f"[WARN] decode.log parse failed for {seq_dir}: {e}")

    # 2) jeśli brak decode.log, spróbuj .info oryginału
    if width is None or height is None:
        orig_info = find_original_info_by_name(data_root, seq["name"])
        if orig_info is not None:
            w, h, fr = read_info_file(orig_info)
            width, height = w, h
            frames_log = fr

    # 3) ostateczny fallback z nazwy katalogu
    if width is None or height is None:
        m = re.search(r"_(\d+)x(\d+)_", seq_dir)
        if not m:
            raise RuntimeError(f"Nie umiem odczytać rozdzielczości dla {seq_dir}")
        width, height = int(m.group(1)), int(m.group(2))

    # --- Wykrywanie formatu recon.yuv po rozmiarze pliku ---
    file_size = os.path.getsize(recon_path)  # w bajtach
    wh = width * height
    bytes_per_frame_8  = (wh * 3) // 2                     # YUV420 8-bit
    bytes_per_frame_16 =  wh * 3                           # YUV420 10/12-bit zapisane w 16-bit

    frames_8  = file_size // bytes_per_frame_8  if bytes_per_frame_8  > 0 else 0
    frames_16 = file_size // bytes_per_frame_16 if bytes_per_frame_16 > 0 else 0

    # wybierz wariant najbliższy temu z loga; jeśli brak loga – wybierz większą liczbę klatek (bardziej konserwatywnie)
    if frames_log is not None and frames_log > 0:
        diff8  = abs(frames_8  - frames_log)
        diff16 = abs(frames_16 - frames_log)
        use_16bit = diff16 <= diff8
    else:
        use_16bit = frames_16 >= frames_8

    frames_file = frames_16 if use_16bit else frames_8
    if frames_log:
        # zabezpieczenie – jeżeli plik ma mniej klatek niż log, bierzemy tyle ile realnie jest
        frames = min(frames_log, frames_file)
    else:
        frames = frames_file

    if frames <= 0:
        raise RuntimeError(f"[ERR] Nie wykryto poprawnej liczby klatek dla {seq_dir} (size={file_size})")

    # --- Wczytanie bufora zgodnie z wariantem ---
    nh = (height * 3) // 2
    if use_16bit:
        # 10/12-bit w 16-bit → skala /4 → uint8
        with open(recon_path, "rb") as f:
            buf16 = np.frombuffer(f.read(frames * bytes_per_frame_16), dtype=np.uint16)
        # sanity: przytnij dokładnie do frames*nh*width elementów
        nelems = frames * nh * width
        buf16 = buf16[:nelems]
        buf16 = np.reshape(buf16, (frames, nh * width))
        buff = np.round(buf16 / 4).astype(np.uint8)
    else:
        # czysty 8-bit
        with open(recon_path, "rb") as f:
            buf8 = np.frombuffer(f.read(frames * bytes_per_frame_8), dtype=np.uint8)
        nelems = frames * nh * width
        buf8 = buf8[:nelems]
        buff = np.reshape(buf8, (frames, nh * width))

    # --- chunkowanie ---
    stride_x = chunk_w - 2 * border
    stride_y = chunk_h - 2 * border
    hor_chunks = math.ceil(width / stride_x)
    ver_chunks = math.ceil(height / stride_y)

    out_seq_dir = os.path.join(out_root, seq_dir)
    os.makedirs(out_seq_dir, exist_ok=True)

    for poc in range(frames):
        frame = buff[poc]
        frame = upsample_uv(frame, width, height)
        frame = cv2.copyMakeBorder(
            frame,
            border,
            2 * border + ((-height) % stride_y),
            border,
            2 * border + ((-width) % stride_x),
            cv2.BORDER_CONSTANT,
            value=0,
        )

        chunks, coords, corners, is_intra = [], [], [], []
        for hy in range(ver_chunks):
            y = hy * stride_y
            for hx in range(hor_chunks):
                x = hx * stride_x
                ch = frame[y:y+chunk_h, x:x+chunk_w, :]
                if ch.shape[:2] != (chunk_h, chunk_w):
                    pad = np.zeros((chunk_h, chunk_w, 3), dtype=np.uint8)
                    pad[:ch.shape[0], :ch.shape[1], :] = ch
                    ch = pad
                chunks.append(ch)

                flags = 0
                if hx == 0:               flags |= 1   # L
                if hx == hor_chunks - 1:  flags |= 2   # R
                if hy == 0:               flags |= 4   # U
                if hy == ver_chunks - 1:  flags |= 8   # B

                coords.append((y, x))
                corners.append(flags)
                is_intra.append(1 if seq["profile"] == "AI" else 0)

        chunks = np.transpose(np.stack(chunks, 0), (0, 3, 1, 2))  # (N,3,H,W)
        torch.save(
            {
                "seq_meta": {
                    "name": seq["name"],
                    "profile": seq["profile"],
                    "qp": int(seq["qp"]),
                    "alf": int(seq["alf"]),
                    "db": int(seq["db"]),
                    "sao": int(seq["sao"]),
                    "width": width,
                    "height": height,
                    "bitdepth": 10 if use_16bit else 8,
                },
                "poc": poc,
                "chunk_size": (chunk_h, chunk_w),
                "chunks": torch.from_numpy(chunks.astype(np.uint8)),
                "coords": torch.tensor(coords, dtype=torch.int32),
                "corner_flags": torch.tensor(corners, dtype=torch.uint8),
                "is_intra": torch.tensor(is_intra, dtype=torch.uint8),
            },
            os.path.join(out_seq_dir, f"chunks_poc{poc:03d}.pt"),
        )


def save_chunks_for_original(orig, out_root, chunk_w=132, chunk_h=132, border=2):
    """
    orig: dict z iter_original_sequences
    zapisze: out_root/<name>/chunks_pocXXX.pt
    """
    name = orig["name"]
    yuv_path = orig["yuv_path"]
    info_path = orig["info_path"]

    width, height, frames = read_info_file(info_path)
    if frames is None:
        frames = 64  # jak u Piotra
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

    for poc in range(frames):
        frame = buff[poc]
        frame = upsample_uv(frame, width, height)
        frame = cv2.copyMakeBorder(
            frame,
            border,
            2 * border + ((-height) % stride_y),
            border,
            2 * border + ((-width) % stride_x),
            cv2.BORDER_CONSTANT,
            value=0,
        )

        chunks = []
        coords = []
        corners = []
        is_intra = []

        for hy in range(ver_chunks):
            y = hy * stride_y
            for hx in range(hor_chunks):
                x = hx * stride_x
                ch = frame[y : y + chunk_h, x : x + chunk_w, :]
                if ch.shape[0] != chunk_h or ch.shape[1] != chunk_w:
                    pad = np.zeros((chunk_h, chunk_w, 3), dtype=np.uint8)
                    pad[: ch.shape[0], : ch.shape[1], :] = ch
                    ch = pad
                chunks.append(ch)

                flags = 0
                if hx == 0:
                    flags |= 1
                if hx == hor_chunks - 1:
                    flags |= 2
                if hy == 0:
                    flags |= 4
                if hy == ver_chunks - 1:
                    flags |= 8

                coords.append((y, x))
                corners.append(flags)
                # tu nie wiemy, które intra => 0
                is_intra.append(0)

        chunks = np.stack(chunks, axis=0)
        chunks = np.transpose(chunks, (0, 3, 1, 2))
        chunks_t = torch.from_numpy(chunks.astype(np.uint8))
        coords_t = torch.tensor(coords, dtype=torch.int32)
        corners_t = torch.tensor(corners, dtype=torch.uint8)
        intra_t = torch.tensor(is_intra, dtype=torch.uint8)

        out_path = os.path.join(out_seq_dir, f"chunks_poc{poc:03d}.pt")
        torch.save(
            {
                "seq_meta": {
                    "name": name,
                    "profile": "orig",
                    "qp": None,
                    "alf": None,
                    "db": None,
                    "sao": None,
                    "width": width,
                    "height": height,
                },
                "poc": poc,
                "chunk_size": (chunk_h, chunk_w),
                "chunks": chunks_t,
                "coords": coords_t,
                "corner_flags": corners_t,
                "is_intra": intra_t,
            },
            out_path,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decoded_or_orig_root", help="albo decoded_new (z podfolderami), albo data/ (z .yuv)")
    ap.add_argument("data_root", help="katalog z .info (dla decoded_new może być taki sam jak pierwszy)")
    ap.add_argument("out_root", help="gdzie zapisywać .pt")
    ap.add_argument("--chunk-w", type=int, default=132)
    ap.add_argument("--chunk-h", type=int, default=132)
    ap.add_argument("--border", type=int, default=2)
    args = ap.parse_args()

    src = args.decoded_or_orig_root
    data_root = args.data_root
    out_root = args.out_root

    os.makedirs(out_root, exist_ok=True)

    # spróbujmy najpierw "decoded" (podfoldery z recon.yuv)
    decoded = list(iter_decoded_sequences(src))
    if decoded:
        print(f"📂 Detected decoded layout: {len(decoded)} sequences")
        for seq in decoded:
            print(f"  -> {seq['dir_name']}")
            save_chunks_for_decoded(
                seq,
                data_root,
                out_root,
                chunk_w=args.chunk_w,
                chunk_h=args.chunk_h,
                border=args.border,
            )
        return

    # jeśli nie decoded -> traktujemy jako oryginały
    originals = list(iter_original_sequences(src))
    if not originals:
        print("❌ No sequences found in either decoded or original format.")
        return

    print(f"📂 Detected original layout: {len(originals)} sequences")
    for orig in originals:
        print(f"  -> {orig['name']}")
        save_chunks_for_original(
            orig,
            out_root,
            chunk_w=args.chunk_w,
            chunk_h=args.chunk_h,
            border=args.border,
        )


if __name__ == "__main__":
    main()
