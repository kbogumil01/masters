#!/usr/bin/env python3
import os
import re
import glob
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm


# nazwa katalogu po dekodowaniu, np.:
# paris_cif_AI_QP37_ALF0_DB1_SAO0
ENCODED_REGEX = re.compile(
    r"^(?P<name>[\d\s\w]+)_(?P<profile>AI|RA)_QP(?P<qp>\d{2})_ALF(?P<alf>\d)_DB(?P<db>\d)_SAO(?P<sao>\d)$"
)

# wzorce z oryginalnych .info
INFO_HEIGHT_REGEX = re.compile(r"^\s*Height\s*:\s*(\d+)\s*$")
INFO_WIDTH_REGEX  = re.compile(r"^\s*Width\s*:\s*(\d+)\s*$")
INFO_FRAMES_REGEX = re.compile(r"^\s*Frame count\s*:\s*(\d+)\s*$")

# wzorce z decode.log
# Twój przypadek: [DEQDBG] ... CS.lumaSize=352x288
DEQDBG_SIZE_REGEX = re.compile(r".*CS\.lumaSize=(\d+)x(\d+)")
# klasyczny VTM: ... 720x1280 ...
VTM_SIZE_REGEX = re.compile(r".*\b(\d+)[xX](\d+)\b")

# użycie: python bin/split_maps_to_chunks.py data decoded maps_chunks


def try_load_from_info(data_path: str, name_parts: dict):
    """spróbuj wczytać jak w oryginalnym projekcie (data/<name>.yuv.info / .y4m.info)"""
    info_glob = glob.glob(os.path.join(data_path, f"{name_parts['name']}.*.info"))
    if not info_glob:
        return None
    width = height = frames = None
    with open(info_glob[0], "r") as f:
        for line in f:
            mh = INFO_HEIGHT_REGEX.match(line)
            if mh:
                height = int(mh.group(1))
            mw = INFO_WIDTH_REGEX.match(line)
            if mw:
                width = int(mw.group(1))
            mf = INFO_FRAMES_REGEX.match(line)
            if mf:
                frames = int(mf.group(1))
    if width is None or height is None:
        return None
    if frames is None:
        frames = 64
    return width, height, frames


def try_load_from_decode_log(decoded_dir: str):
    """
    fallback: weź dane z decoded/<seq>/decode.log
    obsługuje:
      [DEQDBG] ... CS.lumaSize=352x288
    oraz
      ... 720x1280 ...
    """
    log_path = os.path.join(decoded_dir, "decode.log")
    if not os.path.isfile(log_path):
        return None

    width = height = None
    frames = 0

    with open(log_path, "r") as f:
        for line in f:
            # licz klatki
            if re.match(r"^\s*POC\s+\d+", line):
                frames += 1

            # najpierw spróbuj Twojego formatu
            m1 = DEQDBG_SIZE_REGEX.match(line)
            if m1 and (width is None or height is None):
                w = int(m1.group(1))
                h = int(m1.group(2))
                # w logu masz 352x288 → 352 = width, 288 = height
                width, height = w, h
                continue

            # potem standard VTM
            m2 = VTM_SIZE_REGEX.match(line)
            if m2 and (width is None or height is None):
                a = int(m2.group(1))
                b = int(m2.group(2))
                # najczęściej większe to szerokość
                if a >= b:
                    width, height = a, b
                else:
                    width, height = b, a

    if width is None or height is None:
        return None
    if frames == 0:
        frames = 64
    return width, height, frames


def try_infer_from_recon(decoded_dir: str, frames: int = 64):
    """
    ostatnia deska ratunku: policz z rozmiaru pliku recon.yuv przy założeniu 4:2:0
    """
    recon_path = os.path.join(decoded_dir, "recon.yuv")
    if not os.path.isfile(recon_path):
        return None
    size_bytes = os.path.getsize(recon_path)
    # frame_size = w*h*3/2 → w*h = size_bytes*2 / (3*frames)
    pixels = (size_bytes * 2) // (3 * frames)
    # spróbuj standardowych wysokości
    for h in (2160, 1440, 1080, 720, 576, 540, 480, 360, 288, 240):
        if pixels % h == 0:
            w = pixels // h
            return w, h, frames
    return None


def load_video_metadata(data_path: str, decoded_root: str, dirname: str):
    """
    1. spróbuj data/*.info
    2. spróbuj decoded/<dir>/decode.log
    3. spróbuj odgadnąć z recon.yuv
    """
    m = ENCODED_REGEX.match(dirname)
    assert m, f"invalid decoded dir name: {dirname}"
    g = m.groupdict()

    # 1) z .info
    info = try_load_from_info(data_path, g)
    if info is not None:
        width, height, frames = info
    else:
        # 2) z decode.log
        decoded_dir = os.path.join(decoded_root, dirname)
        info = try_load_from_decode_log(decoded_dir)
        if info is not None:
            width, height, frames = info
        else:
            # 3) z recon.yuv
            info = try_infer_from_recon(decoded_dir)
            if info is None:
                raise RuntimeError(f"cannot determine resolution for {dirname}")
            width, height, frames = info

    return {
        "file": g["name"],
        "profile": g["profile"],
        "qp": int(g["qp"]),
        "alf": int(g["alf"]),
        "db": int(g["db"]),
        "sao": int(g["sao"]),
        "width": width,
        "height": height,
        "frames": frames,
    }


def pad_like_chunks(arr: np.ndarray, H: int, W: int, chunk_h: int, chunk_w: int, border: int):
    pad_top = border
    pad_bottom = 2 * border + ((-H) % (chunk_h - 2 * border))
    pad_left = border
    pad_right = 2 * border + ((-W) % (chunk_w - 2 * border))
    if arr.ndim == 2:
        arr = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
    else:
        arr = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return arr


def split_maps_for_sequence(
    decoded_root: str,
    dirname: str,
    data_path: str,
    out_dir: str,
    chunk_h: int = 132,
    chunk_w: int = 132,
    border: int = 2,
):
    decoded_dir = os.path.join(decoded_root, dirname)
    meta = load_video_metadata(data_path, decoded_root, dirname)

    H = meta["height"]
    W = meta["width"]
    F = meta["frames"]

    fused_dir = os.path.join(decoded_dir, "fused_maps")
    if not os.path.isdir(fused_dir):
        print(f"[WARN] no fused_maps/ in {decoded_dir}, skipping")
        return

    first_npz = os.path.join(fused_dir, "fused_maps_poc0.npz")
    assert os.path.isfile(first_npz), f"no fused_maps_poc0.npz in {fused_dir}"
    sample = np.load(first_npz)
    channel_names = list(sample.keys())
    sample.close()

    h_chunks = math.ceil(W / (chunk_w - 2 * border))
    v_chunks = math.ceil(H / (chunk_h - 2 * border))

    all_chunks = []
    all_metadata = []

    for fidx in tqdm(range(F), desc=f"maps {dirname[:25]}"):
        npz_path = os.path.join(fused_dir, f"fused_maps_poc{fidx}.npz")
        if not os.path.isfile(npz_path):
            # jeśli nie ma któregoś POC-a – pomiń
            continue

        maps_npz = np.load(npz_path)
        if any(isinstance(maps_npz[k], np.ndarray) and maps_npz[k].dtype == object for k in maps_npz.keys()):
            print(f"[WARN] object dtype found in {npz_path}")
        maps_list = []
        for name in channel_names:
            m = maps_npz[name]
            # jeżeli któraś mapa jest mniejsza albo większa – dopasuj
            if m.shape[0] != H or m.shape[1] != W:
                fixed = np.zeros((H, W), dtype=m.dtype)
                hh = min(H, m.shape[0])
                ww = min(W, m.shape[1])
                fixed[:hh, :ww] = m[:hh, :ww]
                m = fixed
            maps_list.append(m[..., None])
        frame_maps = np.concatenate(maps_list, axis=2)
        maps_npz.close()

        frame_maps = pad_like_chunks(frame_maps, H, W, chunk_h, chunk_w, border)

        for vh in range(v_chunks):
            y = vh * (chunk_h - 2 * border)
            for hh in range(h_chunks):
                x = hh * (chunk_w - 2 * border)

                chunk = frame_maps[y:y+chunk_h, x:x+chunk_w, :]
                all_chunks.append(chunk.astype(np.float32))

                corner = []
                if hh == 0:
                    corner.append("l")
                if hh == h_chunks - 1:
                    corner.append("r")
                if vh == 0:
                    corner.append("u")
                if vh == v_chunks - 1:
                    corner.append("b")

                all_metadata.append({
                    "position": (y, x),
                    "frame": fidx,
                    "corner": "".join(corner),
                    "file": meta["file"],
                    "qp": meta["qp"],
                    "alf": bool(meta["alf"]),
                    "db": bool(meta["db"]),
                    "sao": bool(meta["sao"]),
                    "width": meta["width"],
                    "height": meta["height"],
                })

    if not all_chunks:
        print(f"[WARN] no maps produced for {dirname}")
        return

    all_chunks = np.stack(all_chunks, axis=0)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dirname}.npz")
    np.savez_compressed(
        out_path,
        maps=all_chunks,
        metadata=all_metadata,
        channels=channel_names,
    )
    print(f"[OK] saved {out_path}  (chunks={len(all_chunks)}, channels={len(channel_names)})")


def main():
    import argparse
    p = argparse.ArgumentParser("split_maps_to_chunks")
    p.add_argument("data_path", help="katalog z .info ORAZ fallback jeśli jest")
    p.add_argument("decoded_path", help="katalog ze zdekodowanymi sekwencjami (ma podkatalogi z recon.yuv + fused_maps/)")
    p.add_argument("out_path", help="gdzie zapisać pocięte mapy")
    p.add_argument("--chunk-width", type=int, default=132)
    p.add_argument("--chunk-height", type=int, default=132)
    p.add_argument("--chunk-border", type=int, default=2)
    args = p.parse_args()

    decoded_dirs = [
        d for d in os.listdir(args.decoded_path)
        if os.path.isdir(os.path.join(args.decoded_path, d)) and not d.startswith(".")
    ]
    decoded_dirs = sorted(decoded_dirs)

    for d in decoded_dirs:
        if "_RA_" in d:
            continue
        split_maps_for_sequence(
            decoded_root=args.decoded_path,
            dirname=d,
            data_path=args.data_path,
            out_dir=args.out_path,
            chunk_h=args.chunk_height,
            chunk_w=args.chunk_width,
            border=args.chunk_border,
        )


if __name__ == "__main__":
    main()
