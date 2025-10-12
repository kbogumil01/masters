#!/usr/bin/env python3
# bin/build_dequant_maps_direct.py
# Bezpośrednie przetwarzanie dequant_poc_*.{csv,bin} -> mapy neuronowej

import argparse, csv, os, sys
from pathlib import Path
import numpy as np

def load_and_process_poc_files(input_dir: Path, debug: bool = False):
    """Ładuje wszystkie pliki CSV/BIN i zwraca połączone meta+coeffs."""
    csv_files = sorted(input_dir.glob("dequant_poc_*.csv"))
    if not csv_files:
        print(f"No dequant_poc_*.csv files found in {input_dir}")
        return None, None, None, None
    
    all_meta = []
    all_coeffs = []
    coeffs_offset = 0
    poc_numbers = []
    
    for csv_file in csv_files:
        bin_file = csv_file.with_suffix(".bin")
        if not bin_file.exists():
            if debug:
                print(f"Skipping {csv_file.name} - no corresponding .bin file")
            continue
        
        # Extract POC number
        try:
            poc = int(csv_file.stem.split("_")[-1])
            poc_numbers.append(poc)
        except ValueError:
            if debug:
                print(f"Cannot extract POC from {csv_file.name}")
            continue
        
        # Load metadata
        meta_rows = []
        with open(csv_file, "r", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)  # skip header
            for line in reader:
                if not line or line[0].startswith("#"):
                    continue
                poc_val, comp, x, y, w, h, off_bytes, len_bytes = map(int, line)
                # Convert bytes to TCoeff positions (TCoeff = int32 = 4 bytes)
                off_coeffs = off_bytes // 4 + coeffs_offset
                len_coeffs = len_bytes // 4
                meta_rows.append([poc_val, comp, x, y, w, h, off_coeffs, len_coeffs])
        
        # Load coefficients
        coeffs = np.fromfile(bin_file, dtype="<i4")  # little-endian int32
        
        all_meta.extend(meta_rows)
        all_coeffs.append(coeffs)
        coeffs_offset += len(coeffs)
        
        if debug:
            print(f"Loaded {csv_file.name}: {len(meta_rows)} blocks, {len(coeffs)} coeffs")
    
    if not all_meta:
        return None, None, None, None
    
    # Combine all data
    combined_meta = np.array(all_meta, dtype=np.int32)
    combined_coeffs = np.concatenate(all_coeffs) if all_coeffs else np.array([], dtype=np.int32)
    
    # Auto-detect resolution
    if combined_meta.size > 0:
        max_x = (combined_meta[:, 2] + combined_meta[:, 4]).max()  # x + w
        max_y = (combined_meta[:, 3] + combined_meta[:, 5]).max()  # y + h
        
        # ZWRÓĆ DOKŁADNY ROZMIAR, BEZ ZAOKRĄGLANIA
        width, height = max_x, max_y
    else:
        width, height = 1920, 1080  # fallback
    
    poc_range = (min(poc_numbers), max(poc_numbers)) if poc_numbers else (0, 0)
    
    if debug:
        print(f"Combined: {combined_meta.shape[0]} blocks, {combined_coeffs.shape[0]} coeffs")
        print(f"Auto-detected: {width}x{height}, POC range: {poc_range}")
    
    return combined_meta, combined_coeffs, (width, height), poc_range

def build_maps_for_poc(meta, coeffs, poc, width, height, with_dc=False, debug=False):
    """Buduje mapy dla jednego POC."""
    # Filter for this POC and luma only (comp=0)
    rows = meta[(meta[:, 0] == poc) & (meta[:, 1] == 0)]
    
    if rows.size == 0:
        if debug:
            print(f"  POC {poc}: no luma blocks found")
        return None
    
    y_dc = np.zeros((height, width), np.float32) if with_dc else None
    y_energy = np.zeros((height, width), np.float32)
    y_nz = np.zeros((height, width), np.float32)
    
    blocks_processed = 0
    
    for poc_i, comp, x, y, w, h, start, length in rows:
        x, y, w, h = int(x), int(y), int(w), int(h)
        start, length = int(start), int(length)
        
        # Clamp to image bounds
        x2 = min(x + w, width)
        y2 = min(y + h, height)
        
        if x >= width or y >= height or x2 <= x or y2 <= y:
            continue
            
        area = (x2 - x) * (y2 - y)
        
        # Extract coefficients
        end = min(start + length, coeffs.shape[0])
        if end <= start:
            continue
            
        block_coeffs = coeffs[start:end]
        
        # DC coefficient (first one)
        if with_dc and len(block_coeffs) > 0:
            dc = float(block_coeffs[0])
            y_dc[y:y2, x:x2] = dc  # Set DC value for the entire block
        
        # AC coefficients (rest)
        if len(block_coeffs) > 1:
            ac_coeffs = block_coeffs[1:]
            energy = np.sum(np.abs(ac_coeffs)) / area
            nz_density = np.count_nonzero(ac_coeffs) / area
            
            y_energy[y:y2, x:x2] = energy     # Set energy density
            y_nz[y:y2, x:x2] = nz_density    # Set non-zero density
        
        blocks_processed += 1
    
    if debug:
        print(f"  POC {poc}: processed {blocks_processed} luma blocks")
    
    result = {"y_ac_energy": y_energy, "y_nz_density": y_nz}
    if with_dc:
        result["y_dc"] = y_dc
    
    return result

def save_png_preview(maps, output_dir, poc, debug=False):
    """Save PNG previews of the maps."""
    try:
        from PIL import Image
        
        def save_png(arr, path, normalize=True):
            if normalize:
                lo, hi = np.percentile(arr, [1, 99])
                if hi > lo:
                    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
                else:
                    arr = np.zeros_like(arr)
            
            img = (arr * 255).astype(np.uint8)
            Image.fromarray(img).save(path)
        
        base = os.path.join(output_dir, f"dequant_maps_poc{poc}")
        save_png(maps["y_ac_energy"], f"{base}_energy.png")
        save_png(maps["y_nz_density"], f"{base}_nz.png")
        
        if "y_dc" in maps:
            # DC can be negative, center around 0
            dc = maps["y_dc"]
            dc_max = max(abs(dc.min()), abs(dc.max()))
            if dc_max > 0:
                dc_norm = (dc / dc_max + 1) / 2  # [-1,1] -> [0,1]
            else:
                dc_norm = np.zeros_like(dc)
            save_png(dc_norm, f"{base}_dc.png", normalize=False)
        
        if debug:
            print(f"  Saved PNG previews for POC {poc}")
            
    except ImportError:
        if debug:
            print("PIL not available, skipping PNG generation")

def main():
    ap = argparse.ArgumentParser(description="Direct dequant processing: .csv/.bin -> neural network maps")
    ap.add_argument("input_dir", help="Directory with dequant_poc_*.{csv,bin} files")
    ap.add_argument("--outdir", required=True, help="Output directory for dequant_maps_poc*.npz")
    ap.add_argument("--width", type=int, help="Force specific width (auto-detect if not given)")
    ap.add_argument("--height", type=int, help="Force specific height (auto-detect if not given)")
    ap.add_argument("--poc-from", type=int, help="Start POC (auto-detect if not given)")
    ap.add_argument("--poc-to", type=int, help="End POC (auto-detect if not given)")
    ap.add_argument("--with-dc", action="store_true", help="Include DC coefficients")
    ap.add_argument("--save-png", action="store_true", help="Save PNG previews")
    ap.add_argument("--debug", action="store_true", help="Verbose output")
    ap.add_argument("--cleanup", action="store_true", help="Remove original .csv/.bin files after processing")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.debug:
        print(f"Processing dequant files from: {input_dir}")
        print(f"Output directory: {args.outdir}")
    
    # Load and combine all data
    meta, coeffs, resolution, poc_range = load_and_process_poc_files(input_dir, args.debug)
    
    if meta is None:
        print("No valid data found")
        sys.exit(1)
    
    # Use provided or auto-detected parameters
    width = args.width or resolution[0]
    height = args.height or resolution[1]
    poc_from = args.poc_from or poc_range[0]
    poc_to = args.poc_to or poc_range[1]
    
    if args.debug:
        print(f"Processing parameters:")
        print(f"  Resolution: {width}x{height}")
        print(f"  POC range: {poc_from}-{poc_to}")
        print(f"  Include DC: {args.with_dc}")
        print(f"  Save PNG: {args.save_png}")
    
    # Process each POC
    maps_saved = 0
    for poc in range(poc_from, poc_to + 1):
        if args.debug:
            print(f"Processing POC {poc}...")
        
        maps = build_maps_for_poc(meta, coeffs, poc, width, height, args.with_dc, args.debug)
        
        if maps is None:
            if args.debug:
                print(f"  POC {poc}: no data, skipping")
            continue
        
        # Save NPZ
        output_file = os.path.join(args.outdir, f"dequant_maps_poc{poc}.npz")
        np.savez_compressed(output_file, **maps)
        maps_saved += 1
        
        if args.debug:
            print(f"  Saved: {output_file}")
        
        # Save PNG previews if requested
        if args.save_png:
            save_png_preview(maps, args.outdir, poc, args.debug)
    
    # Cleanup original files if requested
    if args.cleanup:
        csv_files = list(input_dir.glob("dequant_poc_*.csv"))
        bin_files = list(input_dir.glob("dequant_poc_*.bin"))
        
        for f in csv_files + bin_files:
            try:
                f.unlink()
                if args.debug:
                    print(f"Removed: {f}")
            except Exception as e:
                print(f"Warning: Could not remove {f}: {e}")
    
    print(f"Processing complete!")
    print(f"Generated {maps_saved} neural network map files in: {args.outdir}")
    if args.save_png:
        print(f"PNG previews also saved")

if __name__ == "__main__":
    main()