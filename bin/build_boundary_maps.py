#!/usr/bin/env python3
"""
Integrated boundary map generator for decode_data.sh workflow.
Auto-detects resolution and POC range from dequant data or block_stats.csv.
"""

import argparse, os, csv, numpy as np
import glob, re

def sniff_csv(path):
    """Detect CSV delimiter"""
    with open(path, 'r', newline='') as f:
        head = f.read(4096)
        f.seek(0)
        return csv.Sniffer().sniff(head, delimiters=";\t,")

def auto_detect_resolution_from_csv(csv_path):
    """Try to detect resolution from block_stats.csv header, fallback to max coordinates"""
    try:
        # First try to read from header comment "# Sequence size: [WxH]"
        with open(csv_path, 'r') as f:
            for line in f:
                if line.startswith("# Sequence size:"):
                    # Parse "# Sequence size: [352x 240]" format
                    import re
                    match = re.search(r'\[(\d+)x\s*(\d+)\]', line)
                    if match:
                        width, height = int(match.group(1)), int(match.group(2))
                        return width, height
                elif not line.startswith("#"):
                    # Stop reading when we hit data rows
                    break
    except Exception as e:
        print(f"WARN: Could not read header from CSV: {e}")

    # Fallback: try max coordinates method
    max_x, max_y = 0, 0
    try:
        dialect = sniff_csv(csv_path)
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=dialect.delimiter)
            for row in reader:
                if not row or not row[0].startswith("BlockStat"):
                    continue
                try:
                    x, y, w, h = int(row[2]), int(row[3]), int(row[4]), int(row[5])
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)
                except (ValueError, IndexError):
                    continue

        if max_x > 0 and max_y > 0:
            print(f"WARN: Using fallback max coordinates method: {max_x}x{max_y}")
            return max_x, max_y
    except Exception as e:
        print(f"WARN: Could not auto-detect resolution from CSV: {e}")

    return None, None

def auto_detect_poc_range_from_csv(csv_path):
    """Detect POC range from block_stats.csv"""
    pocs = set()
    try:
        dialect = sniff_csv(csv_path)
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=dialect.delimiter)
            for row in reader:
                if not row or not row[0].startswith("BlockStat"):
                    continue
                try:
                    poc = int(row[1])
                    pocs.add(poc)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"WARN: Could not auto-detect POC range from CSV: {e}")
        return 0, 63
    
    if pocs:
        return min(pocs), max(pocs)
    return 0, 63

def auto_detect_poc_range_from_dequant(directory):
    """Detect POC range from dequant files (fallback)"""
    poc_files = glob.glob(os.path.join(directory, "dequant_poc_*.bin"))
    if not poc_files:
        return 0, 63
    
    pocs = []
    for f in poc_files:
        match = re.search(r'dequant_poc_(\d+)\.bin', os.path.basename(f))
        if match:
            pocs.append(int(match.group(1)))
    
    if pocs:
        return min(pocs), max(pocs)
    return 0, 63

def read_block_rects(csv_path, poc):
    """Read block rectangles for specific POC"""
    rects = []
    try:
        dialect = sniff_csv(csv_path)
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=dialect.delimiter)
            for row in reader:
                if not row or not row[0].startswith("BlockStat"):
                    continue
                try:
                    p = int(row[1])
                    x, y, w, h = int(row[2]), int(row[3]), int(row[4]), int(row[5])
                except (ValueError, IndexError):
                    continue
                if p == poc:
                    rects.append((x, y, w, h))
    except Exception as e:
        print(f"WARN: Error reading blocks for POC {poc}: {e}")
    
    return rects

def build_boundary_maps(rects, width, height, ctu=128, alpha=1.0):
    """Generate boundary maps from block rectangles"""
    # Initialize maps
    id_map = -np.ones((height, width), np.int32)
    size_map = np.zeros((height, width), np.int16)
    
    # Fill block IDs and sizes
    for bid, (x, y, w, h) in enumerate(rects):
        x2 = min(x + w, width)
        y2 = min(y + h, height)
        id_map[y:y2, x:x2] = bid
        size_map[y:y2, x:x2] = min(w, h)
    
    # Detect boundaries (horizontal and vertical)
    b_h = np.zeros_like(id_map, bool)
    b_v = np.zeros_like(id_map, bool)
    b_h[:, 1:] = id_map[:, 1:] != id_map[:, :-1]
    b_v[1:, :] = id_map[1:, :] != id_map[:-1, :]
    boundary_bin = (b_h | b_v).astype(np.uint8)
    
    # Calculate boundary weights
    size_right = np.zeros_like(size_map)
    size_right[:, 1:] = size_map[:, :-1]
    size_down = np.zeros_like(size_map)
    size_down[1:, :] = size_map[:-1, :]
    
    w_h = np.minimum(size_map, size_right)
    w_v = np.minimum(size_map, size_down)
    
    def weight_func(s):
        s = np.maximum(s.astype(np.float32), 1.0)
        return 1.0 + alpha * (np.log2(ctu) - np.log2(s))
    
    wh = np.zeros_like(size_map, float)
    wv = np.zeros_like(size_map, float)
    wh[b_h] = weight_func(w_h[b_h])
    wv[b_v] = weight_func(w_v[b_v])
    boundary_weight = np.maximum(wh, wv).astype(np.float32)
    
    # Normalize size map
    size_map_norm = (size_map.astype(np.float32) / float(ctu))
    
    return boundary_bin, boundary_weight, size_map_norm

def create_boundary_preview(boundary_bin, boundary_weight, size_map_norm, output_path):
    try:
        import matplotlib.pyplot as plt

        h, w = boundary_bin.shape
        ar = w / h

        # trochę bezpieczniejsze skalowanie figury (bez ręcznego set_aspect)
        fig_h = 5
        fig_w = fig_h * (ar * 3 + 0.5)  # +0.5 na colorbar
        fig, axs = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=True)

        im0 = axs[0].imshow(boundary_bin, cmap="gray", origin="upper", interpolation="nearest")
        axs[0].set_title(f"boundary_bin ({w}x{h})")
        axs[0].axis("off")

        im1 = axs[1].imshow(boundary_weight, cmap="magma", origin="upper", interpolation="nearest")
        axs[1].set_title(f"boundary_weight ({w}x{h})")
        axs[1].axis("off")

        im2 = axs[2].imshow(size_map_norm, cmap="viridis", origin="upper", interpolation="nearest")
        axs[2].set_title(f"size_map_norm ({w}x{h})")
        axs[2].axis("off")
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except ImportError:
        print("WARN: matplotlib not available, skipping boundary preview")
        return False
    except Exception as e:
        print(f"WARN: Failed to create boundary preview: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate boundary maps from VTM block statistics")
    parser.add_argument("input_dir", help="Directory containing block_stats.csv and/or dequant files")
    parser.add_argument("--outdir", default="boundary_maps", help="Output directory for boundary maps")
    parser.add_argument("--width", type=int, help="Frame width (auto-detected if not specified)")
    parser.add_argument("--height", type=int, help="Frame height (auto-detected if not specified)")
    parser.add_argument("--ctu", type=int, default=128, help="CTU size (default: 128)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight function alpha parameter")
    parser.add_argument("--poc-from", type=int, help="Start POC (auto-detected if not specified)")
    parser.add_argument("--poc-to", type=int, help="End POC (auto-detected if not specified)")
    parser.add_argument("--save-png", action="store_true", help="Generate PNG previews")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    # Find block_stats.csv
    csv_path = os.path.join(args.input_dir, "block_stats.csv")
    if not os.path.isfile(csv_path):
        print(f"ERROR: block_stats.csv not found in {args.input_dir}")
        return 1
    
    if args.debug:
        print(f"Reading block statistics from: {csv_path}")
    
    # Auto-detect parameters
    width = args.width
    height = args.height
    if width is None or height is None:
        auto_w, auto_h = auto_detect_resolution_from_csv(csv_path)
        if auto_w and auto_h:
            width, height = auto_w, auto_h
            if args.debug:
                print(f"Auto-detected resolution: {width}x{height}")
        else:
            print("ERROR: Could not auto-detect resolution. Please specify --width and --height")
            return 1
    
    poc_from = args.poc_from
    poc_to = args.poc_to
    if poc_from is None or poc_to is None:
        auto_from, auto_to = auto_detect_poc_range_from_csv(csv_path)
        if poc_from is None:
            poc_from = auto_from
        if poc_to is None:
            poc_to = auto_to
        if args.debug:
            print(f"Auto-detected POC range: {poc_from}-{poc_to}")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Process each POC
    generated_count = 0
    for poc in range(poc_from, poc_to + 1):
        rects = read_block_rects(csv_path, poc)
        if not rects:
            if args.debug:
                print(f"No blocks found for POC {poc}")
            continue
        
        if args.debug:
            print(f"Processing POC {poc}: {len(rects)} blocks")
        
        # Generate boundary maps
        boundary_bin, boundary_weight, size_map_norm = build_boundary_maps(
            rects, width, height, args.ctu, args.alpha
        )
        
        # Save NPZ
        npz_path = os.path.join(args.outdir, f"boundary_maps_poc{poc}.npz")
        np.savez_compressed(
            npz_path,
            boundary_bin=boundary_bin,
            boundary_weight=boundary_weight,
            size_map_norm=size_map_norm
        )
        
        # Save PNG preview if requested
        if args.save_png:
            png_path = os.path.join(args.outdir, f"boundary_maps_poc{poc}.png")
            create_boundary_preview(boundary_bin, boundary_weight, size_map_norm, png_path)
        
        generated_count += 1
    
    print(f"Generated {generated_count} boundary map files in {args.outdir}")
    return 0

if __name__ == "__main__":
    exit(main())