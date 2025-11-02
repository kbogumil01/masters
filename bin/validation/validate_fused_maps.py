#!/usr/bin/env python3
"""
Validate fused maps (.pt) produced by fuse_maps.py
Checks for key presence, tensor consistency, dtype correctness,
and basic statistical sanity of data.
"""
# python3 validate_fused_maps.py videos/decoded/part_1/bus_cif_AI_QP37_ALF1_DB1_SAO1/fused_maps --visualize
# or:
# python3 validate_fused_maps.py videos/decoded/part_1/bus_cif_AI_QP37_ALF1_DB1_SAO1/fused_maps

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

EXPECTED_KEYS = [
    "y_ac_energy",
    "y_nz_density",
    "y_dc",
    "boundary_bin",
    "boundary_weight",
    "size_map_norm",
]

def summarize_tensor(t, name):
    arr = t.cpu().numpy()
    shape_str = str(tuple(arr.shape))
    dtype_str = str(arr.dtype)
    print(f"  • {name:18s} | shape={shape_str:>12s} dtype={dtype_str:<8s} "
          f"| min={arr.min():8.3f} max={arr.max():8.3f} mean={arr.mean():8.3f}")

    if np.isnan(arr).any():
        print(f"    ⚠️  {name}: contains NaNs")
    if np.isinf(arr).any():
        print(f"    ⚠️  {name}: contains Infs")
    if (name.startswith('boundary') or name == 'size_map_norm') and arr.min() < 0:
        print(f"    ⚠️  {name}: has negative values (unexpected)")



def visualize_sample(tensors, poc, outdir="validation_previews"):
    """Visualize a few key maps for sanity check"""
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Fused maps visualization (POC {poc})", fontsize=14, fontweight="bold")

    keys = list(tensors.keys())
    for ax, k in zip(axes.flatten(), keys):
        arr = tensors[k].cpu().numpy()
        im = ax.imshow(arr, cmap="viridis", origin="upper")
        ax.set_title(k, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fused_maps_poc{poc}_preview.png"))
    plt.close(fig)

def validate_fused_maps(folder, visualize=False, debug=False):
    pt_files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
    if not pt_files:
        print(f"❌ No .pt fused maps found in: {folder}")
        return

    print(f"🔍 Found {len(pt_files)} fused map files in {folder}")
    inconsistent_shapes = {}
    global_shape = None

    for fname in pt_files:
        fpath = os.path.join(folder, fname)
        data = torch.load(fpath, map_location="cpu")

        poc_match = ''.join([c for c in fname if c.isdigit()])
        print(f"\n🧩 Validating {fname} (POC {poc_match})")

        # --- check expected keys ---
        missing = [k for k in EXPECTED_KEYS if k not in data]
        extra = [k for k in data if k not in EXPECTED_KEYS]
        if missing:
            print(f"  ⚠️ Missing keys: {missing}")
        if extra:
            print(f"  ℹ️ Extra keys: {extra}")

        # --- summarize each tensor ---
        for k in EXPECTED_KEYS:
            if k not in data:
                continue
            summarize_tensor(data[k], k)

        # --- check shape consistency ---
        shapes = [tuple(v.shape) for v in data.values()]
        if len(set(shapes)) > 1:
            print(f"  ⚠️ Inconsistent shapes in {fname}: {set(shapes)}")
            inconsistent_shapes[fname] = shapes
        else:
            if global_shape is None:
                global_shape = shapes[0]
            elif shapes[0] != global_shape:
                print(f"  ⚠️ Global shape mismatch ({shapes[0]} vs {global_shape})")

        # --- visualize first few ---
        if visualize:
            # wybierz kilka POC do pokazania: pierwszy, środkowy i ostatni
            idx = int(poc_match) if poc_match.isdigit() else None
            num_files = len(pt_files)

            # środkowy, pierwszy i ostatni
            selected = {0, num_files // 2, num_files - 1}

            if idx in selected:
                visualize_sample(data, poc=idx, outdir="validation_previews")

    print("\n✅ Validation complete.")
    if inconsistent_shapes:
        print("⚠️ Files with inconsistent tensor shapes:")
        for f, sh in inconsistent_shapes.items():
            print(f"   {f}: {sh}")
    else:
        print("✅ All shapes consistent across fused maps.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate fused maps (.pt) integrity and content.")
    parser.add_argument("folder", help="Path to fused_maps directory")
    parser.add_argument("--visualize", action="store_true", help="Create sample visualizations")
    parser.add_argument("--debug", action="store_true", help="Print extra debug info")
    args = parser.parse_args()

    validate_fused_maps(args.folder, visualize=args.visualize, debug=args.debug)
