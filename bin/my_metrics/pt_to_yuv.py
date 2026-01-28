#!/usr/bin/env python3
"""
Step 2: Convert .pt frames to YUV 4:4:4 video files.

This converts tensors from .pt files to raw YUV 4:4:4 video files.

Usage:
    python -m bin.my_metrics.pt_to_yuv \\
        --input-dir videos_test/enhanced_pt \\
        --output-dir videos_test/enhanced_yuv444

Output structure:
    enhanced_yuv444/
        Johnny_1280x720_60_AI_QP32_ALF0_DB0_SAO0.yuv
        ...
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def pt_frames_to_yuv444(seq_dir: Path, output_path: Path):
    """
    Convert .pt frame files to a single YUV 4:4:4 video file.
    
    Input tensor format: [3, H, W] float32 (0-1) as Y, U, V
    Output format: planar YUV 4:4:4 (Y plane, then U plane, then V plane per frame)
    """
    frame_files = sorted(seq_dir.glob("frame_poc*.pt"))
    
    if len(frame_files) == 0:
        print(f"  Warning: No frames in {seq_dir}")
        return
    
    with open(output_path, "wb") as f:
        for frame_file in frame_files:
            data = torch.load(frame_file, map_location="cpu")
            
            # Get chunk tensor - could be float32 0-1 (enhanced) or uint8 0-255 (decoded)
            chunk = data["chunk"]
            
            if chunk.dtype == torch.float32:
                # Enhanced frames: float32 0-1 → uint8 0-255
                chunk = (chunk.clamp(0, 1) * 255).to(torch.uint8)
            
            # chunk is [3, H, W] as Y, U, V
            Y = chunk[0].numpy()
            U = chunk[1].numpy()
            V = chunk[2].numpy()
            
            # Get original dimensions (remove padding if any)
            seq_meta = data.get("seq_meta", {})
            orig_h = seq_meta.get("orig_height", Y.shape[0])
            orig_w = seq_meta.get("orig_width", Y.shape[1])
            
            # Crop to original size
            Y = Y[:orig_h, :orig_w]
            U = U[:orig_h, :orig_w]
            V = V[:orig_h, :orig_w]
            
            # Write planar YUV 4:4:4
            f.write(Y.tobytes())
            f.write(U.tobytes())
            f.write(V.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Convert .pt frames to YUV 4:4:4")
    parser.add_argument("--input-dir", required=True, help="Input directory with .pt frames")
    parser.add_argument("--output-dir", required=True, help="Output directory for YUV files")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all sequence directories
    seq_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    print(f"Converting {len(seq_dirs)} sequences to YUV 4:4:4...")
    for seq_dir in tqdm(seq_dirs, desc="Converting"):
        output_path = output_dir / f"{seq_dir.name}.yuv"
        pt_frames_to_yuv444(seq_dir, output_path)
    
    print(f"Done! YUV files saved to {output_dir}")


if __name__ == "__main__":
    main()
