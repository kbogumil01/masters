#!/usr/bin/env python3
"""
Convert decoded test frames (.pt) to YUV 4:2:0 for metric comparison.

Usage:
    python -m bin.my_metrics.convert_decoded_to_420 \\
        --input-dir videos_test/test_frames_REAL \\
        --output-dir videos_test/decoded_yuv420
"""

import torch
import numpy as np
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import parse_sequence_name, get_resolution


def pt_frames_to_yuv420(seq_dir: Path, output_path: Path, width: int, height: int):
    """
    Convert .pt frame files to YUV 4:2:0 video file.
    
    Process: .pt (YUV444 upsampled) → YUV444 temp → ffmpeg → YUV420
    """
    frame_files = sorted(seq_dir.glob("frame_poc*.pt"))
    
    if len(frame_files) == 0:
        print(f"  Warning: No frames in {seq_dir}")
        return
    
    # First, create temporary YUV444 file
    with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # Write YUV444 to temp file
        with open(tmp_path, "wb") as f:
            for frame_file in frame_files:
                data = torch.load(frame_file, map_location="cpu")
                chunk = data["chunk"]
                
                if chunk.dtype == torch.float32:
                    chunk = (chunk.clamp(0, 1) * 255).to(torch.uint8)
                
                # Get original dimensions
                seq_meta = data.get("seq_meta", {})
                orig_h = seq_meta.get("orig_height", chunk.shape[1])
                orig_w = seq_meta.get("orig_width", chunk.shape[2])
                
                # Crop to original size
                Y = chunk[0, :orig_h, :orig_w].numpy()
                U = chunk[1, :orig_h, :orig_w].numpy()
                V = chunk[2, :orig_h, :orig_w].numpy()
                
                # Write planar YUV 4:4:4
                f.write(Y.tobytes())
                f.write(U.tobytes())
                f.write(V.tobytes())
        
        # Convert YUV444 → YUV420 with ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "yuv444p",
            "-s", f"{width}x{height}",
            "-i", str(tmp_path),
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Convert decoded .pt frames to YUV 4:2:0")
    parser.add_argument("--input-dir", required=True, help="Input directory with decoded .pt frames")
    parser.add_argument("--output-dir", required=True, help="Output directory for YUV 4:2:0 files")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all sequence directories
    seq_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    print(f"Converting {len(seq_dirs)} sequences to YUV 4:2:0...")
    for seq_dir in tqdm(seq_dirs, desc="Converting"):
        seq_name = seq_dir.name
        info = parse_sequence_name(seq_name)
        
        if info:
            width, height = get_resolution(info['name'])
        else:
            # Try to parse from filename directly
            try:
                width, height = get_resolution(seq_name)
            except:
                print(f"Warning: Could not determine resolution for {seq_name}")
                continue
        
        output_path = output_dir / f"{seq_name}.yuv"
        pt_frames_to_yuv420(seq_dir, output_path, width, height)
    
    print(f"Done! YUV 4:2:0 files saved to {output_dir}")


if __name__ == "__main__":
    main()
