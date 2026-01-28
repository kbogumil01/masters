#!/usr/bin/env python3
"""
Step 3: Convert YUV 4:4:4 to YUV 4:2:0 using ffmpeg.

Usage:
    python -m bin.my_metrics.convert_to_420 \\
        --input-dir videos_test/enhanced_yuv444 \\
        --output-dir videos_test/enhanced_yuv420

Or generate task file for parallel execution:
    python -m bin.my_metrics.convert_to_420 \\
        --input-dir videos_test/enhanced_yuv444 \\
        --output-dir videos_test/enhanced_yuv420 \\
        --generate-tasks tasks_convert_420.txt
"""

import argparse
import subprocess
import re
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import parse_sequence_name, get_resolution


def convert_yuv444_to_420(input_path: Path, output_path: Path, width: int, height: int):
    """Convert YUV 4:4:4 to YUV 4:2:0 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "yuv444p",
        "-s", f"{width}x{height}",
        "-i", str(input_path),
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def generate_ffmpeg_command(input_path: Path, output_path: Path, width: int, height: int) -> str:
    """Generate ffmpeg command string."""
    return (
        f"ffmpeg -y -f rawvideo -pix_fmt yuv444p -s {width}x{height} "
        f"-i {input_path} -pix_fmt yuv420p {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert YUV 4:4:4 to YUV 4:2:0")
    parser.add_argument("--input-dir", required=True, help="Input directory with YUV 4:4:4 files")
    parser.add_argument("--output-dir", required=True, help="Output directory for YUV 4:2:0 files")
    parser.add_argument("--generate-tasks", help="Generate task file instead of running")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yuv_files = sorted(input_dir.glob("*.yuv"))
    
    if args.generate_tasks:
        # Generate task file
        tasks = []
        for yuv_file in yuv_files:
            seq_name = yuv_file.stem
            info = parse_sequence_name(seq_name)
            if info:
                width, height = get_resolution(info['name'])
            else:
                # Try to get from filename
                width, height = get_resolution(seq_name)
            
            output_path = output_dir / yuv_file.name
            tasks.append(generate_ffmpeg_command(yuv_file, output_path, width, height))
        
        with open(args.generate_tasks, "w") as f:
            f.write("\n".join(tasks))
        print(f"Generated {len(tasks)} tasks to {args.generate_tasks}")
    else:
        # Run conversions
        print(f"Converting {len(yuv_files)} files...")
        for yuv_file in tqdm(yuv_files, desc="Converting"):
            seq_name = yuv_file.stem
            info = parse_sequence_name(seq_name)
            if info:
                width, height = get_resolution(info['name'])
            else:
                width, height = get_resolution(seq_name)
            
            output_path = output_dir / yuv_file.name
            convert_yuv444_to_420(yuv_file, output_path, width, height)
        
        print(f"Done! Files saved to {output_dir}")


if __name__ == "__main__":
    main()
