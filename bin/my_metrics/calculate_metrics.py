#!/usr/bin/env python3
"""
Step 4: Calculate PSNR and SSIM metrics using ffmpeg (matching Piotr's pipeline).

Usage:
    python -m bin.my_metrics.calculate_metrics \\
        --enhanced-dir videos_test/enhanced_yuv420 \\
        --decoded-dir videos_test/decoded_yuv420 \\
        --original-dir videos_test/test_dataset \\
        --output-dir videos_test/metrics

Output:
    metrics/
        Johnny_1280x720_60_AI_QP32_ALF0_DB0_SAO0_enhanced_psnr.info
        Johnny_1280x720_60_AI_QP32_ALF0_DB0_SAO0_enhanced_ssim.info
        Johnny_1280x720_60_AI_QP32_ALF0_DB0_SAO0_decoded_psnr.info
        Johnny_1280x720_60_AI_QP32_ALF0_DB0_SAO0_decoded_ssim.info
        ...
"""

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import parse_sequence_name, get_resolution, get_original_yuv_path


def calculate_metric_ffmpeg(
    input_path: Path,
    reference_path: Path,
    width: int,
    height: int,
    metric: str,  # "psnr" or "ssim"
    output_log: Path
):
    """Calculate PSNR or SSIM using ffmpeg filter."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
        "-i", str(reference_path),
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
        "-i", str(input_path),
        "-filter_complex", metric,
        "-f", "null", "/dev/null"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # ffmpeg outputs metrics to stderr
    with open(output_log, "w") as f:
        f.write(result.stderr)


def generate_ffmpeg_command(
    input_path: Path,
    reference_path: Path,
    width: int,
    height: int,
    metric: str,
    output_log: Path
) -> str:
    """Generate ffmpeg command string."""
    return (
        f"ffmpeg -y "
        f"-f rawvideo -pix_fmt yuv420p -s {width}x{height} -i {reference_path} "
        f"-f rawvideo -pix_fmt yuv420p -s {width}x{height} -i {input_path} "
        f'-filter_complex "{metric}" '
        f"-f null /dev/null 2> {output_log}"
    )


def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR/SSIM metrics")
    parser.add_argument("--enhanced-dir", required=True, help="Enhanced YUV 4:2:0 directory")
    parser.add_argument("--decoded-dir", required=True, help="Decoded YUV 4:2:0 directory")
    parser.add_argument("--original-dir", required=True, help="Original YUV directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for metric logs")
    parser.add_argument("--generate-tasks", help="Generate task file instead of running")
    args = parser.parse_args()
    
    enhanced_dir = Path(args.enhanced_dir)
    decoded_dir = Path(args.decoded_dir)
    original_dir = Path(args.original_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all enhanced YUV files
    enhanced_files = sorted(enhanced_dir.glob("*.yuv"))
    
    tasks = []
    
    for enh_file in enhanced_files:
        seq_name = enh_file.stem
        info = parse_sequence_name(seq_name)
        
        if not info:
            print(f"Warning: Could not parse {seq_name}, skipping")
            continue
        
        # Get resolution and paths
        width, height = get_resolution(info['name'])
        original_path = original_dir / f"{info['name']}.yuv"
        decoded_path = decoded_dir / f"{seq_name}.yuv"
        
        if not original_path.exists():
            print(f"Warning: Original not found: {original_path}")
            continue
        
        # Generate tasks for both enhanced and decoded
        for source, prefix in [(enh_file, "enhanced"), (decoded_path, "decoded")]:
            if not source.exists():
                continue
            
            for metric in ["psnr", "ssim"]:
                log_file = output_dir / f"{seq_name}_{prefix}_{metric}.info"
                
                if args.generate_tasks:
                    cmd = generate_ffmpeg_command(
                        source, original_path, width, height, metric, log_file
                    )
                    tasks.append(cmd)
                else:
                    calculate_metric_ffmpeg(
                        source, original_path, width, height, metric, log_file
                    )
    
    if args.generate_tasks:
        with open(args.generate_tasks, "w") as f:
            f.write("\n".join(tasks))
        print(f"Generated {len(tasks)} tasks to {args.generate_tasks}")
    else:
        print(f"Done! Metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
