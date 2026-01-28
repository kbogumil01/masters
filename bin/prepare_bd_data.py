#!/usr/bin/env python3
"""
Convert decoded_test directory structure to format expected by BD-rate analysis.
Creates symbolic links from decoded frames to expected format.
"""

import os
from pathlib import Path
import torch
from tqdm import tqdm
import argparse


def convert_yuv_to_pt(yuv_dir: Path, output_dir: Path, width: int, height: int):
    """
    Convert .yuv files to .pt format expected by BD-rate script.
    
    Note: This is a placeholder. If you have .yuv files, implement conversion.
    If you already have .pt files in decoded_test subdirectories, use create_symlinks instead.
    """
    raise NotImplementedError("Implement YUV to PT conversion if needed")


def find_pt_files_in_subdirs(root_dir: Path):
    """Check if .pt files exist in subdirectories."""
    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            pt_files = list(subdir.glob("*.pt"))
            if pt_files:
                print(f"Found {len(pt_files)} .pt files in {subdir.name}")
                return True
    return False


def create_symlinks(source_dir: Path, target_dir: Path):
    """
    Create symbolic links from source to target directory.
    
    This allows using decoded_test data without duplicating files.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for seq_dir in tqdm(sorted(source_dir.iterdir()), desc="Creating links"):
        if not seq_dir.is_dir():
            continue
        
        # Create corresponding directory in target
        target_seq_dir = target_dir / seq_dir.name
        
        if target_seq_dir.exists():
            # Check if it's already a symlink or has files
            if target_seq_dir.is_symlink():
                print(f"  ⚠️  Skipping {seq_dir.name} (already linked)")
                continue
            elif any(target_seq_dir.iterdir()):
                print(f"  ⚠️  Skipping {seq_dir.name} (directory exists and not empty)")
                continue
        
        # Create symlink to entire directory
        try:
            target_seq_dir.symlink_to(seq_dir.absolute(), target_is_directory=True)
            count += 1
        except FileExistsError:
            print(f"  ⚠️  Skipping {seq_dir.name} (already exists)")
        except Exception as e:
            print(f"  ❌ Error linking {seq_dir.name}: {e}")
    
    print(f"\n✅ Created {count} symbolic links")


def copy_files(source_dir: Path, target_dir: Path, file_pattern: str = "frame_*.pt"):
    """
    Copy files from source to target (if symlinks not supported).
    """
    import shutil
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    for seq_dir in tqdm(sorted(source_dir.iterdir()), desc="Copying files"):
        if not seq_dir.is_dir():
            continue
        
        target_seq_dir = target_dir / seq_dir.name
        target_seq_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(seq_dir.glob(file_pattern))
        for f in files:
            target_file = target_seq_dir / f.name
            if not target_file.exists():
                shutil.copy2(f, target_file)
                total_files += 1
    
    print(f"\n✅ Copied {total_files} files")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare test data for BD-rate analysis"
    )
    parser.add_argument(
        '--source',
        type=str,
        default='videos_test/decoded_test',
        help='Source directory with decoded frames'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='videos_test/test_frames_REAL',
        help='Target directory for BD-rate analysis'
    )
    parser.add_argument(
        '--method',
        choices=['symlink', 'copy'],
        default='symlink',
        help='Method to use: symlink (fast) or copy (safe)'
    )
    
    args = parser.parse_args()
    
    source = Path(args.source)
    target = Path(args.target)
    
    if not source.exists():
        print(f"❌ Source directory not found: {source}")
        return 1
    
    print("=" * 80)
    print("PREPARING BD-RATE TEST DATA")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Method: {args.method}")
    print()
    
    # Check if source has .pt files in subdirectories
    has_pt_files = find_pt_files_in_subdirs(source)
    
    if not has_pt_files:
        print("⚠️  No .pt files found in source subdirectories")
        print("   Make sure your decoded frames are in .pt format")
        return 1
    
    print()
    
    if args.method == 'symlink':
        print("Creating symbolic links (fast, no disk space used)...")
        create_symlinks(source, target)
    else:
        print("Copying files (slower, uses disk space)...")
        copy_files(source, target)
    
    print()
    print("=" * 80)
    print("✅ PREPARATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Verify data: python3 bin/verify_bd_data.py")
    print("  2. Run analysis: ./run_bd_analysis.sh")
    print()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
