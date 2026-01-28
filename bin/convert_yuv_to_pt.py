#!/usr/bin/env python3
"""
Convert decoded YUV files to .pt format for BD-rate analysis.
Extracts frames from recon.yuv files in decoded_test directories.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import re
from tqdm import tqdm


def read_yuv_frame(yuv_file, width, height, frame_idx):
    """
    Read a single YUV 420 frame from file.
    
    Returns:
        torch.Tensor of shape [3, height, width] with Y, U, V channels
    """
    # YUV 420: Y plane is W*H, U and V planes are W/2 * H/2
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size
    
    yuv_file.seek(frame_idx * frame_size)
    
    # Read Y plane
    y_data = np.frombuffer(yuv_file.read(y_size), dtype=np.uint8)
    y_plane = y_data.reshape(height, width)
    
    # Read U plane
    u_data = np.frombuffer(yuv_file.read(uv_size), dtype=np.uint8)
    u_plane = u_data.reshape(height // 2, width // 2)
    
    # Read V plane
    v_data = np.frombuffer(yuv_file.read(uv_size), dtype=np.uint8)
    v_plane = v_data.reshape(height // 2, width // 2)
    
    # Upsample U and V to full resolution
    u_upsampled = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)
    v_upsampled = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)
    
    # Stack into YUV tensor
    yuv_tensor = torch.from_numpy(
        np.stack([y_plane, u_upsampled, v_upsampled], axis=0)
    ).to(torch.uint8)
    
    return yuv_tensor


def extract_metadata_from_dirname(dirname: str):
    """Extract sequence metadata from directory name."""
    # Pattern: sequence_name_PROFILE_QP##_ALF#_DB#_SAO#
    match = re.match(r'(.+?)_(AI|RA)_QP(\d+)_ALF(\d)_DB(\d)_SAO(\d)', dirname)
    
    if not match:
        return None
    
    seq_name = match.group(1)
    profile = match.group(2)
    qp = int(match.group(3))
    alf = int(match.group(4))
    db = int(match.group(5))
    sao = int(match.group(6))
    
    return {
        'name': seq_name,
        'profile': profile,
        'qp': qp,
        'alf': alf,
        'db': db,
        'sao': sao
    }


def get_sequence_dimensions(seq_name: str):
    """Get video dimensions from sequence name."""
    # Common test sequences
    dimensions = {
        'students_qcif': (176, 144),
        'bridge_far_cif': (352, 288),
        'stefan_cif': (352, 288),
        'Johnny_1280x720_60': (1280, 720),
        'tractor_1080p25': (1920, 1080),
    }
    
    for key, dims in dimensions.items():
        if key in seq_name:
            return dims
    
    # Try to extract from name (e.g., "name_WxH")
    match = re.search(r'(\d+)x(\d+)', seq_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None


def convert_yuv_to_pt(yuv_path: Path, output_dir: Path, metadata: dict, num_frames: int = 64):
    """Convert YUV file to individual .pt frame files."""
    
    width, height = get_sequence_dimensions(metadata['name'])
    if not width or not height:
        print(f"  ⚠️  Unknown dimensions for {metadata['name']}")
        return 0
    
    # Add padding to dimensions (round up to nearest 64)
    padded_width = ((width + 63) // 64) * 64
    padded_height = ((height + 63) // 64) * 64
    
    metadata.update({
        'orig_width': width,
        'orig_height': height,
        'padded_width': padded_width,
        'padded_height': padded_height
    })
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    
    with open(yuv_path, 'rb') as yuv_file:
        for frame_idx in range(num_frames):
            output_file = output_dir / f"frame_poc{frame_idx:03d}.pt"
            
            if output_file.exists():
                continue  # Skip existing files
            
            try:
                # Read YUV frame
                yuv_tensor = read_yuv_frame(yuv_file, padded_width, padded_height, frame_idx)
                
                # Create .pt file structure
                frame_data = {
                    'chunk': yuv_tensor,  # [3, H, W] uint8
                    'seq_meta': metadata,
                    'poc': frame_idx,
                    'is_intra': (frame_idx == 0),  # First frame is usually intra
                    # vvc_features can be added later if available
                }
                
                torch.save(frame_data, output_file)
                converted += 1
                
            except Exception as e:
                print(f"    ⚠️  Error converting frame {frame_idx}: {e}")
                break
    
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert YUV files to .pt format for BD-rate analysis"
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory with decoded video (containing recon.yuv)'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Target directory for .pt frame files'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=64,
        help='Number of frames to extract from each video'
    )
    
    args = parser.parse_args()
    
    source = Path(args.source)
    target = Path(args.target)
    
    if not source.exists():
        print(f"❌ Source directory not found: {source}")
        return 1
    
    # Check if source is a single decoded directory or a parent directory
    yuv_file = source / "recon.yuv"
    
    if yuv_file.exists():
        # Single directory mode
        print("=" * 80)
        print("YUV TO .PT CONVERSION - Single Directory")
        print("=" * 80)
        print(f"Source: {source}")
        print(f"Target: {target}")
        print(f"Frames: {args.num_frames}")
        print()
        
        metadata = extract_metadata_from_dirname(source.name)
        if not metadata:
            print(f"❌ Could not parse metadata from: {source.name}")
            return 1
        
        print(f"Sequence: {metadata['name']}, QP={metadata['qp']}")
        
        converted = convert_yuv_to_pt(yuv_file, target, metadata, args.num_frames)
        
        print()
        print("=" * 80)
        print(f"✅ CONVERSION COMPLETE: {converted} frames")
        print("=" * 80)
        
        return 0
    
    # Multi-directory mode (original logic)
    print("=" * 80)
    print("YUV TO .PT CONVERSION - Multi Directory")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Frames per video: {args.num_frames}")
    print()
    
    # Find all decoded directories with YUV files
    yuv_dirs = []
    for d in sorted(source.iterdir()):
        if not d.is_dir():
            continue
        
        yuv_file = d / "recon.yuv"
        if not yuv_file.exists():
            continue
        
        metadata = extract_metadata_from_dirname(d.name)
        if not metadata:
            print(f"⚠️  Could not parse metadata from: {d.name}")
            continue
        
        yuv_dirs.append((d, yuv_file, metadata))
    
    print(f"Found {len(yuv_dirs)} directories to convert")
    print()
    
    total_converted = 0
    
    for dec_dir, yuv_file, metadata in tqdm(yuv_dirs, desc="Converting"):
        output_dir = target / dec_dir.name
        
        print(f"\n{dec_dir.name}")
        print(f"  QP={metadata['qp']}, Sequence={metadata['name']}")
        
        converted = convert_yuv_to_pt(
            yuv_file,
            output_dir,
            metadata,
            args.num_frames
        )
        
        total_converted += converted
        print(f"  ✅ Converted {converted} frames")
    
    print()
    print("=" * 80)
    print(f"✅ CONVERSION COMPLETE: {total_converted} frames total")
    print("=" * 80)
    print()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
