#!/usr/bin/env python3
"""
Convert existing chunk folders to NPZ format to reduce fragmentation

This script converts millions of small chunk files to compressed NPZ archives,
dramatically reducing disk usage and filesystem overhead.

Usage:
    python convert_chunks_to_npz.py /path/to/chunks /path/to/orig_chunks
"""

import os
import sys
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
import re
import shutil


def load_chunk_file(filepath: str) -> np.ndarray:
    """Load a chunk from binary file (was saved as .tobytes())"""
    with open(filepath, 'rb') as f:
        # Chunks are 132x132x3 uint8
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape((132, 132, 3))


def parse_chunk_path(filepath: str) -> dict:
    """Parse chunk metadata from file path"""
    # Example path: chunks/deadline_cif/AI_QP37_ALF1_DB0_SAO1/0_True/0_0_lu.png
    path_parts = Path(filepath).parts
    
    if len(path_parts) < 4:
        return None
        
    file_name = path_parts[-4]  # deadline_cif
    config = path_parts[-3]     # AI_QP37_ALF1_DB0_SAO1
    frame_info = path_parts[-2]  # 0_True
    chunk_info = path_parts[-1]  # 0_0_lu.png
    
    # Parse config: AI_QP37_ALF1_DB0_SAO1
    config_match = re.match(r'(?P<profile>AI|RA)_QP(?P<qp>\d+)_ALF(?P<alf>\d)_DB(?P<db>\d)_SAO(?P<sao>\d)', config)
    if not config_match:
        return None
        
    # Parse frame info: 0_True
    frame_parts = frame_info.split('_')
    if len(frame_parts) != 2:
        return None
    frame_num = int(frame_parts[0])
    is_intra = frame_parts[1] == 'True'
    
    # Parse chunk info: 0_0_lu.png
    chunk_name = chunk_info.replace('.png', '')
    chunk_parts = chunk_name.split('_')
    if len(chunk_parts) < 3:
        return None
    
    pos_v = int(chunk_parts[0])
    pos_h = int(chunk_parts[1])
    corner = '_'.join(chunk_parts[2:])
    
    return {
        'file': file_name,
        'profile': config_match.group('profile'),
        'qp': int(config_match.group('qp')),
        'alf': bool(int(config_match.group('alf'))),
        'db': bool(int(config_match.group('db'))),
        'sao': bool(int(config_match.group('sao'))),
        'frame': frame_num,
        'is_intra': is_intra,
        'position': (pos_v, pos_h),
        'corner': corner
    }


def convert_video_chunks(video_name: str, chunk_files: list, output_path: str):
    """Convert all chunks for one video to NPZ format"""
    
    chunks_data = []
    metadata_list = []
    
    print(f"📦 Converting {len(chunk_files)} chunks for {video_name}...")
    
    for chunk_file in tqdm(chunk_files, desc=f"Loading {video_name}"):
        try:
            # Load chunk data
            chunk_data = load_chunk_file(chunk_file)
            chunks_data.append(chunk_data)
            
            # Parse metadata
            metadata = parse_chunk_path(chunk_file)
            if metadata:
                metadata_list.append(metadata)
            else:
                print(f"⚠️  Warning: Could not parse metadata from {chunk_file}")
                
        except Exception as e:
            print(f"❌ Error loading {chunk_file}: {e}")
            continue
    
    if not chunks_data:
        print(f"❌ No valid chunks found for {video_name}")
        return False
        
    # Convert to numpy array
    chunks_array = np.array(chunks_data, dtype=np.uint8)
    
    # Save as compressed NPZ
    output_file = os.path.join(output_path, f"{video_name}.npz")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving to {output_file}...")
    np.savez_compressed(
        output_file,
        chunks=chunks_array,
        metadata=metadata_list
    )
    
    # Report savings
    original_files = len(chunk_files)
    compressed_size_mb = os.path.getsize(output_file) / (1024*1024)
    
    print(f"✅ {video_name}: {original_files} files → 1 NPZ ({compressed_size_mb:.1f} MB)")
    return True


def convert_chunks_folder(chunks_folder: str, output_folder: str):
    """Convert entire chunks folder to NPZ format"""
    
    print(f"🔄 Converting chunks from {chunks_folder} to {output_folder}")
    
    # Find all chunk files
    chunk_pattern = os.path.join(chunks_folder, "*/*/*/*.png")
    all_chunks = glob.glob(chunk_pattern)
    
    if not all_chunks:
        print(f"❌ No chunk files found in {chunks_folder}")
        return
        
    print(f"📊 Found {len(all_chunks)} chunk files")
    
    # Group by video name (first directory level)
    videos = {}
    for chunk_file in all_chunks:
        video_name = Path(chunk_file).parts[-4]  # Extract video name
        if video_name not in videos:
            videos[video_name] = []
        videos[video_name].append(chunk_file)
    
    print(f"📹 Found {len(videos)} videos to convert")
    
    # Convert each video
    success_count = 0
    for video_name, chunk_files in videos.items():
        if convert_video_chunks(video_name, chunk_files, output_folder):
            success_count += 1
    
    print(f"🎉 Successfully converted {success_count}/{len(videos)} videos")
    print(f"💾 NPZ files saved to: {output_folder}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_chunks_to_npz.py <chunks_folder> <orig_chunks_folder>")
        print("Example: python convert_chunks_to_npz.py /mnt/d/data_mgr/chunks /mnt/d/data_mgr/orig_chunks")
        sys.exit(1)
    
    chunks_folder = sys.argv[1]
    orig_chunks_folder = sys.argv[2]
    
    # Create output folders
    chunks_npz = chunks_folder + "_npz"
    orig_chunks_npz = orig_chunks_folder + "_npz"
    
    print("🚀 CHUNK TO NPZ CONVERTER")
    print("=" * 50)
    print(f"Input chunks: {chunks_folder}")
    print(f"Input orig chunks: {orig_chunks_folder}")
    print(f"Output chunks NPZ: {chunks_npz}")
    print(f"Output orig chunks NPZ: {orig_chunks_npz}")
    print("=" * 50)
    
    # Convert both folders
    print("\n1️⃣ Converting main chunks...")
    convert_chunks_folder(chunks_folder, chunks_npz)
    
    print("\n2️⃣ Converting orig chunks...")
    convert_chunks_folder(orig_chunks_folder, orig_chunks_npz)
    
    print("\n✅ CONVERSION COMPLETE!")
    print(f"📁 Original folders: {chunks_folder}, {orig_chunks_folder}")
    print(f"📦 NPZ folders: {chunks_npz}, {orig_chunks_npz}")
    print("\n⚠️  IMPORTANT: Test the NPZ files before deleting originals!")
    print("   Example test: python -c \"import numpy as np; data=np.load('chunks_npz/video.npz'); print(data['chunks'].shape)\"")


if __name__ == "__main__":
    main()