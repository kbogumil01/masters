import os
import re
import math
import numpy as np
import cv2
import json
import glob
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, asdict
from pydantic import validate_arguments
from random import choice
from pathlib import Path
import concurrent.futures
import threading
import multiprocessing
import shutil


@validate_arguments
@dataclass
class Metadata:
    file: str
    width: int
    height: int
    frames: int
    profile: str
    qp: int
    alf: bool
    sao: bool
    db: bool


@validate_arguments
@dataclass
class Chunk:
    position: Tuple[int, int]
    metadata: Any
    frame: int
    is_intra: bool
    corner: str


class ParallelSplitter:
    """
    Parallel version - Splits videos into chunks for easy processing using multiple threads
    """

    INFO_HEIGHT_REGEX: str = re.compile(r"^\s*Height\s*:\s*(\d+)\s*$")
    INFO_WIDTH_REGEX: str = re.compile(r"^\s*Width\s*:\s*(\d+)\s*$")
    INFO_FRAMES_REGEX: str = re.compile(r"^\s*Frame count\s*:\s*(\d+)\s*$")
    ENCODED_REGEX: str = re.compile(
        r"^(?P<name>[\d\s\w]+)_(?P<profile>AI|RA)_QP(?P<qp>\d{2})_ALF(?P<alf>\d{1})_DB(?P<db>\d{1})_SAO(?P<sao>\d{1}).yuv"
    )

    METADATA_FORMAT: str = "{name}.*.info"
    DECODED_FORMAT: str = "{file}_{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}.yuv"
    DECODED_LOG_FORMAT: str = (
        "{file}_{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}.yuv.log"
    )
    ORIGINAL_FORMAT: str = "{file}.yuv"
    FILE_FORMAT: str = "yuv"

    CHUNK_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{is_intra}/{position[0]}_{position[1]}_{corner}.png"
    ORIG_CHUNK_NAME = "{file}/{frame}_{position[0]}_{position[1]}.png"

    def __init__(
        self,
        data_path: str,
        encoded_path: str,
        chunk_folder: str,
        orig_chunk_folder: str,
        done_cache: str,
        chunk_width: int = 132,
        chunk_height: int = 132,
        chunk_border: int = 2,
        frame_folder: str = None,
        orig_frame_folder: str = None,
        auto_cleanup_frames: bool = True,
        max_workers: int = None,
    ) -> None:
        super().__init__()

        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
        self.chunk_border = chunk_border

        self.data_path = data_path
        self.encoded_path = encoded_path

        self.chunk_folder = chunk_folder
        self.orig_chunk_folder = orig_chunk_folder
        self.done_cache = done_cache
        
        # Frame folders for cleanup
        self.frame_folder = frame_folder or "videos/frames"
        self.orig_frame_folder = orig_frame_folder or "videos/orig_frames"
        self.auto_cleanup_frames = auto_cleanup_frames

        # Parallel processing settings
        if max_workers is None:
            # For I/O bound tasks, use more workers than CPU cores
            max_workers = min(multiprocessing.cpu_count() * 2, 8)
        self.max_workers = max_workers
        
        # Thread-safe cache management
        self._cache_lock = threading.Lock()

    def load_intra_frames(self, metadata: Metadata, dirname: str) -> List[int]:
        # Look for decode.log in the directory
        file_path = os.path.join(self.encoded_path, dirname, "decode.log")

        with open(file_path) as f:
            lines = f.read().splitlines()

        lines = [l for l in lines if l.startswith("POC")]
        return {i for i, l in enumerate(lines) if "I-SLICE" in l}

    def _update_done_cache_thread_safe(self, dirname: str) -> None:
        """Thread-safe method to update done cache"""
        with self._cache_lock:
            with open(self.done_cache, "a") as f:
                f.write(f"\n{dirname}")

    def _process_single_video(self, dirname: str) -> str:
        """Process a single video directory - designed to be run in parallel"""
        try:
            # Skip RA sequences for now (focus on AI)
            if "_RA_" in dirname:
                return f"SKIPPED {dirname} (RA profile)"

            # Check if recon.yuv exists
            recon_path = os.path.join(self.encoded_path, dirname, "recon.yuv")
            if not os.path.exists(recon_path):
                return f"SKIPPED {dirname} (no recon.yuv)"

            metadata = self.load_metadata_for(dirname)
            intra_frames = self.load_intra_frames(metadata, dirname)

            horizontal_chunks = math.ceil(
                metadata.width / (self.chunk_width - 2 * self.chunk_border)
            )
            vertical_chunks = math.ceil(
                metadata.height / (self.chunk_height - 2 * self.chunk_border)
            )

            video_chunks = []

            for frame in range(metadata.frames):
                for h, h_part in enumerate(range(horizontal_chunks)):
                    h_pos = h_part * (self.chunk_width - self.chunk_border * 2)

                    for v, v_part in enumerate(range(vertical_chunks)):
                        v_pos = v_part * (self.chunk_height - self.chunk_border * 2)

                        corner = []

                        if h == 0:
                            corner.append("l")
                        if h == horizontal_chunks - 1:
                            corner.append("r")
                        if v == 0:
                            corner.append("u")
                        if v == vertical_chunks - 1:
                            corner.append("b")

                        chunk = Chunk(
                            metadata=metadata,
                            frame=frame,
                            position=(v_pos, h_pos),
                            is_intra=frame in intra_frames or metadata.profile == "AI",
                            corner="".join(corner),
                        )
                        video_chunks.append(chunk)

            self.save_chunks(video_chunks, dirname)
            
            # Automatically clean up frames folder after chunks are created
            if self.auto_cleanup_frames:
                self.cleanup_frames(metadata)
            
            # Thread-safe cache update
            self._update_done_cache_thread_safe(dirname)
            
            return f"DONE {dirname}"
            
        except Exception as e:
            return f"ERROR {dirname}: {str(e)}"

    def split_chunks(self) -> None:
        """
        splits chunks :) - PARALLEL VERSION
        """
        dirs = [d for d in os.listdir(self.encoded_path) 
                if os.path.isdir(os.path.join(self.encoded_path, d)) and not d.startswith('.')]
        dirs = sorted(dirs)

        # Load existing done cache
        try:
            with open(self.done_cache) as f:
                done = set(f.read().splitlines())
        except:
            done = set()

        # Filter out already done directories
        remaining_dirs = [d for d in dirs if d not in done]
        
        if not remaining_dirs:
            print("All videos already processed!")
            return
            
        print(f"Processing {len(remaining_dirs)} videos with {self.max_workers} workers...")

        # Process videos in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_dir = {
                executor.submit(self._process_single_video, dirname): dirname 
                for dirname in remaining_dirs
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(remaining_dirs), desc="Processing videos") as pbar:
                for future in concurrent.futures.as_completed(future_to_dir):
                    dirname = future_to_dir[future]
                    try:
                        result = future.result()
                        print(result)
                        pbar.set_postfix_str(f"Last: {dirname[:20]}...")
                    except Exception as exc:
                        print(f"ERROR processing {dirname}: {exc}")
                    finally:
                        pbar.update(1)

    def load_metadata_for(self, dirname: str) -> Metadata:
        """
        Loads metadata for given directory
        """
        # Parse directory name: deadline_cif_AI_QP37_ALF1_DB0_SAO1
        m = re.match(self.ENCODED_REGEX, dirname + ".yuv")  # Add .yuv for regex match
        assert m, f"Invalid directory name: {dirname} not matching regex: {self.ENCODED_REGEX}"
        match_group = m.groupdict()

        height = width = frames = None

        fname = glob.glob(
            os.path.join(self.data_path, self.METADATA_FORMAT.format_map(match_group))
        )[0]

        with open(fname) as f:
            for line in f.readlines():
                h = re.match(self.INFO_HEIGHT_REGEX, line)
                height = h.groups()[0] if h else height
                w = re.match(self.INFO_WIDTH_REGEX, line)
                width = w.groups()[0] if w else width
                f = re.match(self.INFO_FRAMES_REGEX, line)
                frames = f.groups()[0] if f else frames

        return Metadata(
            file=match_group["name"],
            width=int(width),
            height=int(height),
            frames=64,
            profile=match_group["profile"],
            qp=int(match_group["qp"]),
            alf=bool(int(match_group["alf"])),
            db=bool(int(match_group["db"])),
            sao=bool(int(match_group["sao"])),
        )

    def save_chunks(self, chunks: List[Chunk], dirname: str) -> None:
        """
        Splits chunks and saves them as compressed NPZ files (much more efficient!)
        """
        metadata = chunks[0].metadata
        nh = metadata.height * 3 // 2

        orig_file_path = os.path.join(
            self.data_path, self.ORIGINAL_FORMAT.format_map(asdict(metadata))
        )
        # Point to recon.yuv in the directory
        file_path = os.path.join(self.encoded_path, dirname, "recon.yuv")

        with open(file_path, "rb") as f:
            buff = np.frombuffer(f.read(), dtype=np.uint16)

        with open(orig_file_path, "rb") as f:
            orig_buff = np.frombuffer(f.read(), dtype=np.uint8)

        buff = np.round(buff / 4).astype(np.uint8)
        buff = np.resize(buff, (metadata.frames, nh * metadata.width))

        orig_buff = np.resize(orig_buff, (metadata.frames, nh * metadata.width))

        # Collect all chunks for this video in memory
        all_chunks = []
        all_orig_chunks = []
        chunk_metadata = []

        for frame_num in tqdm(range(metadata.frames), desc=f"Processing {dirname[:20]}"):
            frame = buff[frame_num]
            frame = self.upsample_uv(frame, metadata.width, metadata.height)
            frame = cv2.copyMakeBorder(
                frame,
                self.chunk_border,
                2 * self.chunk_border
                + ((-metadata.height) % (self.chunk_height - 2 * self.chunk_border)),
                self.chunk_border,
                2 * self.chunk_border
                + ((-metadata.width) % (self.chunk_height - 2 * self.chunk_border)),
                cv2.BORDER_CONSTANT,
                value=0.0,
            )

            orig_frame = orig_buff[frame_num]
            orig_frame = self.upsample_uv(orig_frame, metadata.width, metadata.height)
            orig_frame = cv2.copyMakeBorder(
                orig_frame,
                self.chunk_border,
                2 * self.chunk_border
                + ((-metadata.height) % (self.chunk_height - 2 * self.chunk_border)),
                self.chunk_border,
                2 * self.chunk_border
                + ((-metadata.width) % (self.chunk_height - 2 * self.chunk_border)),
                cv2.BORDER_CONSTANT,
                value=0.0,
            )

            for chunk in (c for c in chunks if c.frame == frame_num):
                start_h = chunk.position[0]
                start_w = chunk.position[1]

                chunk_h = self.chunk_height
                chunk_w = self.chunk_width

                # Extract chunk data
                frame_chunk = frame[start_h:, start_w:, :][:chunk_h, :chunk_w, :]
                orig_frame_chunk = orig_frame[start_h:, start_w:, :][:chunk_h, :chunk_w, :]

                # Add to arrays
                all_chunks.append(frame_chunk)
                all_orig_chunks.append(orig_frame_chunk)
                
                # Store metadata for this chunk
                chunk_info = {
                    'position': chunk.position,
                    'frame': chunk.frame,
                    'is_intra': chunk.is_intra,
                    'corner': chunk.corner,
                    'file': metadata.file,
                    'profile': metadata.profile,
                    'qp': metadata.qp,
                    'alf': metadata.alf,
                    'db': metadata.db,
                    'sao': metadata.sao,
                    'width': metadata.width,
                    'height': metadata.height
                }
                chunk_metadata.append(chunk_info)

        # Convert to numpy arrays for efficient storage
        all_chunks = np.array(all_chunks, dtype=np.uint8)
        all_orig_chunks = np.array(all_orig_chunks, dtype=np.uint8)

        # Create output directories
        Path(self.chunk_folder).mkdir(parents=True, exist_ok=True)
        Path(self.orig_chunk_folder).mkdir(parents=True, exist_ok=True)

        # Save as compressed NPZ files (one per video)
        chunk_file = os.path.join(self.chunk_folder, f"{dirname}.npz")
        orig_chunk_file = os.path.join(self.orig_chunk_folder, f"{dirname}.npz")

        print(f"💾 Saving {len(all_chunks)} chunks to NPZ: {chunk_file}")
        np.savez_compressed(
            chunk_file,
            chunks=all_chunks,
            metadata=chunk_metadata
        )

        print(f"💾 Saving {len(all_orig_chunks)} orig chunks to NPZ: {orig_chunk_file}")
        np.savez_compressed(
            orig_chunk_file,
            chunks=all_orig_chunks,
            metadata=chunk_metadata
        )

        print(f"✅ Saved {len(all_chunks)} chunks in 2 NPZ files instead of {len(all_chunks)*2} individual files!")
        
        # Report compression efficiency
        total_size_mb = (all_chunks.nbytes + all_orig_chunks.nbytes) / (1024*1024)
        print(f"📊 Total data: {total_size_mb:.1f} MB compressed in NPZ format")

    def upsample_uv(self, frame_buffer, width, height):
        i = width * height
        Y = frame_buffer[0:i]
        Y = np.reshape(Y, (height, width))

        uv_size = width * height // 4
        V = frame_buffer[i : i + uv_size]
        V = np.reshape(V, (height // 2, width // 2))
        V = cv2.resize(V, (width, height))

        i += uv_size
        U = frame_buffer[i:]
        U = np.reshape(U, (height // 2, width // 2))
        U = cv2.resize(U, (width, height))

        return np.dstack([Y, U, V])

    def cleanup_frames(self, metadata: Metadata) -> None:
        """
        Remove frame files after chunks are created to save disk space
        """        
        print(f"🧹 Cleaning up frames for {metadata.file}...")
        
        # Pattern for frames: videos/frames/{file}__{height}__{width}/{profile}_QP{qp}_ALF{alf}_DB{db}_SAO{sao}/
        frame_dir_pattern = f"{metadata.file}__{metadata.height}__{metadata.width}"
        
        # Cleanup main frames
        frame_dir_path = os.path.join(self.frame_folder, frame_dir_pattern)
        if os.path.exists(frame_dir_path):
            try:
                shutil.rmtree(frame_dir_path)
                print(f"✅ Cleaned up frames directory: {frame_dir_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not cleanup frames directory {frame_dir_path}: {e}")
        
        # Cleanup orig_frames
        orig_frame_dir_path = os.path.join(self.orig_frame_folder, metadata.file)
        if os.path.exists(orig_frame_dir_path):
            try:
                shutil.rmtree(orig_frame_dir_path)
                print(f"✅ Cleaned up orig frames directory: {orig_frame_dir_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not cleanup orig frames directory {orig_frame_dir_path}: {e}")

        # Also try to remove empty parent directories
        try:
            # Remove empty frame_folder if it's empty
            if os.path.exists(self.frame_folder) and not os.listdir(self.frame_folder):
                os.rmdir(self.frame_folder)
                print(f"✅ Removed empty frames folder: {self.frame_folder}")
        except:
            pass  # Ignore if not empty or other errors
            
        try:
            # Remove empty orig_frame_folder if it's empty  
            if os.path.exists(self.orig_frame_folder) and not os.listdir(self.orig_frame_folder):
                os.rmdir(self.orig_frame_folder)
                print(f"✅ Removed empty orig frames folder: {self.orig_frame_folder}")
        except:
            pass  # Ignore if not empty or other errors


# Backward compatibility - alias for the original class name
Splitter = ParallelSplitter


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    
    # Basic required arguments: data_path, encoded_path, chunk_folder, orig_chunk_folder, done_cache
    if len(args) < 5:
        print("Usage: python split_to_chunks_parallel.py data_path encoded_path chunk_folder orig_chunk_folder done_cache [chunk_width] [chunk_height] [chunk_border] [frame_folder] [orig_frame_folder] [auto_cleanup] [max_workers]")
        sys.exit(1)
    
    # Convert optional numeric arguments to int
    if len(args) >= 6:
        args[5] = int(args[5])  # chunk_width
    if len(args) >= 7:
        args[6] = int(args[6])  # chunk_height  
    if len(args) >= 8:
        args[7] = int(args[7])  # chunk_border
    if len(args) >= 11:
        args[10] = args[10].lower() == 'true'  # auto_cleanup
    if len(args) >= 12:
        args[11] = int(args[11])  # max_workers

    s = ParallelSplitter(*args)
    s.split_chunks()
