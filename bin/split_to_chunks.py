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


class Splitter:
    """
    Splits videos into chunks for easy processing
    """

    INFO_HEIGHT_REGEX: str = re.compile(r"^\s*Height\s*:\s*(\d+)\s*$")
    INFO_WIDTH_REGEX: str = re.compile(r"^\s*Width\s*:\s*(\d+)\s*$")
    INFO_FRAMES_REGEX: str = re.compile(r"^\s*Frame count\s*:\s*(\d+)\s*$")
    ENCODED_REGEX: str = re.compile(
        "^(?P<name>[\d\s\w]+)_(?P<profile>AI|RA)_QP(?P<qp>\d{2})_ALF(?P<alf>\d{1})_DB(?P<db>\d{1})_SAO(?P<sao>\d{1}).yuv"
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

    def load_intra_frames(self, metadata: Metadata, dirname: str) -> List[int]:
        # Look for decode.log in the directory
        file_path = os.path.join(self.encoded_path, dirname, "decode.log")

        with open(file_path) as f:
            lines = f.read().splitlines()

        lines = [l for l in lines if l.startswith("POC")]
        return {i for i, l in enumerate(lines) if "I-SLICE" in l}

    def split_chunks(self) -> None:
        """
        splits chunks :)
        """
        files = [d for d in os.listdir(self.encoded_path) 
                if os.path.isdir(os.path.join(self.encoded_path, d)) and not d.startswith('.')]
        files = sorted(files)

        try:
            with open(self.done_cache) as f:
                done = f.read().splitlines()
        except:
            done = []

        for dirname in tqdm(files):
            # Skip RA sequences for now (focus on AI)
            if "_RA_" in dirname:
                continue

            if dirname in done:
                continue

            # Check if recon.yuv exists
            recon_path = os.path.join(self.encoded_path, dirname, "recon.yuv")
            if not os.path.exists(recon_path):
                continue

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
            with open(self.done_cache, "a") as f:
                f.write(f"\n{dirname}")
            print(f"DONE {dirname}")

    def load_metadata_for(self, dirname: str) -> Metadata:
        """
        Loads metadata for given directory name
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
        CRITICAL IMPLEMENTATION: Saves chunks as NPZ with proper orig_chunks logic
        
        Key fixes:
        1. orig_chunks saved ONCE per video (not per compression config)  
        2. Separate metadata for orig_chunks (no compression params)
        3. Uses recon.yuv from dirname folder (not old DECODED_FORMAT)
        4. 100% compatibility with dataset_npz.py
        """
        metadata = chunks[0].metadata
        nh = metadata.height * 3 // 2

        orig_file_path = os.path.join(
            self.data_path, self.ORIGINAL_FORMAT.format_map(asdict(metadata))
        )
        # CRITICAL: Use recon.yuv from the specific directory
        file_path = os.path.join(self.encoded_path, dirname, "recon.yuv")

        with open(file_path, "rb") as f:
            buff = np.frombuffer(f.read(), dtype=np.uint8)

        with open(orig_file_path, "rb") as f:
            orig_buff = np.frombuffer(f.read(), dtype=np.uint8)

        buff = np.resize(buff, (metadata.frames, nh * metadata.width))

        orig_buff = np.resize(orig_buff, (metadata.frames, nh * metadata.width))

        # Collect all chunks for this video in memory
        all_chunks = []
        all_orig_chunks = []
        chunk_metadata = []
        orig_chunk_metadata = []

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
                + ((-metadata.width) % (self.chunk_width - 2 * self.chunk_border)),
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
                + ((-metadata.width) % (self.chunk_width - 2 * self.chunk_border)),
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
                
                # COMPRESSED chunks metadata (WITH compression params)
                # OPTIMIZED: Removed profile (always 'AI') and is_intra (always True)
                chunk_info = {
                    'position': chunk.position,
                    'frame': chunk.frame,
                    'corner': chunk.corner,
                    'file': metadata.file,
                    # REMOVED: 'profile': metadata.profile,  # Always 'AI' for ALL_INTRA
                    'qp': metadata.qp,
                    'alf': metadata.alf,
                    'db': metadata.db,
                    'sao': metadata.sao,
                    'width': metadata.width,
                    'height': metadata.height
                }
                chunk_metadata.append(chunk_info)
                
                # ORIGINAL chunks metadata (WITHOUT compression params!)
                orig_chunk_info = {
                    'position': chunk.position,
                    'frame': chunk.frame,
                    # REMOVED: 'is_intra': chunk.is_intra,  # Always True for ALL_INTRA
                    'corner': chunk.corner,
                    'file': metadata.file,
                    'width': metadata.width,
                    'height': metadata.height
                    # INTENTIONALLY NO: profile, qp, alf, db, sao
                }
                orig_chunk_metadata.append(orig_chunk_info)

        # Convert to numpy arrays for efficient storage
        all_chunks = np.array(all_chunks, dtype=np.uint8)
        all_orig_chunks = np.array(all_orig_chunks, dtype=np.uint8)

        # Create output directories
        Path(self.chunk_folder).mkdir(parents=True, exist_ok=True)
        Path(self.orig_chunk_folder).mkdir(parents=True, exist_ok=True)

        # Save compressed chunks (with full compression metadata)
        chunk_file = os.path.join(self.chunk_folder, f"{dirname}.npz")
        print(f"💾 Saving {len(all_chunks)} compressed chunks: {chunk_file}")
        np.savez_compressed(
            chunk_file,
            chunks=all_chunks,
            metadata=chunk_metadata
        )

        # CRITICAL: Save original chunks ONLY ONCE per video
        orig_chunk_file = os.path.join(self.orig_chunk_folder, f"{metadata.file}.npz")
        if not os.path.exists(orig_chunk_file):
            print(f"💾 Saving {len(all_orig_chunks)} original chunks: {orig_chunk_file}")
            np.savez_compressed(
                orig_chunk_file,
                chunks=all_orig_chunks,
                metadata=orig_chunk_metadata
            )
            print(f"✅ NEW original chunks for: {metadata.file}")
        else:
            print(f"⏭️  Original chunks exist, skipping: {orig_chunk_file}")

        print(f"✅ Processed {len(all_chunks)} chunks for {dirname}")
        
        # Report efficiency
        total_size_mb = (all_chunks.nbytes + all_orig_chunks.nbytes) / (1024*1024)
        print(f"📊 Data size: {total_size_mb:.1f} MB in NPZ format")

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


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    
    # Basic required arguments: data_path, encoded_path, chunk_folder, orig_chunk_folder, done_cache
    if len(args) < 5:
        print("Usage: python split_to_chunks.py data_path encoded_path chunk_folder orig_chunk_folder done_cache [chunk_width] [chunk_height] [chunk_border]")
        sys.exit(1)
    
    # Convert optional numeric arguments to int
    if len(args) >= 6:
        args[5] = int(args[5])  # chunk_width
    if len(args) >= 7:
        args[6] = int(args[6])  # chunk_height  
    if len(args) >= 8:
        args[7] = int(args[7])  # chunk_border

    s = Splitter(*args)
    s.split_chunks()