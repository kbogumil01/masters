import torch
import os
import numpy as np
from typing import Tuple, Any
from dataclasses import dataclass, asdict
from pydantic import validate_arguments
from glob import glob
from pathlib import Path
from .config import SubDatasetConfig


@validate_arguments
@dataclass
class Metadata:
    file: str
    profile: str
    qp: int
    alf: bool
    sao: bool
    db: bool
    frame: int
    is_intra: bool
    height: int
    width: int


@validate_arguments
@dataclass
class Chunk:
    position: Tuple[int, int]
    corner: str
    metadata: Any


def chunk_to_tuple(chunk: Chunk) -> Tuple:
    return (
        chunk.position[0],
        chunk.position[1],
        chunk.corner,
        chunk.metadata.file,
        chunk.metadata.profile,
        chunk.metadata.qp,
        chunk.metadata.alf,
        chunk.metadata.sao,
        chunk.metadata.db,
        chunk.metadata.frame,
        chunk.metadata.is_intra,
        chunk.metadata.height,
        chunk.metadata.width,
    )


def metadata_to_tuple(metadata: Metadata) -> Tuple:
    return (
        metadata.file,
        metadata.profile,
        metadata.qp,
        metadata.alf,
        metadata.sao,
        metadata.db,
        metadata.frame,
        metadata.is_intra,
        metadata.height,
        metadata.width,
    )


def chunk_from_tuple(data: Tuple) -> Chunk:
    return Chunk(
        position=(data[0], data[1]),
        corner=data[2],
        metadata=Metadata(
            file=data[3],
            profile=data[4],
            qp=data[5],
            alf=bool(data[6]),
            sao=bool(data[7]),
            db=bool(data[8]),
            frame=data[9],
            is_intra=bool(data[10]),
            height=data[11],
            width=data[12],
        ),
    )


def metadata_from_tuple(data: Tuple) -> Metadata:
    return Metadata(
        file=data[0],
        profile=data[1],
        qp=data[2],
        alf=bool(data[3]),
        sao=bool(data[4]),
        db=bool(data[5]),
        frame=data[6],
        is_intra=bool(data[7]),
        height=data[8],
        width=data[9],
    )


class VVCDataset(torch.utils.data.Dataset):
    """
    Custom DataSet loader

    It handles extracting features from each frame
    decoding YUV files etc, pretty time consuming tasks
    """

    CHUNK_GLOB = "{folder}/*/*/*/*.png"
    CHUNK_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{is_intra}/{position[0]}_{position[1]}_{corner}.png"
    ORIG_CHUNK_NAME = "{file}/{frame}_{position[0]}_{position[1]}.png"

    def __init__(
        self,
        settings: SubDatasetConfig,
        chunk_transform: Any,
        metadata_transform: Any,
        fused_maps_dir: str = None,  # NEW: Path to fused_maps directory
    ) -> None:
        super().__init__()

        self.chunk_folder = settings.chunk_folder
        self.orig_chunk_folder = settings.orig_chunk_folder
        self.fused_maps_dir = fused_maps_dir  # NEW

        self.chunk_height = settings.chunk_height
        self.chunk_width = settings.chunk_width

        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform

        self.chunk_files = glob(self.CHUNK_GLOB.format(folder=self.chunk_folder))
        
        # NEW: Cache for fused maps to avoid reloading
        self._fused_maps_cache = {} if fused_maps_dir else None

    def get_chunk(self, fname: str) -> Chunk:
        """load_chunks.
        Loads list of available chunks

        :rtype: List[Chunk]
        """
        fname, profiles, frame, position = fname.split("/")[-4:]
        profile, qp, alf, db, sao = profiles.split("_")
        frame, is_intra = frame.split("_")
        pos0, pos1, corner = position.split(".")[0].split("_")

        metadata = Metadata(
            file=fname,
            profile=profile,
            qp=int(qp[2:]),
            alf=bool(int(alf[3:])),
            sao=bool(int(sao[3:])),
            db=bool(int(db[2:])),
            frame=int(frame),
            is_intra=is_intra == "True",
            height=0,
            width=0,
        )
        chunk = Chunk(
            position=(int(pos0), int(pos1)),
            metadata=metadata,
            corner=corner,
        )
        return chunk

    def load_chunk(self, chunk: Chunk) -> Tuple[Any, Any, Any]:
        chunk_path = self.CHUNK_NAME.format_map(
            dict(**asdict(chunk), **asdict(chunk.metadata))
        )
        chunk_path = os.path.join(self.chunk_folder, chunk_path)
        orig_chunk_path = self.ORIG_CHUNK_NAME.format_map(
            dict(**asdict(chunk), **asdict(chunk.metadata))
        )
        orig_chunk_path = os.path.join(self.orig_chunk_folder, orig_chunk_path)

        with open(chunk_path, "rb") as f:
            _chunk = np.frombuffer(f.read(), dtype=np.uint8)
            _chunk = np.resize(_chunk, (self.chunk_height, self.chunk_width, 3))

        with open(orig_chunk_path, "rb") as f:
            orig_chunk = np.frombuffer(f.read(), dtype=np.uint8)
            orig_chunk = np.resize(orig_chunk, (self.chunk_height, self.chunk_width, 3))

        return (_chunk, orig_chunk, self._metadata_to_np(chunk.metadata))

    @classmethod
    def save_chunk(cls, chunk: Tuple, chunk_data: Any, saved_chunk_folder: str) -> Any:
        chunk = chunk_from_tuple(chunk)
        chunk_path = cls.CHUNK_NAME.format_map(
            dict(**asdict(chunk), **asdict(chunk.metadata))
        )
        chunk_path = os.path.join(saved_chunk_folder, chunk_path)
        folder = os.path.dirname(chunk_path)
        Path(folder).mkdir(parents=True, exist_ok=True)

        with open(chunk_path, "wb") as f:
            f.write(chunk_data.tobytes())

    def _metadata_to_np(self, metadata: Metadata) -> Any:
        """
        Numpy representation of metadata
        """
        return np.array(
            (
                0 if metadata.profile == "RA" else 1,
                metadata.qp / 64,
                metadata.alf,
                metadata.sao,
                metadata.db,
                metadata.is_intra,
            )
        )

    def _load_vvc_features(self, chunk_obj: Chunk) -> torch.Tensor:
        """Load VVC features for a specific chunk"""
        if not self.fused_maps_dir:
            # Return zeros if no VVC features available
            return torch.zeros(13, self.chunk_height, self.chunk_width, dtype=torch.float32)
        
        try:
            # Extract metadata from chunk
            sequence_name = chunk_obj.metadata.file
            poc = chunk_obj.metadata.frame  # FIXED: use metadata.frame
            y_pos, x_pos = chunk_obj.position  # position is (y, x)
            
            # Create cache key
            cache_key = f"{sequence_name}_poc{poc}"
            
            # Load fused maps (with caching)
            if cache_key not in self._fused_maps_cache:
                fused_map_pattern = os.path.join(
                    self.fused_maps_dir, 
                    f"*{sequence_name}*", 
                    "fused_maps", 
                    f"fused_maps_poc{poc}.npz"
                )
                fused_map_files = glob(fused_map_pattern)
                
                if not fused_map_files:
                    # Fallback: try direct path
                    fused_map_file = os.path.join(
                        self.fused_maps_dir,
                        f"fused_maps_poc{poc}.npz"
                    )
                    if not os.path.exists(fused_map_file):
                        print(f"Warning: No fused maps found for {sequence_name} POC {poc}")
                        return torch.zeros(13, self.chunk_height, self.chunk_width, dtype=torch.float32)
                    fused_map_files = [fused_map_file]
                
                # Load and cache
                fused_data = np.load(fused_map_files[0])
                self._fused_maps_cache[cache_key] = fused_data
            else:
                fused_data = self._fused_maps_cache[cache_key]
            
            # Extract chunk region (132x132) from each map
            map_names = [
                'y_ac_energy', 'y_nz_density', 'y_dc',
                'boundary_bin', 'boundary_weight', 'size_map_norm',
                'block_energy_contrast', 'quantization_severity', 'boundary_energy_drop',
                'block_size_category', 'complexity_mismatch', 'ac_density_per_block', 'dc_variation'
            ]
            
            chunk_maps = []
            for map_name in map_names:
                if map_name in fused_data:
                    full_map = fused_data[map_name]  # Shape: (H, W)
                    
                    # Extract chunk region: (y_pos:y_pos+132, x_pos:x_pos+132)
                    map_height, map_width = full_map.shape
                    
                    # Clamp positions to map bounds
                    y_start = max(0, min(y_pos, map_height - self.chunk_height))
                    y_end = min(map_height, y_start + self.chunk_height)
                    x_start = max(0, min(x_pos, map_width - self.chunk_width))
                    x_end = min(map_width, x_start + self.chunk_width)
                    
                    chunk_map = full_map[y_start:y_end, x_start:x_end]
                    
                    # Pad if necessary (in case we're at edges)
                    if chunk_map.shape != (self.chunk_height, self.chunk_width):
                        padded = np.zeros((self.chunk_height, self.chunk_width), dtype=chunk_map.dtype)
                        h, w = chunk_map.shape
                        padded[:h, :w] = chunk_map
                        chunk_map = padded
                    
                    chunk_maps.append(chunk_map)
                else:
                    # Missing map - fill with zeros
                    chunk_maps.append(np.zeros((self.chunk_height, self.chunk_width), dtype=np.float32))
            
            # Stack into (13, H, W) tensor
            vvc_tensor = torch.from_numpy(np.stack(chunk_maps, axis=0)).float()
            return vvc_tensor
            
        except Exception as e:
            print(f"Error loading VVC features for chunk {chunk_obj}: {e}")
            return torch.zeros(13, self.chunk_height, self.chunk_width, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.chunk_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any, Any]:
        chunk_obj = self.get_chunk(self.chunk_files[idx])
        chunk, orig_chunk, metadata = self.load_chunk(chunk_obj)
        
        # NEW: Load VVC features if available
        vvc_features = self._load_vvc_features(chunk_obj) if self.fused_maps_dir else None
        
        return (
            self.chunk_transform(chunk),
            self.chunk_transform(orig_chunk),
            self.metadata_transform(metadata),
            chunk_to_tuple(chunk_obj),
            vvc_features,  # NEW: VVC features as 5th element
        )


class FrameDataset(torch.utils.data.Dataset):
    FRAME_GLOB = "{folder}/*__*__*/*/*.png"
    FRAME_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{is_intra}.png"
    ORIG_FRAME_NAME = "{file}/{frame}.png"

    def __init__(
        self,
        settings: SubDatasetConfig,
        chunk_transform: Any,
        metadata_transform: Any,
    ) -> None:
        super().__init__()

        self.chunk_folder = settings.chunk_folder
        self.orig_chunk_folder = settings.orig_chunk_folder

        self.chunk_height = settings.chunk_height
        self.chunk_width = settings.chunk_width

        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform

        path = self.FRAME_GLOB.format(folder=self.chunk_folder)
        self.frame_files = glob(path)

    def get_metadata(self, fname: str) -> Metadata:
        """load_chunks.
        Loads list of available chunks

        :rtype: List[Chunk]
        """
        fname, profiles, frame = fname.split("/")[-3:]
        frame = frame.split(".")[0]
        _, heigh, width = fname.split("__")
        profile, qp, alf, db, sao = profiles.split("_")
        frame, is_intra = frame.split("_")

        metadata = Metadata(
            file=fname,
            profile=profile,
            qp=int(qp[2:]),
            alf=bool(int(alf[3:])),
            sao=bool(int(sao[3:])),
            db=bool(int(db[2:])),
            frame=int(frame),
            is_intra=is_intra == "True",
            height=int(heigh),
            width=int(width),
        )
        return metadata

    def _metadata_to_np(self, metadata: Metadata) -> Any:
        """
        Numpy representation of metadata
        """
        return np.array(
            (
                0 if metadata.profile == "RA" else 1,
                metadata.qp / 64,
                metadata.alf,
                metadata.sao,
                metadata.db,
                metadata.is_intra,
            )
        )

    def load_frame(self, metadata: Metadata) -> Tuple[Any, Any, Any]:
        mdict = dict(**asdict(metadata))
        frame_path = self.FRAME_NAME.format_map(
            mdict,
        )
        frame_path = os.path.join(self.chunk_folder, frame_path)

        mdict["file"] = metadata.file.split("__")[0]
        orig_frame_path = self.ORIG_FRAME_NAME.format_map(mdict)
        orig_frame_path = os.path.join(self.orig_chunk_folder, orig_frame_path)

        with open(frame_path, "rb") as f:
            _frame = np.frombuffer(f.read(), dtype=np.uint8)
            _frame = np.resize(_frame, (metadata.height, metadata.width, 3))

        with open(orig_frame_path, "rb") as f:
            orig_frame = np.frombuffer(f.read(), dtype=np.uint8)
            orig_frame = np.resize(orig_frame, (metadata.height, metadata.width, 3))

        return (_frame, orig_frame, self._metadata_to_np(metadata))

    @classmethod
    def save_frame(cls, meta: Tuple, frame_data: Any, saved_frame_folder: str) -> Any:
        metadata = metadata_from_tuple(meta)
        mdict = dict(**asdict(metadata))

        frame_path = cls.FRAME_NAME.format_map(mdict)

        frame_path = os.path.join(saved_frame_folder, frame_path)
        folder = os.path.dirname(frame_path)
        Path(folder).mkdir(parents=True, exist_ok=True)

        with open(frame_path, "wb") as f:
            f.write(frame_data.tobytes())

    def __len__(self) -> int:
        return len(self.frame_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        metadata = self.get_metadata(self.frame_files[idx])
        frame, orig_frame, meta = self.load_frame(metadata)
        return (
            self.chunk_transform(frame),
            self.chunk_transform(orig_frame),
            self.metadata_transform(meta),
            metadata_to_tuple(metadata),
        )


if __name__ == "__main__":
    dataset = FrameDataset(
        SubDatasetConfig(
            chunk_folder="test_chunks", orig_chunk_folder="test_orig_chunks"
        ),
        lambda x: x,
        lambda x: x,
    )
    print(dataset[0])
    print(dataset[1][0].shape)
    print(dataset[1][1].shape)
