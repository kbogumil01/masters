import numpy as np
import torch
import os
import glob
from typing import Tuple, Any, List
from dataclasses import dataclass
from pathlib import Path
from .config import SubDatasetConfig


@dataclass
class NPZMetadata:
    """Metadata for NPZ chunks - compatible with original"""
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
    position: Tuple[int, int]
    corner: str


class VVCDatasetNPZ(torch.utils.data.Dataset):
    """
    NPZ-based dataset loader - reads compressed chunk archives instead of individual files
    Compatible with original VVCDataset but much more efficient
    """

    def __init__(
        self,
        settings: SubDatasetConfig,
        chunk_transform: Any,
        metadata_transform: Any,
        fused_maps_dir: str = None,
    ) -> None:
        super().__init__()

        self.chunk_folder = settings.chunk_folder
        self.orig_chunk_folder = settings.orig_chunk_folder
        self.fused_maps_dir = fused_maps_dir

        self.chunk_height = settings.chunk_height
        self.chunk_width = settings.chunk_width

        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform

        # Load all NPZ files and build index
        self._load_npz_index()
        
        # Cache for fused maps
        self._fused_maps_cache = {} if fused_maps_dir else None

    def _load_npz_index(self):
        """Load all NPZ files and create index for fast access"""
        print("🔄 Loading NPZ chunk index...")
        
        # Find all NPZ files
        npz_pattern = os.path.join(self.chunk_folder, "*.npz")
        npz_files = glob.glob(npz_pattern)
        
        if not npz_files:
            raise RuntimeError(f"No NPZ files found in {self.chunk_folder}")
            
        self.chunks_data = []
        self.orig_chunks_data = []
        self.metadata_list = []
        
        total_chunks = 0
        for npz_file in npz_files:
            # Load chunks
            chunks_npz = np.load(npz_file)
            chunks = chunks_npz['chunks']  # Shape: (N, 132, 132, 3)
            metadata = chunks_npz['metadata']  # List of dicts
            
            # Load corresponding orig chunks
            npz_name = os.path.basename(npz_file)
            orig_npz_file = os.path.join(self.orig_chunk_folder, npz_name)
            
            if os.path.exists(orig_npz_file):
                orig_chunks_npz = np.load(orig_npz_file)
                orig_chunks = orig_chunks_npz['chunks']
                
                # Verify same length
                assert len(chunks) == len(orig_chunks), f"Mismatch in {npz_file}: {len(chunks)} vs {len(orig_chunks)}"
                assert len(chunks) == len(metadata), f"Metadata mismatch in {npz_file}: {len(chunks)} vs {len(metadata)}"
                
                # Add to master lists
                self.chunks_data.extend(chunks)
                self.orig_chunks_data.extend(orig_chunks)
                
                # Convert metadata dicts to NPZMetadata objects
                for meta_dict in metadata:
                    npz_meta = NPZMetadata(
                        file=meta_dict['file'],
                        profile=meta_dict['profile'],
                        qp=meta_dict['qp'],
                        alf=meta_dict['alf'],
                        sao=meta_dict['sao'],
                        db=meta_dict['db'],
                        frame=meta_dict['frame'],
                        is_intra=meta_dict['is_intra'],
                        height=meta_dict['height'],
                        width=meta_dict['width'],
                        position=tuple(meta_dict['position']),
                        corner=meta_dict['corner']
                    )
                    self.metadata_list.append(npz_meta)
                
                total_chunks += len(chunks)
                print(f"✅ Loaded {len(chunks)} chunks from {npz_name}")
            else:
                print(f"⚠️  Warning: No corresponding orig chunks for {npz_file}")
        
        print(f"🎉 Total chunks loaded: {total_chunks}")
        print(f"📊 Memory usage: ~{total_chunks * 132 * 132 * 3 / (1024*1024):.1f} MB")

    def _metadata_to_np(self, metadata: NPZMetadata) -> np.ndarray:
        """Convert metadata to numpy array - IDENTICAL to original"""
        return np.array([
            0 if metadata.profile == "RA" else 1,  # Profile encoding: RA=0, AI=1
            metadata.qp / 64,                      # Normalized QP (0-1 range)
            metadata.alf,                          # Boolean as float
            metadata.sao,                          # Boolean as float  
            metadata.db,                           # Boolean as float
            metadata.is_intra,                     # Boolean as float
        ])

    def _load_vvc_features(self, metadata: NPZMetadata) -> torch.Tensor:
        """Load VVC features - adapted from original dataset"""
        if not self.fused_maps_dir:
            # Return zeros if no VVC features available
            return torch.zeros(13, self.chunk_height, self.chunk_width)
            
        # Build cache key
        cache_key = f"{metadata.file}_{metadata.profile}_QP{metadata.qp}_ALF{int(metadata.alf)}_DB{int(metadata.db)}_SAO{int(metadata.sao)}"
        
        if cache_key in self._fused_maps_cache:
            fused_maps = self._fused_maps_cache[cache_key]
        else:
            # Load fused maps for this video configuration
            fused_pattern = f"fused_maps_poc{metadata.frame:03d}_*.npz"
            fused_dir = os.path.join(self.fused_maps_dir, cache_key, "fused_maps")
            fused_files = glob.glob(os.path.join(fused_dir, fused_pattern))
            
            if fused_files:
                fused_data = np.load(fused_files[0])
                fused_maps = torch.from_numpy(fused_data['fused_maps']).float()
                self._fused_maps_cache[cache_key] = fused_maps
            else:
                # No VVC features available
                return torch.zeros(13, self.chunk_height, self.chunk_width)
        
        # Extract chunk from fused maps
        pos_v, pos_h = metadata.position
        chunk_vvc = fused_maps[:, pos_v:pos_v+self.chunk_height, pos_h:pos_h+self.chunk_width]
        
        # Ensure correct size
        if chunk_vvc.shape[1:] != (self.chunk_height, self.chunk_width):
            chunk_vvc = torch.nn.functional.interpolate(
                chunk_vvc.unsqueeze(0), 
                size=(self.chunk_height, self.chunk_width), 
                mode='bilinear'
            ).squeeze(0)
        
        return chunk_vvc

    def __len__(self) -> int:
        return len(self.chunks_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple, torch.Tensor]:
        """Get chunk data - IDENTICAL format to original dataset"""
        
        # Get data
        chunk_data = self.chunks_data[idx]  # (132, 132, 3)
        orig_chunk_data = self.orig_chunks_data[idx]  # (132, 132, 3)
        metadata = self.metadata_list[idx]
        
        # Convert numpy arrays to format expected by ToTensor() 
        # ToTensor() expects HWC numpy arrays and converts to CHW tensors
        chunk_np = chunk_data.astype(np.uint8)  # Keep as uint8 for ToTensor()
        orig_chunk_np = orig_chunk_data.astype(np.uint8)
        
        # Apply transforms (ToTensor will convert HWC->CHW and normalize 0-1)
        if self.chunk_transform:
            chunk_tensor = self.chunk_transform(chunk_np)
            orig_chunk_tensor = self.chunk_transform(orig_chunk_np)
        else:
            # Fallback: manual conversion
            chunk_tensor = torch.from_numpy(chunk_np).permute(2, 0, 1).float() / 255.0
            orig_chunk_tensor = torch.from_numpy(orig_chunk_np).permute(2, 0, 1).float() / 255.0
        
        # Metadata - same format as original
        metadata_np = self._metadata_to_np(metadata)
        if self.metadata_transform:
            metadata_tensor = self.metadata_transform(metadata_np)
        else:
            metadata_tensor = torch.from_numpy(metadata_np).float()
        
        # VVC features
        vvc_features = self._load_vvc_features(metadata)
        
        # Create chunk_to_tuple equivalent
        chunk_tuple = (
            metadata.position[0],
            metadata.position[1], 
            metadata.corner,
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

        return chunk_tensor, orig_chunk_tensor, metadata_tensor, chunk_tuple, vvc_features


# Make it easy to switch between datasets
def get_vvc_dataset(use_npz: bool = True, **kwargs):
    """Factory function to get either NPZ or original dataset"""
    if use_npz:
        return VVCDatasetNPZ(**kwargs)
    else:
        from .dataset import VVCDataset
        return VVCDataset(**kwargs)