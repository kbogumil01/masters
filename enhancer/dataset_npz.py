import numpy as np
import torch
import os
import glob
import pickle
from typing import Tuple, Any, List
from dataclasses import dataclass
from pathlib import Path
from .config import SubDatasetConfig


@dataclass
class NPZMetadata:
    """Metadata for NPZ chunks - OPTIMIZED for ALL_INTRA AI-only"""
    file: str
    qp: int
    alf: bool
    sao: bool
    db: bool
    frame: int
    height: int
    width: int
    position: Tuple[int, int]
    corner: str
    # REMOVED: profile (always AI), is_intra (always True)


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
        split: str = 'train',  # NEW: 'train', 'val', or 'test'
        train_ratio: float = 0.8,  # NEW: 80% train
        val_ratio: float = 0.1,    # NEW: 10% val (10% test remains)
    ) -> None:
        super().__init__()

        self.chunk_folder = settings.chunk_folder
        self.orig_chunk_folder = settings.orig_chunk_folder
        self.fused_maps_dir = fused_maps_dir
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.chunk_height = settings.chunk_height
        self.chunk_width = settings.chunk_width

        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform

        # Load all NPZ files and build index
        self._load_npz_index()
        
        # Cache for fused maps (VVC features are still cached per video)
        self._fused_maps_cache = {} if fused_maps_dir else None
        
        # NEW: Cache glob results to avoid filesystem overhead
        self._fused_file_cache = {} if fused_maps_dir else None
        
        # SIMPLIFIED: Direct mmap access without cache (files on fast SSD)
        # Each access will use mmap - OS handles caching efficiently!
        self._npz_handles = {}  # Keep handles open for mmap
        
        # Cache empty VVC tensor for baseline (avoid recreating every time)
        self._empty_vvc = torch.zeros(13, self.chunk_height, self.chunk_width) if not fused_maps_dir else None

    def _load_npz_index(self):
        """Build index without loading data into memory (lazy loading)"""
        print("🔄 Building NPZ chunk index (lazy loading)...")
        
        # Find all NPZ files
        npz_pattern = os.path.join(self.chunk_folder, "*.npz")
        npz_files = sorted(glob.glob(npz_pattern))
        
        if not npz_files:
            raise RuntimeError(f"No NPZ files found in {self.chunk_folder}")
        
        # NEW: Instead of loading data, store file paths and indices
        self.chunk_index = []  # List of (npz_path, orig_npz_path, chunk_idx, metadata)
        
        total_chunks = 0
        for npz_file in npz_files:
            # Only load metadata (lightweight)
            with np.load(npz_file, allow_pickle=True) as chunks_npz:
                metadata = chunks_npz['metadata']
                num_chunks = len(metadata)
            
            # Extract video name for orig_chunks
            chunk_basename = os.path.basename(npz_file)
            video_name = chunk_basename.split('_AI_')[0] if '_AI_' in chunk_basename else chunk_basename.split('_RA_')[0]
            orig_npz_file = os.path.join(self.orig_chunk_folder, f"{video_name}.npz")
            
            if not os.path.exists(orig_npz_file):
                print(f"⚠️  Warning: No corresponding orig chunks for {npz_file}")
                continue
            
            # Build index entries
            for chunk_idx, meta_dict in enumerate(metadata):
                npz_meta = NPZMetadata(
                    file=meta_dict['file'],
                    qp=meta_dict['qp'],
                    alf=meta_dict['alf'],
                    sao=meta_dict['sao'],
                    db=meta_dict['db'],
                    frame=meta_dict['frame'],
                    height=meta_dict['height'],
                    width=meta_dict['width'],
                    position=tuple(meta_dict['position']),
                    corner=meta_dict['corner']
                )
                
                self.chunk_index.append({
                    'npz_path': npz_file,
                    'orig_npz_path': orig_npz_file,
                    'chunk_idx': chunk_idx,
                    'metadata': npz_meta
                })
            
            total_chunks += num_chunks
            print(f"✅ Indexed {num_chunks} chunks from {chunk_basename}")
        
        print(f"🎉 Total chunks indexed: {total_chunks}")
        print(f"� Memory usage: ~{len(self.chunk_index) * 0.001:.1f} MB (index only, lazy loading)")
        
        # NEW: Apply train/val/test split
        if self.split != 'all':
            self._apply_split()
    
    def _apply_split(self):
        """Apply train/val/test split based on video names (deterministic)"""
        import hashlib
        
        # Group chunks by video name
        video_chunks = {}
        for i, entry in enumerate(self.chunk_index):
            video = entry['metadata'].file
            if video not in video_chunks:
                video_chunks[video] = []
            video_chunks[video].append(i)
        
        # Sort videos by name for determinism
        sorted_videos = sorted(video_chunks.keys())
        
        # Split videos deterministically using hash
        train_videos = []
        val_videos = []
        test_videos = []
        
        for video in sorted_videos:
            # Hash video name to get deterministic split
            hash_val = int(hashlib.md5(video.encode()).hexdigest(), 16)
            ratio = (hash_val % 100) / 100.0  # 0.00 to 0.99
            
            if ratio < self.train_ratio:
                train_videos.append(video)
            elif ratio < self.train_ratio + self.val_ratio:
                val_videos.append(video)
            else:
                test_videos.append(video)
        
        # Select indices based on split
        if self.split == 'train':
            selected_videos = train_videos
        elif self.split == 'val':
            selected_videos = val_videos
        elif self.split == 'test':
            selected_videos = test_videos
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Get indices for selected videos
        selected_indices = []
        for video in selected_videos:
            selected_indices.extend(video_chunks[video])
        
        # Filter chunk index
        self.chunk_index = [self.chunk_index[i] for i in selected_indices]
        
        print(f"📂 Split '{self.split}': {len(self.chunk_index)} chunks from {len(selected_videos)} videos")
        print(f"   Train videos: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")

    def _metadata_to_np(self, metadata: NPZMetadata) -> np.ndarray:
        """Convert metadata to numpy array - OPTIMIZED for ALL_INTRA AI-only data
        
        Only includes features that vary across samples:
        - QP: Main compression quality parameter (varies)
        - ALF, SAO, DB: Filter on/off flags (vary)
        
        Excluded constant features (no learning value):
        - profile: Always AI for ALL_INTRA data
        - is_intra: Always True for ALL_INTRA data
        """
        return np.array([
            metadata.qp / 64,     # Normalized QP (0-1 range) - MAIN SIGNAL!
            metadata.alf,         # Boolean as float - ALF filter on/off
            metadata.sao,         # Boolean as float - SAO filter on/off  
            metadata.db,          # Boolean as float - DB filter on/off
        ], dtype=np.float32)

    def _load_vvc_features(self, metadata: NPZMetadata) -> torch.Tensor:
        """Load VVC features - FIXED for ALL_INTRA AI-only data"""
        if not self.fused_maps_dir:
            # Return cached empty tensor (avoid recreating)
            return self._empty_vvc
            
        # FIXED: Build cache key without removed fields
        cache_key = f"{metadata.file}_AI_QP{metadata.qp}_ALF{int(metadata.alf)}_DB{int(metadata.db)}_SAO{int(metadata.sao)}"
        
        if cache_key in self._fused_maps_cache:
            fused_maps = self._fused_maps_cache[cache_key]
        else:
            # MEMORY SAFETY: Limit fused_maps cache size
            if len(self._fused_maps_cache) >= 20:  # Max 20 videos cached
                # Remove oldest entry
                oldest = next(iter(self._fused_maps_cache))
                del self._fused_maps_cache[oldest]
            
            # Load fused maps for this video configuration
            # FIXED: Cache glob results to avoid repeated filesystem calls
            fused_dir = os.path.join(self.fused_maps_dir, cache_key, "fused_maps")
            
            # Check glob cache first
            glob_cache_key = f"{cache_key}_poc{metadata.frame}"
            if glob_cache_key in self._fused_file_cache:
                fused_files = self._fused_file_cache[glob_cache_key]
            else:
                # Try multiple patterns to find the file
                patterns = [
                    f"fused_maps_poc{metadata.frame}.npz",           # poc1.npz (no leading zeros)
                    f"fused_maps_poc{metadata.frame:03d}.npz",       # poc001.npz (3 digits)
                    f"fused_maps_poc{metadata.frame:03d}_*.npz",     # poc001_*.npz (with suffix)
                ]
                
                fused_files = []
                for pattern in patterns:
                    fused_files = glob.glob(os.path.join(fused_dir, pattern))
                    if fused_files:
                        break
                
                # Cache result (even if empty!)
                self._fused_file_cache[glob_cache_key] = fused_files
            
            if fused_files:
                try:
                    fused_data = np.load(fused_files[0], allow_pickle=True)
                except (pickle.UnpicklingError, OSError, ValueError) as e:
                    # File corrupted - return zeros instead of crashing
                    print(f"WARNING: Corrupted fused_maps file {fused_files[0]}, using zeros: {e}")
                    return torch.zeros(13, self.chunk_height, self.chunk_width)
                
                # CRITICAL FIX: Fused maps are stored as separate keys, not single 'fused_maps' tensor
                # Stack all 13 channels in correct order
                channel_keys = [
                    'y_ac_energy',
                    'y_nz_density', 
                    'y_dc',
                    'boundary_bin',
                    'boundary_weight',
                    'size_map_norm',
                    'block_energy_contrast',
                    'quantization_severity',
                    'boundary_energy_drop',
                    'block_size_category',
                    'complexity_mismatch',
                    'ac_density_per_block',
                    'dc_variation'
                ]
                
                # Stack channels into single tensor (13, H, W)
                channels = [fused_data[key] for key in channel_keys]
                fused_maps = torch.from_numpy(np.stack(channels, axis=0)).float()
                
                # CRITICAL: Apply same padding as in split_to_chunks.py
                # Padding formula from split_to_chunks.py:
                # top = chunk_border
                # bottom = 2 * chunk_border + ((-height) % (chunk_height - 2 * chunk_border))
                # left = chunk_border
                # right = 2 * chunk_border + ((-width) % (chunk_width - 2 * chunk_border))
                _, orig_height, orig_width = fused_maps.shape
                chunk_border = 2
                stride = self.chunk_height - 2 * chunk_border
                
                top = chunk_border
                bottom = 2 * chunk_border + ((-orig_height) % stride)
                left = chunk_border
                right = 2 * chunk_border + ((-orig_width) % stride)
                
                # Apply padding (using torch.nn.functional.pad)
                # pad format: (left, right, top, bottom) for last 2 dimensions
                fused_maps = torch.nn.functional.pad(
                    fused_maps, 
                    (left, right, top, bottom), 
                    mode='constant', 
                    value=0.0
                )
                
                self._fused_maps_cache[cache_key] = fused_maps
            else:
                # No VVC features available - WARNING!
                print(f"⚠️  WARNING: No VVC features found in {fused_dir}")
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
        return len(self.chunk_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple, torch.Tensor]:
        """Get chunk data - Simple NPZ loading with LRU cache"""
        
        # Get index entry
        entry = self.chunk_index[idx]
        npz_path = entry['npz_path']
        orig_npz_path = entry['orig_npz_path']
        chunk_idx = entry['chunk_idx']
        metadata = entry['metadata']
        
        # DISABLED: Worker affinity caused swap thrashing with multiple workers
        # Solution: Use num_workers=1 (simpler, more stable)
        
        # Per-worker NPZ handle caching WITHOUT mmap (safer for multiprocessing)
        # Each worker only loads files assigned to it (via affinity above)
        if npz_path not in self._npz_handles:
            self._npz_handles[npz_path] = np.load(npz_path, allow_pickle=True)
        
        if orig_npz_path not in self._npz_handles:
            self._npz_handles[orig_npz_path] = np.load(orig_npz_path, allow_pickle=True)
        
        # Access cached data - already in RAM from np.load()
        chunk_np = self._npz_handles[npz_path]['chunks'][chunk_idx].copy()
        orig_chunk_np = self._npz_handles[orig_npz_path]['chunks'][chunk_idx].copy()
        
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
        
        # Create chunk_to_tuple equivalent - OPTIMIZED for ALL_INTRA AI
        chunk_tuple = (
            metadata.position[0],
            metadata.position[1], 
            metadata.corner,
            metadata.file,
            "AI",  # Always AI profile
            metadata.qp,
            metadata.alf,
            metadata.sao,
            metadata.db,
            metadata.frame,
            True,  # Always True for ALL_INTRA is_intra
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
        # Remove split parameter if using original dataset
        kwargs.pop('split', None)
        kwargs.pop('train_ratio', None)
        kwargs.pop('val_ratio', None)
        return VVCDataset(**kwargs)