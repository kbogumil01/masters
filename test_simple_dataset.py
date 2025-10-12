import os
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional
import glob


class SimpleVVCDataset(Dataset):
    """Simplified dataset using only VVC fused maps for testing."""
    
    def __init__(self, fused_maps_dir: str, chunk_height: int = 132, chunk_width: int = 132):
        self.fused_maps_dir = fused_maps_dir
        self.chunk_height = chunk_height
        self.chunk_width = chunk_width
        
        # Find all fused map files
        pattern = os.path.join(fused_maps_dir, "**/fused_maps/fused_maps_poc*.npz")
        self.fused_map_files = glob.glob(pattern, recursive=True)
        
        print(f"Found {len(self.fused_map_files)} VVC fused map files")
        if len(self.fused_map_files) == 0:
            print(f"No files found with pattern: {pattern}")
            
    def __len__(self):
        return len(self.fused_map_files)
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
        # Load VVC fused maps
        fused_path = self.fused_map_files[idx]
        vvc_data = np.load(fused_path)
        vvc_features = vvc_data['fused']  # Shape: (132, 132, 13)
        
        # Create dummy RGB data for testing (replace with real data later)
        dummy_rgb = np.random.randint(0, 255, (self.chunk_height, self.chunk_width, 3), dtype=np.uint8)
        dummy_orig_rgb = dummy_rgb.copy()
        
        # Create dummy metadata (6 features)
        dummy_metadata = np.random.randn(6).astype(np.float32)
        
        # Create dummy chunk object
        chunk_obj = {
            'file': 'test_file',
            'frame': 0,
            'position': (0, 0),
            'qp': 37,
            'profile': 'AI'
        }
        
        return dummy_rgb, dummy_orig_rgb, dummy_metadata, chunk_obj, vvc_features


# Test script
if __name__ == "__main__":
    dataset = SimpleVVCDataset("videos/decoded")
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shapes:")
        print(f"  RGB: {sample[0].shape}")
        print(f"  Original RGB: {sample[1].shape}")
        print(f"  Metadata: {sample[2].shape}")
        print(f"  VVC features: {sample[4].shape}")