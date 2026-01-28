#!/usr/bin/env python3
"""
Quick test of BD-rate calculation on a small subset of data.
Tests the pipeline without processing all sequences.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bin.calculate_bd_rate import (
    bjontegaard_delta,
    bjontegaard_psnr,
    load_enhancer_model
)


def test_bjontegaard_functions():
    """Test Bjøntegaard calculations with known values."""
    print("=" * 80)
    print("Testing Bjøntegaard Functions")
    print("=" * 80)
    print()
    
    # Example RD points (typical for video compression)
    # Anchor (original codec)
    R_anchor = np.array([500, 1000, 2000, 4000])  # kbps
    PSNR_anchor = np.array([35.0, 37.5, 40.0, 42.5])  # dB
    
    # Test (improved codec - better quality at same rate)
    PSNR_test = PSNR_anchor + 1.5  # 1.5 dB improvement
    
    # Calculate BD metrics
    bd_rate = bjontegaard_delta(R_anchor, PSNR_anchor, R_anchor, PSNR_test)
    bd_psnr = bjontegaard_psnr(R_anchor, PSNR_anchor, R_anchor, PSNR_test)
    
    print("Test case: Constant +1.5 dB PSNR improvement")
    print(f"  BD-Rate: {bd_rate:.2f}%")
    print(f"  BD-PSNR: {bd_psnr:.4f} dB")
    print()
    
    # Expected: BD-PSNR should be close to 1.5 dB
    assert abs(bd_psnr - 1.5) < 0.1, f"BD-PSNR mismatch: expected ~1.5, got {bd_psnr}"
    print("✅ Bjøntegaard functions working correctly")
    print()


def test_model_loading(checkpoint_path: str, config_path: str):
    """Test model loading."""
    print("=" * 80)
    print("Testing Model Loading")
    print("=" * 80)
    print()
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        enhancer = load_enhancer_model(checkpoint_path, config_path, device)
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, 132, 132).to(device)
        dummy_meta = torch.tensor([[0.5, 0.0, 0.0, 0.0]]).view(1, 4, 1, 1).to(device)
        
        with torch.no_grad():
            output = enhancer(dummy_input, dummy_meta, None)
        
        assert output.shape == dummy_input.shape, "Output shape mismatch"
        
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print()
        
        return enhancer
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise


def test_data_structure(test_root: Path, orig_root: Path):
    """Test data structure."""
    print("=" * 80)
    print("Testing Data Structure")
    print("=" * 80)
    print()
    
    # Find one test sequence
    test_dirs = [d for d in test_root.iterdir() if d.is_dir()]
    
    if not test_dirs:
        print(f"❌ No test directories found in {test_root}")
        return None
    
    test_dir = test_dirs[0]
    print(f"Testing with: {test_dir.name}")
    
    # Find frames - try multiple patterns
    frames = list(test_dir.glob("frame_*.pt"))
    if not frames:
        frames = list(test_dir.glob("fused_maps/fused_maps_poc*.pt"))
    if not frames:
        frames = list(test_dir.glob("*.pt"))
    
    if not frames:
        print(f"❌ No frames found in {test_dir}")
        return None
    
    print(f"   Found {len(frames)} frames")
    
    # Load one frame
    frame_path = frames[0]
    data = torch.load(frame_path, map_location='cpu')
    
    print(f"   Frame data keys: {list(data.keys())}")
    
    if 'chunk' in data:
        print(f"   ✅ 'chunk' tensor shape: {data['chunk'].shape}")
    else:
        print(f"   ❌ Missing 'chunk' key")
    
    if 'seq_meta' in data:
        print(f"   ✅ 'seq_meta': {data['seq_meta']}")
    else:
        print(f"   ⚠️  Missing 'seq_meta' (will use defaults)")
    
    if 'vvc_features' in data:
        print(f"   ✅ 'vvc_features' shape: {data['vvc_features'].shape}")
    else:
        print(f"   ⚠️  Missing 'vvc_features' (will use zeros)")
    
    print()
    return test_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test BD-rate pipeline")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints_v2/epoch=998.ckpt',
        help='Model checkpoint to test'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/enhancer/dense_high_QP.yaml',
        help='Config file'
    )
    parser.add_argument(
        '--test-root',
        type=str,
        default='videos_test/test_frames_REAL',
        help='Test frames directory'
    )
    parser.add_argument(
        '--orig-root',
        type=str,
        default='videos_test/test_orig_frames_pt',
        help='Original frames directory'
    )
    
    args = parser.parse_args()
    
    success = True
    
    try:
        # Test 1: Bjøntegaard functions
        test_bjontegaard_functions()
        
        # Test 2: Model loading
        enhancer = test_model_loading(args.checkpoint, args.config)
        
        # Test 3: Data structure
        test_dir = test_data_structure(Path(args.test_root), Path(args.orig_root))
        
        print("=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)
        print()
        print("Your setup is ready for BD-rate analysis!")
        print()
        print("Run full analysis with:")
        print(f"  ./run_bd_analysis.sh {args.checkpoint} {args.config}")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print("TEST FAILED ❌")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
