#!/usr/bin/env python3
"""Visually compare enhanced outputs from baseline vs VVC models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import yaml

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancer.datamodule import VVCDataModule
from enhancer.trainer_module import TrainingModule


def load_model_from_checkpoint(checkpoint_path, config_path):
    """Load trained model from checkpoint."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    module = TrainingModule.load_from_checkpoint(
        checkpoint_path,
        enhancer_config=config['enhancer'],
        discriminator_config=config.get('discriminator', {}),
        trainer_config=config['trainer'],
    )
    module.eval()
    return module


def enhance_test_images(model, datamodule, num_samples=5):
    """Generate enhanced images from test set."""
    test_loader = datamodule.test_dataloader()
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_samples:
                break
            
            decoded, original, metadata_tensor, video_name, fused_maps = batch
            
            # Move to GPU if available
            device = next(model.parameters()).device
            decoded = decoded.to(device)
            
            # Enhance
            enhanced = model.enhancer(decoded)
            
            # Move back to CPU for visualization
            results.append({
                'decoded': decoded.cpu(),
                'enhanced': enhanced.cpu(),
                'original': original.cpu(),
                'video_name': video_name,
            })
    
    return results


def visualize_comparison(baseline_results, vvc_results, output_dir='experiments/vvc_comparison/visual'):
    """Create side-by-side comparison visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx in range(min(len(baseline_results), len(vvc_results))):
        baseline = baseline_results[idx]
        vvc = vvc_results[idx]
        
        # Take first image from batch
        decoded = baseline['decoded'][0].permute(1, 2, 0).numpy()
        baseline_enh = baseline['enhanced'][0].permute(1, 2, 0).numpy()
        vvc_enh = vvc['enhanced'][0].permute(1, 2, 0).numpy()
        original = baseline['original'][0].permute(1, 2, 0).numpy()
        
        # Clip to valid range
        decoded = np.clip(decoded, 0, 1)
        baseline_enh = np.clip(baseline_enh, 0, 1)
        vvc_enh = np.clip(vvc_enh, 0, 1)
        original = np.clip(original, 0, 1)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original)
        axes[0].set_title('Original (Ground Truth)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(decoded)
        axes[1].set_title('Decoded (VVC Compressed)', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(baseline_enh)
        axes[2].set_title('Baseline Enhanced', fontsize=12)
        axes[2].axis('off')
        
        axes[3].imshow(vvc_enh)
        axes[3].set_title('VVC-Enhanced', fontsize=12, fontweight='bold', color='green')
        axes[3].axis('off')
        
        plt.suptitle(f'Sample {idx+1}: {baseline["video_name"][0]}', fontsize=14)
        plt.tight_layout()
        
        output_path = f'{output_dir}/comparison_sample_{idx+1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {output_path}")


def main():
    """Compare visual quality of both models."""
    import argparse
    parser = argparse.ArgumentParser(description='Compare enhanced image quality')
    parser.add_argument('--baseline-ckpt', required=True, help='Baseline checkpoint path')
    parser.add_argument('--vvc-ckpt', required=True, help='VVC-enhanced checkpoint path')
    parser.add_argument('--baseline-config', default='experiments/vvc_comparison/baseline_dense.yaml')
    parser.add_argument('--vvc-config', default='experiments/vvc_comparison/vvc_enhanced_dense.yaml')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of test samples')
    
    args = parser.parse_args()
    
    print("🔄 Loading models...")
    
    # Load baseline model
    print(f"  Baseline: {args.baseline_ckpt}")
    baseline_model = load_model_from_checkpoint(args.baseline_ckpt, args.baseline_config)
    
    # Load VVC model
    print(f"  VVC Enhanced: {args.vvc_ckpt}")
    vvc_model = load_model_from_checkpoint(args.vvc_ckpt, args.vvc_config)
    
    # Load test data (using baseline config, test set is same)
    print("📂 Loading test data...")
    with open(args.baseline_config) as f:
        config = yaml.safe_load(f)
    
    from enhancer.config import Config
    cfg = Config(config)
    datamodule = VVCDataModule(
        dataloader_config=cfg.dataloader,
        dataset_config=cfg.dataset,
        fused_maps_dir=None,  # Baseline doesn't need VVC features for test
    )
    datamodule.setup('test')
    
    # Generate enhanced images
    print(f"🎨 Enhancing {args.num_samples} test samples with baseline model...")
    baseline_results = enhance_test_images(baseline_model, datamodule, args.num_samples)
    
    # For VVC model, need to reload datamodule with VVC features
    with open(args.vvc_config) as f:
        vvc_config = yaml.safe_load(f)
    vvc_cfg = Config(vvc_config)
    vvc_datamodule = VVCDataModule(
        dataloader_config=vvc_cfg.dataloader,
        dataset_config=vvc_cfg.dataset,
        fused_maps_dir=vvc_config.get('fused_maps_dir'),
    )
    vvc_datamodule.setup('test')
    
    print(f"🎨 Enhancing {args.num_samples} test samples with VVC model...")
    vvc_results = enhance_test_images(vvc_model, vvc_datamodule, args.num_samples)
    
    # Create visualizations
    print("📊 Creating comparison visualizations...")
    visualize_comparison(baseline_results, vvc_results)
    
    print("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()
