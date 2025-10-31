#!/usr/bin/env python3
"""Full quantitative evaluation on test set."""

import torch
import numpy as np
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancer.trainer_module import TrainingModule
from enhancer.datamodule import VVCDataModule
from enhancer.config import Config
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def evaluate_model(model, datamodule, device='cuda'):
    """Evaluate model on full test set."""
    test_loader = datamodule.test_dataloader()
    
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    
    results = {
        'psnr_decoded': [],
        'psnr_enhanced': [],
        'ssim_decoded': [],
        'ssim_enhanced': [],
        'video_names': [],
    }
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            decoded, original, metadata_tensor, video_name, fused_maps = batch
            
            decoded = decoded.to(device)
            original = original.to(device)
            
            # Enhance
            enhanced = model.enhancer(decoded)
            
            # Calculate metrics (per sample in batch)
            for i in range(decoded.size(0)):
                # PSNR
                psnr_dec = psnr_metric(decoded[i:i+1], original[i:i+1])
                psnr_enh = psnr_metric(enhanced[i:i+1], original[i:i+1])
                
                # SSIM
                ssim_dec = ssim_metric(decoded[i:i+1], original[i:i+1])
                ssim_enh = ssim_metric(enhanced[i:i+1], original[i:i+1])
                
                results['psnr_decoded'].append(psnr_dec.item())
                results['psnr_enhanced'].append(psnr_enh.item())
                results['ssim_decoded'].append(ssim_dec.item())
                results['ssim_enhanced'].append(ssim_enh.item())
                results['video_names'].append(video_name[i])
    
    return pd.DataFrame(results)


def compare_models(baseline_df, vvc_df, output_path='experiments/vvc_comparison/test_evaluation.txt'):
    """Compare test set results."""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        def write(s):
            print(s)
            f.write(s + '\n')
        
        write("="*80)
        write("📊 TEST SET EVALUATION COMPARISON")
        write("="*80)
        
        write("\n" + "─"*80)
        write("BASELINE (No VVC Features)")
        write("─"*80)
        write(f"  PSNR (decoded):  {baseline_df['psnr_decoded'].mean():.4f} ± {baseline_df['psnr_decoded'].std():.4f} dB")
        write(f"  PSNR (enhanced): {baseline_df['psnr_enhanced'].mean():.4f} ± {baseline_df['psnr_enhanced'].std():.4f} dB")
        write(f"  PSNR Gain:       {(baseline_df['psnr_enhanced'] - baseline_df['psnr_decoded']).mean():.4f} dB")
        write(f"\n  SSIM (decoded):  {baseline_df['ssim_decoded'].mean():.4f} ± {baseline_df['ssim_decoded'].std():.4f}")
        write(f"  SSIM (enhanced): {baseline_df['ssim_enhanced'].mean():.4f} ± {baseline_df['ssim_enhanced'].std():.4f}")
        write(f"  SSIM Gain:       {(baseline_df['ssim_enhanced'] - baseline_df['ssim_decoded']).mean():.4f}")
        
        write("\n" + "─"*80)
        write("VVC-ENHANCED (With VVC Features)")
        write("─"*80)
        write(f"  PSNR (decoded):  {vvc_df['psnr_decoded'].mean():.4f} ± {vvc_df['psnr_decoded'].std():.4f} dB")
        write(f"  PSNR (enhanced): {vvc_df['psnr_enhanced'].mean():.4f} ± {vvc_df['psnr_enhanced'].std():.4f} dB")
        write(f"  PSNR Gain:       {(vvc_df['psnr_enhanced'] - vvc_df['psnr_decoded']).mean():.4f} dB")
        write(f"\n  SSIM (decoded):  {vvc_df['ssim_decoded'].mean():.4f} ± {vvc_df['ssim_decoded'].std():.4f}")
        write(f"  SSIM (enhanced): {vvc_df['ssim_enhanced'].mean():.4f} ± {vvc_df['ssim_enhanced'].std():.4f}")
        write(f"  SSIM Gain:       {(vvc_df['ssim_enhanced'] - vvc_df['ssim_decoded']).mean():.4f}")
        
        write("\n" + "─"*80)
        write("🎯 VVC vs BASELINE IMPROVEMENT")
        write("─"*80)
        
        psnr_improvement = vvc_df['psnr_enhanced'].mean() - baseline_df['psnr_enhanced'].mean()
        ssim_improvement = vvc_df['ssim_enhanced'].mean() - baseline_df['ssim_enhanced'].mean()
        
        write(f"  PSNR Improvement: {psnr_improvement:+.4f} dB ({psnr_improvement/baseline_df['psnr_enhanced'].mean()*100:+.2f}%)")
        write(f"  SSIM Improvement: {ssim_improvement:+.6f} ({ssim_improvement/baseline_df['ssim_enhanced'].mean()*100:+.2f}%)")
        
        if psnr_improvement > 0:
            write(f"\n  ✅ VVC-enhanced model is BETTER by {psnr_improvement:.4f} dB PSNR")
        else:
            write(f"\n  ⚠️  Baseline model is better by {abs(psnr_improvement):.4f} dB PSNR")
        
        write("\n" + "="*80)
    
    print(f"\n💾 Saved detailed report: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--vvc-ckpt', required=True)
    parser.add_argument('--baseline-config', default='experiments/vvc_comparison/baseline_dense.yaml')
    parser.add_argument('--vvc-config', default='experiments/vvc_comparison/vvc_enhanced_dense.yaml')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    # Evaluate baseline
    print("🔄 Evaluating BASELINE model...")
    with open(args.baseline_config) as f:
        baseline_config = yaml.safe_load(f)
    
    baseline_cfg = Config(baseline_config)
    baseline_dm = VVCDataModule(
        baseline_cfg.dataloader,
        baseline_cfg.dataset,
        fused_maps_dir=None,
    )
    baseline_dm.setup('test')
    
    baseline_model = TrainingModule.load_from_checkpoint(
        args.baseline_ckpt,
        enhancer_config=baseline_config['enhancer'],
        discriminator_config=baseline_config.get('discriminator', {}),
        trainer_config=baseline_config['trainer'],
    )
    
    baseline_results = evaluate_model(baseline_model, baseline_dm, args.device)
    baseline_results.to_csv('experiments/vvc_comparison/baseline_test_results.csv', index=False)
    
    # Evaluate VVC
    print("\n🔄 Evaluating VVC-ENHANCED model...")
    with open(args.vvc_config) as f:
        vvc_config = yaml.safe_load(f)
    
    vvc_cfg = Config(vvc_config)
    vvc_dm = VVCDataModule(
        vvc_cfg.dataloader,
        vvc_cfg.dataset,
        fused_maps_dir=vvc_config.get('fused_maps_dir'),
    )
    vvc_dm.setup('test')
    
    vvc_model = TrainingModule.load_from_checkpoint(
        args.vvc_ckpt,
        enhancer_config=vvc_config['enhancer'],
        discriminator_config=vvc_config.get('discriminator', {}),
        trainer_config=vvc_config['trainer'],
    )
    
    vvc_results = evaluate_model(vvc_model, vvc_dm, args.device)
    vvc_results.to_csv('experiments/vvc_comparison/vvc_test_results.csv', index=False)
    
    # Compare
    print("\n📊 Comparing results...")
    compare_models(baseline_results, vvc_results)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
