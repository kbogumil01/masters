#!/usr/bin/env python3
"""Compare training results between baseline and VVC-enhanced models."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(csv_path):
    """Load metrics from Lightning CSV log."""
    df = pd.read_csv(csv_path)
    # Drop rows with all NaN values
    df = df.dropna(how='all')
    return df


def plot_comparison(baseline_df, vvc_df, output_dir='experiments/vvc_comparison/plots'):
    """Create comparison plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract training and validation losses
    metrics_to_plot = [
        ('g_loss', 'train_g_loss', 'Generator Loss (Train)'),
        ('val_g_loss', 'val_g_loss', 'Generator Loss (Validation)'),
        ('val_psnr', 'val_psnr', 'PSNR (Validation)'),
        ('val_ssim', 'val_ssim', 'SSIM (Validation)'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Baseline vs VVC-Enhanced Training Comparison', fontsize=16)
    
    for idx, (col_name, csv_col, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        # Plot baseline
        if csv_col in baseline_df.columns:
            baseline_data = baseline_df[csv_col].dropna()
            baseline_epochs = baseline_df.loc[baseline_data.index, 'epoch'].values
            ax.plot(baseline_epochs, baseline_data.values, 
                   label='Baseline (No VVC)', marker='o', alpha=0.7)
        
        # Plot VVC enhanced
        if csv_col in vvc_df.columns:
            vvc_data = vvc_df[csv_col].dropna()
            vvc_epochs = vvc_df.loc[vvc_data.index, 'epoch'].values
            ax.plot(vvc_epochs, vvc_data.values, 
                   label='VVC Enhanced', marker='s', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_comparison.png', dpi=150)
    print(f"📊 Saved plot: {output_dir}/training_comparison.png")
    
    return fig


def print_summary(baseline_df, vvc_df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("📈 TRAINING COMPARISON SUMMARY")
    print("="*60)
    
    # Get final epoch values
    metrics = ['g_loss', 'val_g_loss', 'val_psnr', 'val_ssim']
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'VVC Enhanced':<15} {'Improvement':<15}")
    print("-"*70)
    
    for metric in metrics:
        baseline_val = baseline_df[metric].dropna().iloc[-1] if metric in baseline_df.columns else None
        vvc_val = vvc_df[metric].dropna().iloc[-1] if metric in vvc_df.columns else None
        
        if baseline_val is not None and vvc_val is not None:
            # For losses (lower is better), for PSNR/SSIM (higher is better)
            if 'loss' in metric:
                improvement = ((baseline_val - vvc_val) / baseline_val) * 100
                sign = '↓' if improvement > 0 else '↑'
            else:
                improvement = ((vvc_val - baseline_val) / baseline_val) * 100
                sign = '↑' if improvement > 0 else '↓'
            
            print(f"{metric:<25} {baseline_val:<15.4f} {vvc_val:<15.4f} "
                  f"{sign} {abs(improvement):<.2f}%")
    
    print("\n" + "="*60)
    
    # Training time estimation
    baseline_epochs = baseline_df['epoch'].max() if 'epoch' in baseline_df.columns else 0
    vvc_epochs = vvc_df['epoch'].max() if 'epoch' in vvc_df.columns else 0
    
    print(f"\n📊 Epochs Completed:")
    print(f"  Baseline:      {baseline_epochs}")
    print(f"  VVC Enhanced:  {vvc_epochs}")
    print("="*60 + "\n")


def main():
    """Main comparison function."""
    import argparse
    parser = argparse.ArgumentParser(description='Compare training runs')
    parser.add_argument('--baseline-version', default='52', help='Baseline version number')
    parser.add_argument('--vvc-version', default='53', help='VVC-enhanced version number')
    args = parser.parse_args()
    
    # Find version directories
    logs_dir = Path('lightning_logs')
    
    baseline_csv = logs_dir / f'version_{args.baseline_version}' / 'metrics.csv'
    vvc_csv = logs_dir / f'version_{args.vvc_version}' / 'metrics.csv'
    
    print(f"📂 Comparing:")
    print(f"  Baseline:     {baseline_csv}")
    print(f"  VVC Enhanced: {vvc_csv}")
    
    if not baseline_csv.exists() or not vvc_csv.exists():
        print("❌ Metrics CSV files not found!")
        return
    
    # Load data
    baseline_df = load_metrics(baseline_csv)
    vvc_df = load_metrics(vvc_csv)
    
    # Print summary
    print_summary(baseline_df, vvc_df)
    
    # Create plots
    plot_comparison(baseline_df, vvc_df)
    
    # Save detailed comparison to CSV
    output_path = 'experiments/vvc_comparison/detailed_comparison.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Merge on epoch for detailed comparison
    comparison = pd.merge(
        baseline_df[['epoch', 'g_loss', 'val_g_loss', 'val_psnr', 'val_ssim']],
        vvc_df[['epoch', 'g_loss', 'val_g_loss', 'val_psnr', 'val_ssim']],
        on='epoch', 
        how='outer', 
        suffixes=('_baseline', '_vvc')
    )
    comparison.to_csv(output_path, index=False)
    print(f"💾 Saved detailed comparison: {output_path}")


if __name__ == '__main__':
    main()
