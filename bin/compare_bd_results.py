#!/usr/bin/env python3
"""
Compare BD-rate results across multiple model checkpoints.
Useful for comparing different training epochs or model architectures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_bd_results(csv_path: Path) -> pd.DataFrame:
    """Load BD-rate results from CSV."""
    return pd.read_csv(csv_path)


def compare_models(results_dict: dict, output_dir: Path):
    """
    Compare BD-rate results from multiple models.
    
    Args:
        results_dict: {model_name: DataFrame}
        output_dir: Where to save comparison plots and reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate statistics
    comparison = []
    for model_name, df in results_dict.items():
        comparison.append({
            'model': model_name,
            'avg_bd_br_psnr': df['bd_br_psnr'].mean(),
            'avg_bd_psnr': df['bd_psnr'].mean(),
            'avg_bd_br_ssim': df['bd_br_ssim'].mean(),
            'avg_bd_ssim': df['bd_ssim'].mean(),
            'median_bd_br_psnr': df['bd_br_psnr'].median(),
            'median_bd_psnr': df['bd_psnr'].median(),
            'num_sequences': len(df)
        })
    
    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values('avg_bd_psnr', ascending=False)
    
    # Print comparison table
    print("\n" + "=" * 120)
    print("MODEL COMPARISON - BD-RATE METRICS")
    print("=" * 120)
    print()
    print(f"{'Model':<40} {'BD-BR(PSNR)':>12} {'BD-PSNR':>10} {'BD-BR(SSIM)':>12} {'BD-SSIM':>10} {'Seqs':>6}")
    print(f"{'':40} {'[%]':>12} {'[dB]':>10} {'[%]':>12} {'':>10} {'':>6}")
    print("─" * 120)
    
    for _, row in comp_df.iterrows():
        print(f"{row['model']:<40} {row['avg_bd_br_psnr']:>11.2f}% {row['avg_bd_psnr']:>9.4f} dB "
              f"{row['avg_bd_br_ssim']:>11.2f}% {row['avg_bd_ssim']:>9.6f} {row['num_sequences']:>6.0f}")
    
    print("=" * 120)
    
    # Save to CSV
    comp_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\n💾 Comparison saved to: {output_dir / 'model_comparison.csv'}")
    
    # Create visualization
    create_comparison_plots(results_dict, comp_df, output_dir)


def create_comparison_plots(results_dict: dict, summary_df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # 1. Bar plot of average BD metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BD-Rate Metrics Comparison Across Models', fontsize=16, fontweight='bold')
    
    models = summary_df['model'].tolist()
    
    # BD-BR (PSNR)
    axes[0, 0].barh(models, summary_df['avg_bd_br_psnr'], color='steelblue')
    axes[0, 0].set_xlabel('BD-BR(PSNR) [%]', fontsize=12)
    axes[0, 0].set_title('Bitrate Savings (PSNR-based)\nNegative = Better', fontsize=12)
    axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # BD-PSNR
    axes[0, 1].barh(models, summary_df['avg_bd_psnr'], color='forestgreen')
    axes[0, 1].set_xlabel('BD-PSNR [dB]', fontsize=12)
    axes[0, 1].set_title('Quality Improvement (PSNR)\nPositive = Better', fontsize=12)
    axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # BD-BR (SSIM)
    axes[1, 0].barh(models, summary_df['avg_bd_br_ssim'], color='coral')
    axes[1, 0].set_xlabel('BD-BR(SSIM) [%]', fontsize=12)
    axes[1, 0].set_title('Bitrate Savings (SSIM-based)\nNegative = Better', fontsize=12)
    axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # BD-SSIM
    axes[1, 1].barh(models, summary_df['avg_bd_ssim'], color='mediumorchid')
    axes[1, 1].set_xlabel('BD-SSIM', fontsize=12)
    axes[1, 1].set_title('Quality Improvement (SSIM)\nPositive = Better', fontsize=12)
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bd_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved to: {output_dir / 'bd_metrics_comparison.png'}")
    plt.close()
    
    # 2. Per-sequence comparison (if only 2 models)
    if len(results_dict) == 2:
        create_per_sequence_comparison(results_dict, output_dir)


def create_per_sequence_comparison(results_dict: dict, output_dir: Path):
    """Create detailed per-sequence comparison for 2 models."""
    
    model_names = list(results_dict.keys())
    df1 = results_dict[model_names[0]].set_index('sequence')
    df2 = results_dict[model_names[1]].set_index('sequence')
    
    # Find common sequences
    common_seqs = df1.index.intersection(df2.index)
    
    if len(common_seqs) == 0:
        print("⚠️  No common sequences found for detailed comparison")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Per-Sequence Comparison: {model_names[0]} vs {model_names[1]}', 
                 fontsize=14, fontweight='bold')
    
    x = np.arange(len(common_seqs))
    width = 0.35
    
    # BD-PSNR comparison
    axes[0].bar(x - width/2, [df1.loc[s, 'bd_psnr'] for s in common_seqs], 
                width, label=model_names[0], color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, [df2.loc[s, 'bd_psnr'] for s in common_seqs], 
                width, label=model_names[1], color='forestgreen', alpha=0.8)
    axes[0].set_ylabel('BD-PSNR [dB]', fontsize=11)
    axes[0].set_title('Quality Improvement per Sequence', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(common_seqs, rotation=45, ha='right', fontsize=9)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    # BD-BR comparison
    axes[1].bar(x - width/2, [df1.loc[s, 'bd_br_psnr'] for s in common_seqs], 
                width, label=model_names[0], color='steelblue', alpha=0.8)
    axes[1].bar(x + width/2, [df2.loc[s, 'bd_br_psnr'] for s in common_seqs], 
                width, label=model_names[1], color='forestgreen', alpha=0.8)
    axes[1].set_ylabel('BD-BR(PSNR) [%]', fontsize=11)
    axes[1].set_title('Bitrate Savings per Sequence', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(common_seqs, rotation=45, ha='right', fontsize=9)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_sequence_comparison.png', dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved to: {output_dir / 'per_sequence_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare BD-rate results across multiple models"
    )
    parser.add_argument(
        'results',
        nargs='+',
        help='CSV files with BD-rate results (format: model_name:path/to/results.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/bd_rate_comparison',
        help='Output directory for comparison results'
    )
    
    args = parser.parse_args()
    
    # Load all results
    results_dict = {}
    for result_spec in args.results:
        if ':' in result_spec:
            name, path = result_spec.split(':', 1)
        else:
            # Use filename as name
            path = result_spec
            name = Path(path).stem
        
        print(f"Loading {name} from {path}...")
        results_dict[name] = load_bd_results(Path(path))
    
    print(f"\nLoaded {len(results_dict)} model results")
    
    # Compare
    compare_models(results_dict, Path(args.output))


if __name__ == '__main__':
    main()
