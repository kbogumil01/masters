#!/usr/bin/env python3
"""
Simple script to compare training results from Lightning logs
Usage: python compare_results.py version_42 version_43
"""
import sys
import pandas as pd
from pathlib import Path

def compare_versions(v1, v2):
    base_path = Path("lightning_logs")
    
    # Read metrics
    df1 = pd.read_csv(base_path / v1 / "metrics.csv")
    df2 = pd.read_csv(base_path / v2 / "metrics.csv")
    
    # Get final validation metrics (last epoch)
    metrics = ['val_psnr', 'val_ssim', 'val_g_loss']
    
    print(f"\n{'='*60}")
    print(f"Comparing: {v1} vs {v2}")
    print(f"{'='*60}\n")
    
    for metric in metrics:
        if metric in df1.columns and metric in df2.columns:
            v1_val = df1[metric].dropna().iloc[-1]
            v2_val = df2[metric].dropna().iloc[-1]
            diff = v2_val - v1_val
            better = "✓" if (diff > 0 and 'loss' not in metric) or (diff < 0 and 'loss' in metric) else "✗"
            
            print(f"{metric:20} | {v1}: {v1_val:.5f} | {v2}: {v2_val:.5f} | Δ: {diff:+.5f} {better}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py version_XX version_YY")
        sys.exit(1)
    
    compare_versions(sys.argv[1], sys.argv[2])
