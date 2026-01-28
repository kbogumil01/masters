#!/usr/bin/env python3
"""
Step 5: Parse ffmpeg metric logs and generate BD-rate report.

Usage:
    python -m bin.my_metrics.generate_report \\
        --metrics-dir videos_test/metrics \\
        --encoder-logs videos_test/encoded_test \\
        --output results/bd_report.csv

Output:
    - CSV with per-sequence BD-BR and BD-PSNR
    - Summary statistics
"""

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import parse_sequence_name, get_bitrate_from_encoder_log


def parse_ffmpeg_psnr_log(log_path: Path) -> dict:
    """
    Parse ffmpeg PSNR output.
    
    Format line:
    [Parsed_psnr_0 @ ...] PSNR y:XX.XX u:XX.XX v:XX.XX average:XX.XX min:XX.XX max:XX.XX
    """
    if not log_path.exists():
        return None
    
    with open(log_path, "r") as f:
        content = f.read()
    
    # Find the summary line
    match = re.search(
        r'PSNR\s+y:(\d+\.?\d*)\s+u:(\d+\.?\d*)\s+v:(\d+\.?\d*)\s+average:(\d+\.?\d*)',
        content
    )
    
    if match:
        return {
            'psnr_Y': float(match.group(1)),
            'psnr_U': float(match.group(2)),
            'psnr_V': float(match.group(3)),
            'psnr_avg': float(match.group(4)),
        }
    return None


def parse_ffmpeg_ssim_log(log_path: Path) -> dict:
    """
    Parse ffmpeg SSIM output.
    
    Format line:
    [Parsed_ssim_0 @ ...] SSIM Y:X.XXXXX (XX.XX) U:X.XXXXX (XX.XX) V:X.XXXXX (XX.XX) All:X.XXXXX (XX.XX)
    """
    if not log_path.exists():
        return None
    
    with open(log_path, "r") as f:
        content = f.read()
    
    # Find the summary line
    match = re.search(
        r'SSIM\s+Y:(\d+\.?\d*)\s*\([^)]+\)\s*U:(\d+\.?\d*)\s*\([^)]+\)\s*V:(\d+\.?\d*)\s*\([^)]+\)\s*All:(\d+\.?\d*)',
        content
    )
    
    if match:
        return {
            'ssim_Y': float(match.group(1)),
            'ssim_U': float(match.group(2)),
            'ssim_V': float(match.group(3)),
            'ssim_avg': float(match.group(4)),
        }
    return None


def bjontegaard_delta(R1, PSNR1, R2, PSNR2, mode='rate'):
    """
    Calculate Bjøntegaard Delta metric.
    
    mode='rate': BD-BR (% bitrate change at same quality)
    mode='psnr': BD-PSNR (dB quality change at same bitrate)
    """
    # Need at least 4 points
    if len(R1) < 4 or len(R2) < 4:
        return np.nan
    
    lR1 = np.log(R1)
    lR2 = np.log(R2)
    
    # Fit 3rd order polynomial
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)
    
    # Integration bounds
    min_psnr = max(min(PSNR1), min(PSNR2))
    max_psnr = min(max(PSNR1), max(PSNR2))
    
    if min_psnr >= max_psnr:
        return np.nan
    
    # Integrate
    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)
    
    int1 = np.polyval(p_int1, max_psnr) - np.polyval(p_int1, min_psnr)
    int2 = np.polyval(p_int2, max_psnr) - np.polyval(p_int2, min_psnr)
    
    avg_diff = (int2 - int1) / (max_psnr - min_psnr)
    
    if mode == 'rate':
        # BD-BR as percentage
        return (np.exp(avg_diff) - 1) * 100
    else:
        # BD-PSNR in dB
        # Fit inverse: PSNR as function of log(R)
        p1_inv = np.polyfit(lR1, PSNR1, 3)
        p2_inv = np.polyfit(lR2, PSNR2, 3)
        
        min_lr = max(min(lR1), min(lR2))
        max_lr = min(max(lR1), max(lR2))
        
        if min_lr >= max_lr:
            return np.nan
        
        p_int1_inv = np.polyint(p1_inv)
        p_int2_inv = np.polyint(p2_inv)
        
        int1_inv = np.polyval(p_int1_inv, max_lr) - np.polyval(p_int1_inv, min_lr)
        int2_inv = np.polyval(p_int2_inv, max_lr) - np.polyval(p_int2_inv, min_lr)
        
        return (int2_inv - int1_inv) / (max_lr - min_lr)


def collect_rd_points(metrics_dir: Path, encoder_logs_dir: Path):
    """Collect RD points from metrics and encoder logs."""
    
    # Group by sequence name
    sequences = defaultdict(lambda: {'decoded': [], 'enhanced': []})
    
    # Find all metric files
    for psnr_file in metrics_dir.glob("*_psnr.info"):
        name = psnr_file.stem
        
        # Determine if enhanced or decoded
        if "_enhanced_psnr" in name:
            seq_name = name.replace("_enhanced_psnr", "")
            source_type = "enhanced"
        elif "_decoded_psnr" in name:
            seq_name = name.replace("_decoded_psnr", "")
            source_type = "decoded"
        else:
            continue
        
        # Parse sequence info
        info = parse_sequence_name(seq_name)
        if not info:
            continue
        
        # Get metrics
        psnr = parse_ffmpeg_psnr_log(psnr_file)
        ssim_file = psnr_file.parent / name.replace("_psnr", "_ssim") + ".info"
        ssim_file = psnr_file.parent / (name.replace("_psnr", "_ssim") + ".info")
        ssim = parse_ffmpeg_ssim_log(ssim_file)
        
        if not psnr:
            continue
        
        # Get bitrate from encoder log
        log_path = encoder_logs_dir / f"{seq_name}.log"
        bitrate = get_bitrate_from_encoder_log(str(log_path))
        
        if bitrate is None:
            # Try alternative pattern
            log_path = encoder_logs_dir / f"{info['name']}" / f"{seq_name}.log"
            bitrate = get_bitrate_from_encoder_log(str(log_path))
        
        if bitrate is None:
            print(f"Warning: No bitrate found for {seq_name}")
            continue
        
        point = {
            'seq_name': info['name'],
            'qp': info['qp'],
            'alf': info['alf'],
            'db': info['db'],
            'sao': info['sao'],
            'bitrate': bitrate,
            **psnr,
        }
        
        if ssim:
            point.update(ssim)
        
        sequences[info['name']][source_type].append(point)
    
    return sequences


def calculate_bd_metrics(sequences: dict) -> pd.DataFrame:
    """Calculate BD-BR and BD-PSNR for each sequence."""
    
    results = []
    
    for seq_name, data in sequences.items():
        decoded_points = data['decoded']
        enhanced_points = data['enhanced']
        
        if len(decoded_points) < 4 or len(enhanced_points) < 4:
            print(f"Warning: Not enough points for {seq_name}")
            continue
        
        # Sort by QP
        decoded_points = sorted(decoded_points, key=lambda x: x['qp'])
        enhanced_points = sorted(enhanced_points, key=lambda x: x['qp'])
        
        # Extract arrays
        R_dec = np.array([p['bitrate'] for p in decoded_points])
        R_enh = np.array([p['bitrate'] for p in enhanced_points])
        
        # Calculate BD for each channel
        for channel in ['Y', 'U', 'V', 'avg']:
            psnr_key = f'psnr_{channel}'
            
            PSNR_dec = np.array([p[psnr_key] for p in decoded_points])
            PSNR_enh = np.array([p[psnr_key] for p in enhanced_points])
            
            # BD-BR (PSNR) - negative means bitrate reduction (good)
            bd_br_psnr = bjontegaard_delta(R_dec, PSNR_dec, R_enh, PSNR_enh, mode='rate')
            
            # BD-PSNR - positive means quality improvement (good)
            bd_psnr = bjontegaard_delta(R_dec, PSNR_dec, R_enh, PSNR_enh, mode='psnr')
            
            results.append({
                'sequence': seq_name,
                'channel': channel,
                'BD-BR (PSNR) %': bd_br_psnr,
                'BD-PSNR dB': bd_psnr,
            })
        
        # SSIM (overall only, as per your specification)
        if 'ssim_avg' in decoded_points[0]:
            SSIM_dec = np.array([p['ssim_avg'] for p in decoded_points])
            SSIM_enh = np.array([p['ssim_avg'] for p in enhanced_points])
            
            # Convert SSIM to dB-like scale for BD calculation
            SSIM_dec_db = -10 * np.log10(1 - SSIM_dec + 1e-10)
            SSIM_enh_db = -10 * np.log10(1 - SSIM_enh + 1e-10)
            
            bd_br_ssim = bjontegaard_delta(R_dec, SSIM_dec_db, R_enh, SSIM_enh_db, mode='rate')
            bd_ssim = bjontegaard_delta(R_dec, SSIM_dec_db, R_enh, SSIM_enh_db, mode='psnr')
            
            results.append({
                'sequence': seq_name,
                'channel': 'SSIM',
                'BD-BR (SSIM) %': bd_br_ssim,
                'BD-SSIM dB': bd_ssim,
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Generate BD-rate report")
    parser.add_argument("--metrics-dir", required=True, help="Directory with metric logs")
    parser.add_argument("--encoder-logs", required=True, help="Directory with encoder logs")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()
    
    metrics_dir = Path(args.metrics_dir)
    encoder_logs_dir = Path(args.encoder_logs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Collecting RD points...")
    sequences = collect_rd_points(metrics_dir, encoder_logs_dir)
    
    print(f"Found {len(sequences)} sequences")
    for name, data in sequences.items():
        print(f"  {name}: {len(data['decoded'])} decoded, {len(data['enhanced'])} enhanced points")
    
    print("\nCalculating BD metrics...")
    df = calculate_bd_metrics(sequences)
    
    # Save detailed results
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BD-RATE SUMMARY (per channel)")
    print("="*60)
    
    # Group by channel and calculate mean
    summary = df.groupby('channel').agg({
        'BD-BR (PSNR) %': 'mean',
        'BD-PSNR dB': 'mean',
    }).round(2)
    
    print(summary.to_string())
    
    # Overall average
    print("\n" + "-"*60)
    psnr_channels = df[df['channel'].isin(['Y', 'U', 'V'])]
    print(f"Average BD-BR (PSNR): {psnr_channels['BD-BR (PSNR) %'].mean():.2f}%")
    print(f"Average BD-PSNR:      {psnr_channels['BD-PSNR dB'].mean():.2f} dB")
    
    if 'BD-BR (SSIM) %' in df.columns:
        ssim_rows = df[df['channel'] == 'SSIM']
        if not ssim_rows.empty:
            print(f"Average BD-BR (SSIM): {ssim_rows['BD-BR (SSIM) %'].mean():.2f}%")


if __name__ == "__main__":
    main()
