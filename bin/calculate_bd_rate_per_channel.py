#!/usr/bin/env python3
"""
Calculate per-channel BD-BR metrics (Y, U, V separately) to compare with reference papers.
This matches the format used in academic papers where BD-rate is reported for each YUV channel.
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import re
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bin.calculate_bd_rate import (
    bjontegaard_delta,
    bjontegaard_psnr,
    load_enhancer_model,
    extract_qp_from_path,
    get_bitrate_from_log
)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def calculate_per_channel_metrics(
    enhancer,
    decoded_frames_dir: Path,
    original_frames_dir: Path,
    device: str = 'cuda'
):
    """
    Calculate PSNR and SSIM separately for Y, U, V channels.
    
    Returns dict with per-channel metrics matching academic paper format.
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Find frame files (skip fused_maps)
    decoded_files = sorted(decoded_frames_dir.glob("frame_*.pt"))
    if len(decoded_files) == 0:
        all_pt_files = sorted(decoded_frames_dir.glob("*.pt"))
        decoded_files = [f for f in all_pt_files if "fused_maps" not in f.name.lower()]
    
    if len(decoded_files) == 0:
        return None
    
    # Store per-channel results
    results = {
        'psnr_Y_dec': [], 'psnr_U_dec': [], 'psnr_V_dec': [],
        'psnr_Y_enh': [], 'psnr_U_enh': [], 'psnr_V_enh': [],
        'ssim_Y_dec': [], 'ssim_U_dec': [], 'ssim_V_dec': [],
        'ssim_Y_enh': [], 'ssim_U_enh': [], 'ssim_V_enh': [],
        'psnr_full_dec': [], 'psnr_full_enh': [],
        'ssim_full_dec': [], 'ssim_full_enh': []
    }
    
    with torch.no_grad():
        for dec_file in tqdm(decoded_files, desc="  Processing frames", leave=False):
            try:
                dec_data = torch.load(dec_file, map_location="cpu")
                if "chunk" not in dec_data:
                    continue
                decoded = dec_data["chunk"].float() / 255.0
            except:
                continue
            
            # Find original
            if "poc" in dec_file.name:
                poc_match = re.search(r'poc(\d+)', dec_file.name)
                if poc_match:
                    orig_file = original_frames_dir / f"frame_poc{poc_match.group(1)}.pt"
                else:
                    orig_file = original_frames_dir / dec_file.name
            else:
                orig_file = original_frames_dir / dec_file.name
            
            if not orig_file.exists():
                continue
            
            try:
                orig_data = torch.load(orig_file, map_location="cpu")
                if "chunk" not in orig_data:
                    continue
                original = orig_data["chunk"].float() / 255.0
            except:
                continue
            
            # Move to device
            decoded = decoded.unsqueeze(0).to(device)
            original = original.unsqueeze(0).to(device)
            
            # Get metadata - 6 channels matching Piotr's format: [profile, qp, alf, sao, db, is_intra]
            metadata = dec_data.get("seq_meta", {})
            profile = str(metadata.get("profile", "AI")).upper()
            profile_ai = 1.0 if "AI" in profile else 0.0
            qp = float(metadata.get("qp", 32))
            qp_n = qp / 64.0
            alf = float(metadata.get("alf", 0))
            sao = float(metadata.get("sao", 0))
            db = float(metadata.get("db", 0))
            is_intra = float(dec_data.get("is_intra", 1))
            meta_tensor = torch.tensor([profile_ai, qp_n, alf, sao, db, is_intra], dtype=torch.float32).view(1, 6, 1, 1).to(device)
            
            # Get VVC features if available
            vvc_feat = None
            if "vvc_features" in dec_data:
                vvc_feat = dec_data["vvc_features"].float().unsqueeze(0).to(device)
            
            # Enhance
            enhanced = enhancer(decoded, meta_tensor, vvc_feat)
            
            # Split into Y, U, V channels (assuming YUV ordering in channel dimension)
            dec_Y, dec_U, dec_V = decoded[:, 0:1, :, :], decoded[:, 1:2, :, :], decoded[:, 2:3, :, :]
            enh_Y, enh_U, enh_V = enhanced[:, 0:1, :, :], enhanced[:, 1:2, :, :], enhanced[:, 2:3, :, :]
            orig_Y, orig_U, orig_V = original[:, 0:1, :, :], original[:, 1:2, :, :], original[:, 2:3, :, :]
            
            # Calculate per-channel PSNR
            results['psnr_Y_dec'].append(psnr_metric(dec_Y, orig_Y).item())
            results['psnr_U_dec'].append(psnr_metric(dec_U, orig_U).item())
            results['psnr_V_dec'].append(psnr_metric(dec_V, orig_V).item())
            
            results['psnr_Y_enh'].append(psnr_metric(enh_Y, orig_Y).item())
            results['psnr_U_enh'].append(psnr_metric(enh_U, orig_U).item())
            results['psnr_V_enh'].append(psnr_metric(enh_V, orig_V).item())
            
            # Calculate per-channel SSIM
            results['ssim_Y_dec'].append(ssim_metric(dec_Y, orig_Y).item())
            results['ssim_U_dec'].append(ssim_metric(dec_U, orig_U).item())
            results['ssim_V_dec'].append(ssim_metric(dec_V, orig_V).item())
            
            results['ssim_Y_enh'].append(ssim_metric(enh_Y, orig_Y).item())
            results['ssim_U_enh'].append(ssim_metric(enh_U, orig_U).item())
            results['ssim_V_enh'].append(ssim_metric(enh_V, orig_V).item())
            
            # Full frame metrics (for reference)
            results['psnr_full_dec'].append(psnr_metric(decoded, original).item())
            results['psnr_full_enh'].append(psnr_metric(enhanced, original).item())
            results['ssim_full_dec'].append(ssim_metric(decoded, original).item())
            results['ssim_full_enh'].append(ssim_metric(enhanced, original).item())
    
    # Calculate averages
    return {k: np.mean(v) if v else np.nan for k, v in results.items()}


def collect_per_channel_rd_points(enhancer, test_root: Path, orig_root: Path, qp_values, device='cuda'):
    """Collect RD points with per-channel metrics."""
    
    results = {}
    all_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
    
    sequences = {}
    for d in all_dirs:
        match = re.match(r'(.+?)_(?:AI|RA)_QP\d+', d.name)
        if match:
            seq_name = match.group(1)
            if seq_name not in sequences:
                sequences[seq_name] = []
            sequences[seq_name].append(d)
    
    print(f"\nFound {len(sequences)} unique sequences")
    
    for seq_name, seq_dirs in sequences.items():
        print(f"\n{'='*80}")
        print(f"Processing sequence: {seq_name}")
        print(f"{'='*80}")
        
        results[seq_name] = {
            'qp': [], 'bitrate': [],
            'psnr_Y_dec': [], 'psnr_U_dec': [], 'psnr_V_dec': [],
            'psnr_Y_enh': [], 'psnr_U_enh': [], 'psnr_V_enh': [],
            'ssim_Y_dec': [], 'ssim_U_dec': [], 'ssim_V_dec': [],
            'ssim_Y_enh': [], 'ssim_U_enh': [], 'ssim_V_enh': [],
            'psnr_full_dec': [], 'psnr_full_enh': [],
            'ssim_full_dec': [], 'ssim_full_enh': []
        }
        
        orig_seq_dir = orig_root / seq_name
        if not orig_seq_dir.exists():
            print(f"  ⚠️  Original directory not found: {orig_seq_dir}")
            continue
        
        for seq_dir in sorted(seq_dirs):
            qp = extract_qp_from_path(seq_dir.name)
            if qp is None or (qp_values and qp not in qp_values):
                continue
            
            print(f"\n  QP={qp}")
            
            metrics = calculate_per_channel_metrics(enhancer, seq_dir, orig_seq_dir, device)
            if metrics is None:
                continue
            
            bitrate = get_bitrate_from_log(seq_dir)
            if bitrate is None:
                bitrate = 10000 * (0.85 ** (qp - 22))
            
            results[seq_name]['qp'].append(qp)
            results[seq_name]['bitrate'].append(bitrate)
            
            for key in metrics.keys():
                results[seq_name][key].append(metrics[key])
            
            print(f"    Bitrate: {bitrate:.2f} kbps")
            print(f"    PSNR-Y dec: {metrics['psnr_Y_dec']:.4f} dB, enh: {metrics['psnr_Y_enh']:.4f} dB (+{metrics['psnr_Y_enh']-metrics['psnr_Y_dec']:.4f})")
            print(f"    PSNR-U dec: {metrics['psnr_U_dec']:.4f} dB, enh: {metrics['psnr_U_enh']:.4f} dB (+{metrics['psnr_U_enh']-metrics['psnr_U_dec']:.4f})")
            print(f"    PSNR-V dec: {metrics['psnr_V_dec']:.4f} dB, enh: {metrics['psnr_V_enh']:.4f} dB (+{metrics['psnr_V_enh']-metrics['psnr_V_dec']:.4f})")
    
    return results


def calculate_per_channel_bd_metrics(rd_points):
    """Calculate BD metrics for each YUV channel separately."""
    
    bd_results = []
    
    for seq_name, data in rd_points.items():
        if len(data['qp']) < 4:
            print(f"⚠️  Skipping {seq_name}: need at least 4 QP points")
            continue
        
        bitrate = np.array(data['bitrate'])
        
        try:
            result = {'sequence': seq_name, 'num_points': len(data['qp'])}
            
            # BD-BR and BD-PSNR for each channel
            for channel in ['Y', 'U', 'V']:
                psnr_dec = np.array(data[f'psnr_{channel}_dec'])
                psnr_enh = np.array(data[f'psnr_{channel}_enh'])
                
                # BD-BR (negative = better)
                bd_br = bjontegaard_delta(bitrate, psnr_dec, bitrate, psnr_enh)
                result[f'bd_br_psnr_{channel}'] = bd_br
                
                # BD-PSNR (positive = better)
                bd_psnr = bjontegaard_psnr(bitrate, psnr_dec, bitrate, psnr_enh)
                result[f'bd_psnr_{channel}'] = bd_psnr
                
                # Average gain
                result[f'avg_psnr_gain_{channel}'] = np.mean(psnr_enh - psnr_dec)
            
            # BD-SSIM for each channel
            for channel in ['Y', 'U', 'V']:
                ssim_dec = np.array(data[f'ssim_{channel}_dec'])
                ssim_enh = np.array(data[f'ssim_{channel}_enh'])
                
                # Convert to dB-like for interpolation
                ssim_dec_db = -10 * np.log10(1 - ssim_dec + 1e-10)
                ssim_enh_db = -10 * np.log10(1 - ssim_enh + 1e-10)
                
                bd_br_ssim = bjontegaard_delta(bitrate, ssim_dec_db, bitrate, ssim_enh_db)
                result[f'bd_br_ssim_{channel}'] = bd_br_ssim
                
                bd_ssim = bjontegaard_psnr(bitrate, ssim_dec, bitrate, ssim_enh)
                result[f'bd_ssim_{channel}'] = bd_ssim
            
            # Full frame metrics for reference
            psnr_full_dec = np.array(data['psnr_full_dec'])
            psnr_full_enh = np.array(data['psnr_full_enh'])
            result['bd_br_psnr_full'] = bjontegaard_delta(bitrate, psnr_full_dec, bitrate, psnr_full_enh)
            result['bd_psnr_full'] = bjontegaard_psnr(bitrate, psnr_full_dec, bitrate, psnr_full_enh)
            
            bd_results.append(result)
            
        except Exception as e:
            print(f"⚠️  Error for {seq_name}: {e}")
            continue
    
    return pd.DataFrame(bd_results)


def print_comparison_table(bd_df: pd.DataFrame, output_file: Path = None):
    """Print results in format matching reference paper."""
    
    lines = []
    def write(s=""): 
        print(s)
        lines.append(s)
    
    write("="*100)
    write("📊 BD-RATE ANALYSIS - Per-Channel Results (Academic Paper Format)")
    write("="*100)
    write()
    
    # Average across all sequences
    avg = {
        'BD-BR(PSNR) Y': bd_df['bd_br_psnr_Y'].mean(),
        'BD-BR(PSNR) U': bd_df['bd_br_psnr_U'].mean(),
        'BD-BR(PSNR) V': bd_df['bd_br_psnr_V'].mean(),
        'BD-BR(SSIM) Y': bd_df['bd_br_ssim_Y'].mean(),
        'BD-PSNR Y': bd_df['bd_psnr_Y'].mean(),
        'BD-PSNR U': bd_df['bd_psnr_U'].mean(),
        'BD-PSNR V': bd_df['bd_psnr_V'].mean(),
        'BD-BR(SSIM)': bd_df['bd_br_ssim_Y'].mean(),  # Full frame based on Y (dominant)
        'BD-SSIM': bd_df['bd_ssim_Y'].mean(),
    }
    
    write("="*100)
    write("SUMMARY TABLE")
    write("="*100)
    write()
    write("Per-Channel Metrics (PSNR):")
    write("| Metric          |   Y Channel |   U Channel |   V Channel |")
    write("|-----------------|-------------|-------------|-------------|")
    write(f"| BD-BR (PSNR) %  | {avg['BD-BR(PSNR) Y']:>11.2f} | {avg['BD-BR(PSNR) U']:>11.2f} | {avg['BD-BR(PSNR) V']:>11.2f} |")
    write(f"| BD-PSNR (dB)    | {avg['BD-PSNR Y']:>11.4f} | {avg['BD-PSNR U']:>11.4f} | {avg['BD-PSNR V']:>11.4f} |")
    write()
    write("Overall Metrics (SSIM):")
    write("| Metric          |       Value |")
    write("|-----------------|-------------|")
    write(f"| BD-BR (SSIM) %  | {avg['BD-BR(SSIM)']:>11.2f} |")
    write(f"| BD-SSIM         | {avg['BD-SSIM']:>11.6f} |")
    write()
    
    write("="*100)
    write("REFERENCE PAPER COMPARISON")
    write("="*100)
    write()
    write("Your Results:")
    write(f"  BD-BR (PSNR) Y = {avg['BD-BR(PSNR) Y']:.2f}%")
    write(f"  BD-BR (PSNR) U = {avg['BD-BR(PSNR) U']:.2f}%")
    write(f"  BD-BR (PSNR) V = {avg['BD-BR(PSNR) V']:.2f}%")
    write()
    write("Reference Paper (DenseNet):")
    write("  BD-BR (PSNR) Y = -7.79%")
    write("  BD-BR (PSNR) U = -10.33%")
    write("  BD-BR (PSNR) V = -7.70%")
    write()
    
    if avg['BD-BR(PSNR) Y'] < -5:
        write("✅ Your model shows competitive performance!")
    else:
        write("📊 Your model shows moderate improvement. Consider:")
        write("   - Longer training")
        write("   - Different loss weights")
        write("   - Architecture modifications")
    
    write()
    write("="*100)
    
    # Per-sequence breakdown
    write()
    write("PER-SEQUENCE RESULTS")
    write("-"*100)
    write(f"{'Sequence':<25} | BD-BR Y | BD-BR U | BD-BR V | BD-PSNR Y | BD-PSNR U | BD-PSNR V |")
    write("-"*100)
    
    for _, row in bd_df.iterrows():
        write(f"{row['sequence']:<25} | {row['bd_br_psnr_Y']:>7.2f} | {row['bd_br_psnr_U']:>7.2f} | "
              f"{row['bd_br_psnr_V']:>7.2f} | {row['bd_psnr_Y']:>9.4f} | {row['bd_psnr_U']:>9.4f} | "
              f"{row['bd_psnr_V']:>9.4f} |")
    
    write("="*100)
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\n💾 Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--test-root', default='videos_test/test_frames_REAL')
    parser.add_argument('--orig-root', default='videos_test/test_orig_frames_pt')
    parser.add_argument('--qp', type=int, nargs='+', default=None)
    parser.add_argument('--output', default='results/bd_rate_per_channel.txt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    enhancer = load_enhancer_model(args.checkpoint, args.config, args.device)
    
    print("\n" + "="*100)
    print("COLLECTING PER-CHANNEL RATE-DISTORTION POINTS")
    print("="*100)
    
    rd_points = collect_per_channel_rd_points(
        enhancer, Path(args.test_root), Path(args.orig_root), args.qp, args.device
    )
    
    print("\n" + "="*100)
    print("CALCULATING PER-CHANNEL BD-RATE METRICS")
    print("="*100)
    
    bd_df = calculate_per_channel_bd_metrics(rd_points)
    
    csv_output = Path(args.output).with_suffix('.csv')
    bd_df.to_csv(csv_output, index=False)
    print(f"\n💾 Detailed results saved to: {csv_output}")
    
    print_comparison_table(bd_df, Path(args.output))


if __name__ == '__main__':
    main()
