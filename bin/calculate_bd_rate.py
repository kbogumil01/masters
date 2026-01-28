#!/usr/bin/env python3
"""
Calculate BD-BR (Bjøntegaard Delta Bit-Rate) and BD-PSNR/BD-SSIM metrics
for neural network enhanced video quality.

This script compares:
- Original decoded frames (VVC output)
- Neural network enhanced frames
Against original uncompressed frames across multiple QP values.
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import yaml
from scipy import interpolate
from typing import Dict, List, Tuple
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancer.models.enhancer import Enhancer
from enhancer.config import Config
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def bjontegaard_delta(R1, PSNR1, R2, PSNR2):
    """
    Calculate Bjøntegaard Delta metric (BD-rate or BD-PSNR).
    
    Args:
        R1, PSNR1: Rate-distortion points for reference (anchor)
        R2, PSNR2: Rate-distortion points for test
        
    Returns:
        BD-rate (%) if calculating bitrate savings
        BD-PSNR (dB) if calculating quality improvement
    """
    lR1 = np.log(R1)
    lR2 = np.log(R2)
    
    # Cubic polynomial fit
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)
    
    # Integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))
    
    # Numerical integration
    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)
    
    int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
    int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    
    # Calculate average difference
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    
    return avg_diff


def bjontegaard_psnr(R1, PSNR1, R2, PSNR2):
    """Calculate BD-PSNR (quality improvement in dB)."""
    # Swap roles: fit rate as function of PSNR
    p1 = np.polyfit(R1, PSNR1, 3)
    p2 = np.polyfit(R2, PSNR2, 3)
    
    # Integration interval
    min_int = max(min(R1), min(R2))
    max_int = min(max(R1), max(R2))
    
    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)
    
    int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
    int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    
    avg_diff = (int2 - int1) / (max_int - min_int)
    
    return avg_diff


def load_enhancer_model(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load trained enhancer model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    config = Config.load(config_path)
    enhancer = Enhancer(config=config.enhancer)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in ckpt:
        # Lightning checkpoint
        state_dict = ckpt["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("enhancer."):
                new_state_dict[k.replace("enhancer.", "")] = v
        
        if new_state_dict:
            enhancer.load_state_dict(new_state_dict, strict=True)
        else:
            enhancer.load_state_dict(state_dict, strict=False)
    else:
        # Direct weights
        enhancer.load_state_dict(ckpt, strict=True)
    
    enhancer.eval()
    enhancer.to(device)
    
    print("Model loaded successfully.")
    return enhancer


def extract_qp_from_path(path: str) -> int:
    """Extract QP value from folder name."""
    match = re.search(r'_QP(\d+)_', path)
    if match:
        return int(match.group(1))
    return None


def calculate_metrics_for_sequence(
    enhancer,
    decoded_frames_dir: Path,
    original_frames_dir: Path,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Calculate PSNR and SSIM for one sequence (all frames).
    
    Returns:
        dict with keys: psnr_decoded, psnr_enhanced, ssim_decoded, ssim_enhanced
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Try multiple file patterns and locations, but SKIP fused_maps
    decoded_files = sorted(decoded_frames_dir.glob("frame_*.pt"))
    
    # If no frame_*.pt files, check subdirectories but skip fused_maps
    if len(decoded_files) == 0:
        all_pt_files = sorted(decoded_frames_dir.glob("*.pt"))
        # Filter out files with "fused_maps" in the name
        decoded_files = [f for f in all_pt_files if "fused_maps" not in f.name.lower()]
    
    if len(decoded_files) == 0:
        print(f"  ⚠️  No frames found in {decoded_frames_dir}")
        return None
    
    psnr_dec_list = []
    psnr_enh_list = []
    ssim_dec_list = []
    ssim_enh_list = []
    
    with torch.no_grad():
        for dec_file in tqdm(decoded_files, desc="  Processing frames", leave=False):
            try:
                # Load decoded frame
                dec_data = torch.load(dec_file, map_location="cpu")
                
                # Check if chunk exists
                if "chunk" not in dec_data:
                    print(f"  ⚠️  Skipping {dec_file.name}: no 'chunk' key (keys: {list(dec_data.keys())})")
                    continue
                    
                decoded = dec_data["chunk"].float() / 255.0  # Normalize to [0, 1]
            except Exception as e:
                print(f"  ⚠️  Error loading {dec_file.name}: {e}")
                continue
            
            # Find matching original frame
            # Handle different naming conventions: frame_*.pt, fused_maps_poc*.pt, frame_poc*.pt
            if "poc" in dec_file.name:
                # Extract POC number
                poc_match = re.search(r'poc(\d+)', dec_file.name)
                if poc_match:
                    poc_num = poc_match.group(1)
                    orig_file = original_frames_dir / f"frame_poc{poc_num}.pt"
                else:
                    orig_file = original_frames_dir / dec_file.name
            else:
                orig_file = original_frames_dir / dec_file.name
            
            if not orig_file.exists():
                print(f"  ⚠️  Original frame not found: {orig_file}")
                continue
            
            try:
                orig_data = torch.load(orig_file, map_location="cpu")
                
                if "chunk" not in orig_data:
                    print(f"  ⚠️  Skipping {orig_file.name}: no 'chunk' key")
                    continue
                    
                original = orig_data["chunk"].float() / 255.0
            except Exception as e:
                print(f"  ⚠️  Error loading original {orig_file.name}: {e}")
                continue
            
            # Move to device and add batch dimension
            decoded = decoded.unsqueeze(0).to(device)
            original = original.unsqueeze(0).to(device)
            
            # Extract metadata - must match training format: [profile, qp/64, alf, sao, db, is_intra]
            metadata = dec_data.get("seq_meta", {})
            
            # Profile: AI=1, RA=0
            profile = str(metadata.get("profile", "AI")).upper()
            profile_ai = 1.0 if "AI" in profile else 0.0
            
            qp = float(metadata.get("qp", 32))
            qp_n = qp / 64.0
            alf = float(metadata.get("alf", 0))
            sao = float(metadata.get("sao", 0))
            db = float(metadata.get("db", 0))
            
            # is_intra: check from data or default based on profile
            is_intra = float(dec_data.get("is_intra", 1 if profile_ai else 0))
            
            # 6-channel metadata tensor matching dataset.py _norm_metadata
            meta_tensor = torch.tensor(
                [profile_ai, qp_n, alf, sao, db, is_intra], 
                dtype=torch.float32
            ).view(1, 6, 1, 1).to(device)
            
            # Get VVC features if available
            if "vvc_features" in dec_data:
                vvc_feat = dec_data["vvc_features"].float().unsqueeze(0).to(device)
            else:
                vvc_feat = None
            
            # Enhance frame
            enhanced = enhancer(decoded, meta_tensor, vvc_feat)
            
            # Calculate metrics
            psnr_dec = psnr_metric(decoded, original)
            psnr_enh = psnr_metric(enhanced, original)
            ssim_dec = ssim_metric(decoded, original)
            ssim_enh = ssim_metric(enhanced, original)
            
            psnr_dec_list.append(psnr_dec.item())
            psnr_enh_list.append(psnr_enh.item())
            ssim_dec_list.append(ssim_dec.item())
            ssim_enh_list.append(ssim_enh.item())
    
    return {
        'psnr_decoded': np.mean(psnr_dec_list),
        'psnr_enhanced': np.mean(psnr_enh_list),
        'ssim_decoded': np.mean(ssim_dec_list),
        'ssim_enhanced': np.mean(ssim_enh_list),
        'num_frames': len(psnr_dec_list)
    }


def get_bitrate_from_log(sequence_dir: Path, encoded_logs_dir: Path = None) -> float:
    """
    Extract bitrate from VVC encoder log file.
    
    Args:
        sequence_dir: Directory with decoded frames (e.g., videos_test/test_frames_REAL/Johnny_...)
        encoded_logs_dir: Directory with encoder logs (e.g., videos_test/encoded_test/)
    
    Returns bitrate in kbps, or estimates from QP if not available.
    """
    # Get sequence folder name (e.g., "Johnny_1280x720_60_AI_QP28_ALF0_DB0_SAO0")
    seq_folder_name = sequence_dir.name
    
    # Try to find log in encoded_logs_dir first
    if encoded_logs_dir is None:
        # Default: look in videos_test/encoded_test/ relative to test_frames_REAL
        encoded_logs_dir = sequence_dir.parent.parent / "encoded_test"
    
    log_file = encoded_logs_dir / f"{seq_folder_name}.log"
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            content = f.read()
            
            # vvenc format: "vvenc [info]:	       64    a    1115.4825   39.3154..."
            # Look for the summary line with bitrate
            match = re.search(r'vvenc \[info\]:\s+\d+\s+a\s+([\d.]+)', content)
            if match:
                return float(match.group(1))
            
            # Alternative: "avg_bitrate= 1115.48 kbps"
            match = re.search(r'avg_bitrate[=:]\s*([\d.]+)\s*kbps', content, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            # Generic: any "XXXX.XX kbps" pattern
            match = re.search(r'(\d+\.?\d*)\s*kbps', content)
            if match:
                return float(match.group(1))
    
    # Try old location (encode.log in sequence_dir)
    old_log_file = sequence_dir / "encode.log"
    if old_log_file.exists():
        with open(old_log_file, 'r') as f:
            content = f.read()
            match = re.search(r'(\d+\.?\d*)\s*kbps', content)
            if match:
                return float(match.group(1))
    
    # Fallback: estimate from QP (very rough!)
    qp = extract_qp_from_path(str(sequence_dir))
    if qp:
        print(f"  ⚠️  No log found for {seq_folder_name}, using QP-based estimate")
        base_bitrate = 10000  # kbps at QP=22
        return base_bitrate * (0.85 ** (qp - 22))
    
    return None


def collect_rd_points(
    enhancer,
    test_root: Path,
    orig_root: Path,
    qp_values: List[int],
    device: str = 'cuda'
) -> Dict[str, Dict[str, List]]:
    """
    Collect Rate-Distortion points for all sequences.
    
    Returns nested dict:
        sequence_name -> {
            'qp': [...],
            'bitrate': [...],
            'psnr_decoded': [...],
            'psnr_enhanced': [...],
            'ssim_decoded': [...],
            'ssim_enhanced': [...]
        }
    """
    results = {}
    
    # Find all unique sequences
    all_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
    
    # Group by sequence name (strip QP and codec parameters)
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
            'qp': [],
            'bitrate': [],
            'psnr_decoded': [],
            'psnr_enhanced': [],
            'ssim_decoded': [],
            'ssim_enhanced': []
        }
        
        # Find original frames directory
        orig_seq_dir = orig_root / seq_name
        if not orig_seq_dir.exists():
            print(f"  ⚠️  Original directory not found: {orig_seq_dir}")
            continue
        
        # Process each QP point
        for seq_dir in sorted(seq_dirs):
            qp = extract_qp_from_path(seq_dir.name)
            if qp is None or (qp_values and qp not in qp_values):
                continue
            
            print(f"\n  QP={qp}")
            
            # Calculate metrics
            metrics = calculate_metrics_for_sequence(
                enhancer,
                seq_dir,
                orig_seq_dir,
                device
            )
            
            if metrics is None:
                continue
            
            # Get bitrate
            bitrate = get_bitrate_from_log(seq_dir)
            if bitrate is None:
                print(f"  ⚠️  Could not determine bitrate, using QP-based estimate")
                bitrate = 10000 * (0.85 ** (qp - 22))
            
            results[seq_name]['qp'].append(qp)
            results[seq_name]['bitrate'].append(bitrate)
            results[seq_name]['psnr_decoded'].append(metrics['psnr_decoded'])
            results[seq_name]['psnr_enhanced'].append(metrics['psnr_enhanced'])
            results[seq_name]['ssim_decoded'].append(metrics['ssim_decoded'])
            results[seq_name]['ssim_enhanced'].append(metrics['ssim_enhanced'])
            
            print(f"    Bitrate: {bitrate:.2f} kbps")
            print(f"    PSNR decoded:  {metrics['psnr_decoded']:.4f} dB")
            print(f"    PSNR enhanced: {metrics['psnr_enhanced']:.4f} dB (+{metrics['psnr_enhanced'] - metrics['psnr_decoded']:.4f} dB)")
            print(f"    SSIM decoded:  {metrics['ssim_decoded']:.6f}")
            print(f"    SSIM enhanced: {metrics['ssim_enhanced']:.6f} (+{metrics['ssim_enhanced'] - metrics['ssim_decoded']:.6f})")
    
    return results


def calculate_bd_metrics(rd_points: Dict) -> pd.DataFrame:
    """
    Calculate BD-BR and BD-PSNR/SSIM for all sequences.
    
    Args:
        rd_points: Output from collect_rd_points()
        
    Returns:
        DataFrame with BD metrics for each sequence
    """
    bd_results = []
    
    for seq_name, data in rd_points.items():
        if len(data['qp']) < 4:
            print(f"⚠️  Skipping {seq_name}: need at least 4 QP points, got {len(data['qp'])}")
            continue
        
        # Convert to numpy arrays
        bitrate = np.array(data['bitrate'])
        psnr_dec = np.array(data['psnr_decoded'])
        psnr_enh = np.array(data['psnr_enhanced'])
        ssim_dec = np.array(data['ssim_decoded'])
        ssim_enh = np.array(data['ssim_enhanced'])
        
        try:
            # BD-BR (PSNR): bitrate savings at same PSNR quality
            # Negative = savings (enhanced uses less bitrate for same quality)
            bd_br_psnr = bjontegaard_delta(bitrate, psnr_dec, bitrate, psnr_enh)
            
            # BD-PSNR: quality improvement at same bitrate
            # Positive = improvement (enhanced has better quality at same bitrate)
            bd_psnr = bjontegaard_psnr(bitrate, psnr_dec, bitrate, psnr_enh)
            
            # BD-BR (SSIM): bitrate savings based on SSIM
            # Convert SSIM to dB-like scale for better interpolation
            ssim_dec_db = -10 * np.log10(1 - ssim_dec + 1e-10)
            ssim_enh_db = -10 * np.log10(1 - ssim_enh + 1e-10)
            bd_br_ssim = bjontegaard_delta(bitrate, ssim_dec_db, bitrate, ssim_enh_db)
            
            # BD-SSIM: SSIM improvement
            bd_ssim = bjontegaard_psnr(bitrate, ssim_dec, bitrate, ssim_enh)
            
            bd_results.append({
                'sequence': seq_name,
                'num_points': len(data['qp']),
                'bd_br_psnr': bd_br_psnr,
                'bd_psnr': bd_psnr,
                'bd_br_ssim': bd_br_ssim,
                'bd_ssim': bd_ssim,
                'avg_psnr_gain': np.mean(psnr_enh - psnr_dec),
                'avg_ssim_gain': np.mean(ssim_enh - ssim_dec)
            })
            
        except Exception as e:
            print(f"⚠️  Error calculating BD metrics for {seq_name}: {e}")
            continue
    
    return pd.DataFrame(bd_results)


def print_bd_report(bd_df: pd.DataFrame, output_file: Path = None):
    """Print comprehensive BD-rate report."""
    
    report_lines = []
    
    def write(line=""):
        print(line)
        report_lines.append(line)
    
    write("=" * 100)
    write("📊 BD-RATE ANALYSIS - Neural Network Video Enhancement")
    write("=" * 100)
    write()
    
    write("─" * 100)
    write("PER-SEQUENCE RESULTS")
    write("─" * 100)
    write()
    write(f"{'Sequence':<30} {'Points':>6} {'BD-BR(PSNR)':>12} {'BD-PSNR':>10} {'BD-BR(SSIM)':>12} {'BD-SSIM':>10}")
    write(f"{'':30} {'':6} {'[%]':>12} {'[dB]':>10} {'[%]':>12} {'':>10}")
    write("─" * 100)
    
    for _, row in bd_df.iterrows():
        write(f"{row['sequence']:<30} {row['num_points']:>6} "
              f"{row['bd_br_psnr']:>11.2f}% {row['bd_psnr']:>9.4f} dB "
              f"{row['bd_br_ssim']:>11.2f}% {row['bd_ssim']:>9.6f}")
    
    write("─" * 100)
    write()
    
    # Overall statistics
    write("─" * 100)
    write("OVERALL STATISTICS")
    write("─" * 100)
    write()
    
    avg_bd_br_psnr = bd_df['bd_br_psnr'].mean()
    avg_bd_psnr = bd_df['bd_psnr'].mean()
    avg_bd_br_ssim = bd_df['bd_br_ssim'].mean()
    avg_bd_ssim = bd_df['bd_ssim'].mean()
    
    write(f"Average BD-BR (PSNR):  {avg_bd_br_psnr:>8.2f}%")
    write(f"Average BD-PSNR:       {avg_bd_psnr:>8.4f} dB")
    write(f"Average BD-BR (SSIM):  {avg_bd_br_ssim:>8.2f}%")
    write(f"Average BD-SSIM:       {avg_bd_ssim:>8.6f}")
    write()
    
    write(f"Median BD-BR (PSNR):   {bd_df['bd_br_psnr'].median():>8.2f}%")
    write(f"Median BD-PSNR:        {bd_df['bd_psnr'].median():>8.4f} dB")
    write(f"Median BD-BR (SSIM):   {bd_df['bd_br_ssim'].median():>8.2f}%")
    write(f"Median BD-SSIM:        {bd_df['bd_ssim'].median():>8.6f}")
    write()
    
    # Interpretation
    write("─" * 100)
    write("INTERPRETATION")
    write("─" * 100)
    write()
    
    if avg_bd_br_psnr < 0:
        write(f"✅ Neural enhancement provides {abs(avg_bd_br_psnr):.2f}% bitrate savings (PSNR metric)")
        write(f"   → At the same PSNR quality, enhanced videos use {abs(avg_bd_br_psnr):.2f}% less bitrate")
    else:
        write(f"⚠️  Neural enhancement requires {avg_bd_br_psnr:.2f}% more bitrate (PSNR metric)")
    
    write()
    
    if avg_bd_psnr > 0:
        write(f"✅ Neural enhancement provides {avg_bd_psnr:.4f} dB PSNR improvement")
        write(f"   → At the same bitrate, enhanced videos have {avg_bd_psnr:.4f} dB better PSNR")
    else:
        write(f"⚠️  Neural enhancement reduces PSNR by {abs(avg_bd_psnr):.4f} dB")
    
    write()
    
    if avg_bd_br_ssim < 0:
        write(f"✅ Neural enhancement provides {abs(avg_bd_br_ssim):.2f}% bitrate savings (SSIM metric)")
    else:
        write(f"⚠️  Neural enhancement requires {avg_bd_br_ssim:.2f}% more bitrate (SSIM metric)")
    
    write()
    
    if avg_bd_ssim > 0:
        write(f"✅ Neural enhancement provides {avg_bd_ssim:.6f} SSIM improvement")
    else:
        write(f"⚠️  Neural enhancement reduces SSIM by {abs(avg_bd_ssim):.6f}")
    
    write()
    write("=" * 100)
    
    # Save to file
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\n💾 Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BD-BR and BD-PSNR/SSIM metrics for neural network video enhancement"
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--test-root',
        type=str,
        default='videos_test/test_frames_REAL',
        help='Root directory with decoded test frames'
    )
    parser.add_argument(
        '--orig-root',
        type=str,
        default='videos_test/test_orig_frames_pt',
        help='Root directory with original uncompressed frames'
    )
    parser.add_argument(
        '--qp',
        type=int,
        nargs='+',
        default=None,
        help='QP values to include (default: all available)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/bd_rate_report.txt',
        help='Output file for report'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Load model
    enhancer = load_enhancer_model(args.checkpoint, args.config, args.device)
    
    # Collect RD points
    print("\n" + "=" * 100)
    print("COLLECTING RATE-DISTORTION POINTS")
    print("=" * 100)
    
    rd_points = collect_rd_points(
        enhancer,
        Path(args.test_root),
        Path(args.orig_root),
        args.qp,
        args.device
    )
    
    # Calculate BD metrics
    print("\n" + "=" * 100)
    print("CALCULATING BD-RATE METRICS")
    print("=" * 100)
    
    bd_df = calculate_bd_metrics(rd_points)
    
    # Save detailed results
    csv_output = Path(args.output).with_suffix('.csv')
    bd_df.to_csv(csv_output, index=False)
    print(f"\n💾 Detailed results saved to: {csv_output}")
    
    # Print and save report
    print_bd_report(bd_df, Path(args.output))


if __name__ == '__main__':
    main()
