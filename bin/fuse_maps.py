#!/usr/bin/env python3
"""
Enhanced maps fusion script - aggregates boundary maps, dequant maps, 
and creates block-level intelligence features with visualization support.
"""

import argparse
import os
import csv
import numpy as np
import glob
import re
from pathlib import Path

def sniff_csv(path):
    """Detect CSV delimiter"""
    with open(path, 'r', newline='') as f:
        head = f.read(4096)
        f.seek(0)
        return csv.Sniffer().sniff(head, delimiters=";\t,")

def load_block_statistics(csv_path, poc):
    """Load block rectangles and info for specific POC"""
    blocks = []
    try:
        dialect = sniff_csv(csv_path)
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=dialect.delimiter)
            for row in reader:
                if not row or not row[0].startswith("BlockStat"):
                    continue
                try:
                    p = int(row[1])
                    x, y, w, h = int(row[2]), int(row[3]), int(row[4]), int(row[5])
                    if p == poc:
                        blocks.append({'x': x, 'y': y, 'w': w, 'h': h})
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"WARN: Error reading blocks for POC {poc}: {e}")
    
    return blocks

def classify_block_size(w, h, ctu=128):
    """Classify block size category"""
    area = w * h
    if area <= 64:  # 8x8 or smaller
        return 0  # small
    elif area <= 256:  # 16x16 or smaller  
        return 1  # medium
    elif area <= 1024:  # 32x32 or smaller
        return 2  # large
    else:
        return 3  # very large

def extract_block_region(map_data, x, y, w, h):
    """Extract values from a map for specific block region"""
    x2 = x + w
    y2 = y + h
    if len(map_data.shape) == 2:
        return map_data[y:y2, x:x2]
    else:
        return map_data[:, y:y2, x:x2]

def calculate_block_features(blocks, dequant_maps, boundary_maps, width, height, debug=False):
    """Calculate enhanced block-level features"""
    
    # Initialize enhanced feature maps
    enhanced_maps = {
        'block_energy_contrast': np.zeros((height, width), dtype=np.float32),
        'quantization_severity': np.zeros((height, width), dtype=np.float32),
        'boundary_energy_drop': np.zeros((height, width), dtype=np.float32),
        'block_size_category': np.zeros((height, width), dtype=np.uint8),
        'complexity_mismatch': np.zeros((height, width), dtype=np.float32),
        'ac_density_per_block': np.zeros((height, width), dtype=np.float32),
        'dc_variation': np.zeros((height, width), dtype=np.float32)
    }
    
    # Extract individual maps
    y_ac_energy = dequant_maps['y_ac_energy']
    y_nz_density = dequant_maps['y_nz_density']
    y_dc = dequant_maps.get('y_dc', np.zeros_like(y_ac_energy))
    
    boundary_bin = boundary_maps['boundary_bin']
    boundary_weight = boundary_maps['boundary_weight']
    size_map_norm = boundary_maps['size_map_norm']
    
    processed_blocks = 0
    
    for block in blocks:
        x, y, w, h = block['x'], block['y'], block['w'], block['h']
        
        # Clamp to image bounds
        x2 = min(x + w, width)
        y2 = min(y + h, height)
        
        if x >= width or y >= height or x2 <= x or y2 <= y:
            continue
        
        # Extract block regions
        block_energy = extract_block_region(y_ac_energy, x, y, w, h)
        block_density = extract_block_region(y_nz_density, x, y, w, h)
        block_dc = extract_block_region(y_dc, x, y, w, h)
        block_boundaries = extract_block_region(boundary_bin, x, y, w, h)
        
        # Calculate block-level statistics
        mean_energy = np.mean(block_energy)
        std_energy = np.std(block_energy)
        mean_density = np.mean(block_density)
        mean_dc = np.mean(block_dc)
        std_dc = np.std(block_dc)
        
        # Block size category
        size_category = classify_block_size(w, h)
        
        # Energy contrast (variation within block)
        energy_contrast = std_energy / (mean_energy + 1e-6)
        
        # Quantization severity (low density despite high original energy suggests heavy quantization)
        quant_severity = mean_energy / (mean_density + 1e-6) if mean_density > 0 else mean_energy
        
        # Boundary energy drop (how much energy drops at boundaries)
        boundary_pixels = np.sum(block_boundaries > 0)
        total_pixels = w * h
        boundary_ratio = boundary_pixels / total_pixels if total_pixels > 0 else 0
        boundary_energy_drop = boundary_ratio * mean_energy
        
        # Complexity mismatch (large blocks with high complexity = potential artifacts)
        expected_complexity = 1.0 / (size_category + 1)  # smaller blocks should have higher complexity
        actual_complexity = mean_energy
        complexity_mismatch = abs(actual_complexity - expected_complexity * 100)
        
        # AC density per block (normalized by block size)
        ac_density_normalized = mean_density * (w * h) / 1024  # normalize to 32x32 block
        
        # Fill the enhanced maps
        enhanced_maps['block_energy_contrast'][y:y2, x:x2] = energy_contrast
        enhanced_maps['quantization_severity'][y:y2, x:x2] = quant_severity
        enhanced_maps['boundary_energy_drop'][y:y2, x:x2] = boundary_energy_drop
        enhanced_maps['block_size_category'][y:y2, x:x2] = size_category
        enhanced_maps['complexity_mismatch'][y:y2, x:x2] = complexity_mismatch
        enhanced_maps['ac_density_per_block'][y:y2, x:x2] = ac_density_normalized
        enhanced_maps['dc_variation'][y:y2, x:x2] = std_dc
        
        processed_blocks += 1
    
    if debug:
        print(f"  Processed {processed_blocks} blocks for enhanced features")
    
    return enhanced_maps

def create_comprehensive_visualization(all_maps, output_path, poc, debug=False):
    """Create comprehensive visualization of all map types"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Determine grid layout (4x3 = 12 subplots)
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Original maps (top 2 rows)
        maps_to_plot = [
            ('y_ac_energy', 'AC Energy', 'viridis'),
            ('y_nz_density', 'NZ Density', 'plasma'),
            ('y_dc', 'DC Coefficients', 'RdBu_r'),
            ('boundary_bin', 'Binary Boundaries', 'gray'),
            ('boundary_weight', 'Boundary Weights', 'magma'),
            ('size_map_norm', 'Block Sizes', 'viridis'),
            
            # Enhanced maps (bottom 2 rows)
            ('block_energy_contrast', 'Energy Contrast', 'coolwarm'),
            ('quantization_severity', 'Quantization Severity', 'Reds'),
            ('boundary_energy_drop', 'Boundary Energy Drop', 'OrRd'),
            ('block_size_category', 'Size Categories', 'tab10'),
            ('complexity_mismatch', 'Complexity Mismatch', 'YlOrRd'),
            ('ac_density_per_block', 'AC Density/Block', 'Blues')
        ]
        
        for idx, (map_name, title, cmap) in enumerate(maps_to_plot):
            if map_name not in all_maps:
                continue
                
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            data = all_maps[map_name]
            
            # Special handling for DC (center around 0)
            if map_name == 'y_dc':
                vmax = max(abs(data.min()), abs(data.max()))
                im = ax.imshow(data, cmap=cmap, origin='upper', 
                             vmin=-vmax, vmax=vmax, interpolation='nearest')
            else:
                im = ax.imshow(data, cmap=cmap, origin='upper', interpolation='nearest')
            
            ax.set_title(f'{title}\n({data.shape[1]}x{data.shape[0]})', fontsize=10)
            ax.axis('off')
            
            # Add colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            
            # Add statistics
            stats_text = f'min: {data.min():.2f}\nmax: {data.max():.2f}\nmean: {data.mean():.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'Comprehensive Maps Visualization - POC {poc}', fontsize=14, fontweight='bold')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if debug:
            print(f"  Saved comprehensive visualization: {output_path}")
        
        return True
        
    except ImportError:
        print("WARN: matplotlib not available, skipping visualization")
        return False
    except Exception as e:
        print(f"WARN: Failed to create visualization: {e}")
        return False

def auto_detect_poc_range_from_files(directory):
    """Auto-detect POC range from available files"""
    # Try dequant files first
    dequant_files = glob.glob(os.path.join(directory, "neural_maps/dequant_maps_poc*.npz"))
    if dequant_files:
        pocs = []
        for f in dequant_files:
            match = re.search(r'dequant_maps_poc(\d+)\.npz', os.path.basename(f))
            if match:
                pocs.append(int(match.group(1)))
        if pocs:
            return min(pocs), max(pocs)
    
    # Try boundary files
    boundary_files = glob.glob(os.path.join(directory, "boundary_maps/boundary_maps_poc*.npz"))
    if boundary_files:
        pocs = []
        for f in boundary_files:
            match = re.search(r'boundary_maps_poc(\d+)\.npz', os.path.basename(f))
            if match:
                pocs.append(int(match.group(1)))
        if pocs:
            return min(pocs), max(pocs)
    
    return 0, 0

def main():
    parser = argparse.ArgumentParser(description="Fuse boundary maps, dequant maps and create enhanced features")
    parser.add_argument("input_dir", help="Directory containing neural_maps/, boundary_maps/, and block_stats.csv")
    parser.add_argument("--outdir", default="fused_maps", help="Output directory for fused maps")
    parser.add_argument("--poc-from", type=int, help="Start POC (auto-detected if not specified)")
    parser.add_argument("--poc-to", type=int, help="End POC (auto-detected if not specified)")
    parser.add_argument("--visualize", action="store_true", help="Generate comprehensive visualizations")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    # Check input directories
    neural_maps_dir = os.path.join(args.input_dir, "neural_maps")
    boundary_maps_dir = os.path.join(args.input_dir, "boundary_maps")
    block_stats_path = os.path.join(args.input_dir, "block_stats.csv")
    
    if not os.path.isdir(neural_maps_dir):
        print(f"ERROR: Neural maps directory not found: {neural_maps_dir}")
        return 1
    
    if not os.path.isdir(boundary_maps_dir):
        print(f"ERROR: Boundary maps directory not found: {boundary_maps_dir}")
        return 1
        
    if not os.path.isfile(block_stats_path):
        print(f"ERROR: Block statistics file not found: {block_stats_path}")
        return 1
    
    if args.debug:
        print(f"Processing maps from: {args.input_dir}")
        print(f"Output directory: {args.outdir}")
    
    # Auto-detect POC range
    poc_from = args.poc_from
    poc_to = args.poc_to
    if poc_from is None or poc_to is None:
        auto_from, auto_to = auto_detect_poc_range_from_files(args.input_dir)
        if poc_from is None:
            poc_from = auto_from
        if poc_to is None:
            poc_to = auto_to
        if args.debug:
            print(f"Auto-detected POC range: {poc_from}-{poc_to}")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Process each POC
    generated_count = 0
    for poc in range(poc_from, poc_to + 1):
        if args.debug:
            print(f"Processing POC {poc}...")
        
        # Load dequant maps
        dequant_file = os.path.join(neural_maps_dir, f"dequant_maps_poc{poc}.npz")
        if not os.path.isfile(dequant_file):
            if args.debug:
                print(f"  Skipping POC {poc}: no dequant maps found")
            continue
        
        dequant_data = np.load(dequant_file)
        
        # Load boundary maps
        boundary_file = os.path.join(boundary_maps_dir, f"boundary_maps_poc{poc}.npz")
        if not os.path.isfile(boundary_file):
            if args.debug:
                print(f"  Skipping POC {poc}: no boundary maps found")
            continue
        
        boundary_data = np.load(boundary_file)
        
        # Get image dimensions
        height, width = dequant_data['y_ac_energy'].shape
        
        # Load block statistics for this POC
        blocks = load_block_statistics(block_stats_path, poc)
        if not blocks:
            if args.debug:
                print(f"  Warning: no blocks found for POC {poc}")
        
        if args.debug:
            print(f"  Found {len(blocks)} blocks, resolution: {width}x{height}")
        
        # Calculate enhanced features
        enhanced_maps = calculate_block_features(blocks, dequant_data, boundary_data, 
                                               width, height, args.debug)
        
        # Combine all maps
        all_maps = {}
        
        # Add original dequant maps
        for key in dequant_data.keys():
            all_maps[key] = dequant_data[key]
        
        # Add original boundary maps  
        for key in boundary_data.keys():
            all_maps[key] = boundary_data[key]
        
        # Add enhanced maps
        for key, value in enhanced_maps.items():
            all_maps[key] = value
        
        # Save fused maps
        output_file = os.path.join(args.outdir, f"fused_maps_poc{poc}.npz")
        np.savez_compressed(output_file, **all_maps)
        
        if args.debug:
            total_maps = len(all_maps)
            print(f"  Saved {total_maps} maps to: {output_file}")
            print(f"    Original: {len(dequant_data)} dequant + {len(boundary_data)} boundary")
            print(f"    Enhanced: {len(enhanced_maps)} new features")
        
        # Generate visualization if requested
        if args.visualize:
            viz_file = os.path.join(args.outdir, f"fused_maps_poc{poc}_visualization.png")
            create_comprehensive_visualization(all_maps, viz_file, poc, args.debug)
        
        generated_count += 1
    
    print(f"Successfully fused {generated_count} POC files")
    print(f"Output directory: {args.outdir}")
    
    if generated_count > 0:
        print("\nGenerated maps per POC:")
        print("  Original dequant: y_ac_energy, y_nz_density, y_dc")
        print("  Original boundary: boundary_bin, boundary_weight, size_map_norm")
        print("  Enhanced features:")
        print("    - block_energy_contrast: energy variation within blocks")
        print("    - quantization_severity: estimated quantization loss")
        print("    - boundary_energy_drop: energy loss at block boundaries")
        print("    - block_size_category: classified block sizes (0-3)")
        print("    - complexity_mismatch: size vs complexity inconsistencies")
        print("    - ac_density_per_block: block-normalized AC density")
        print("    - dc_variation: DC coefficient variation within blocks")
        
        if args.visualize:
            print(f"  + Comprehensive visualizations: *_visualization.png")
    
    return 0

if __name__ == "__main__":
    exit(main())
