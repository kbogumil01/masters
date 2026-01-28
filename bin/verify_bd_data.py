#!/usr/bin/env python3
"""
Verify test data structure before running BD-rate analysis.
Checks if all required files and directories are present.
"""

import sys
from pathlib import Path
from collections import defaultdict
import re


def extract_sequence_info(dirname: str):
    """Extract sequence name and QP from directory name."""
    # Pattern: sequence_name_AI_QP##_...
    match = re.match(r'(.+?)_(?:AI|RA)_QP(\d+)', dirname)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def verify_test_structure(test_root: Path, orig_root: Path):
    """Verify test data structure and report issues."""
    
    print("=" * 80)
    print("BD-RATE DATA VERIFICATION")
    print("=" * 80)
    print()
    
    issues = []
    warnings = []
    
    # Check directories exist
    if not test_root.exists():
        issues.append(f"❌ Test directory not found: {test_root}")
        return issues, warnings
    
    if not orig_root.exists():
        issues.append(f"❌ Original directory not found: {orig_root}")
        return issues, warnings
    
    print(f"✅ Test directory exists: {test_root}")
    print(f"✅ Original directory exists: {orig_root}")
    print()
    
    # Collect sequences and QP points
    sequences = defaultdict(lambda: {'qps': set(), 'dirs': []})
    
    for test_dir in sorted(test_root.iterdir()):
        if not test_dir.is_dir():
            continue
        
        seq_name, qp = extract_sequence_info(test_dir.name)
        if seq_name and qp:
            sequences[seq_name]['qps'].add(qp)
            sequences[seq_name]['dirs'].append(test_dir)
    
    print(f"Found {len(sequences)} unique sequences:")
    print()
    
    # Verify each sequence
    for seq_name, data in sorted(sequences.items()):
        print(f"📹 {seq_name}")
        print(f"   QP points: {sorted(data['qps'])} ({len(data['qps'])} total)")
        
        # Check if enough QP points
        if len(data['qps']) < 4:
            issues.append(f"❌ {seq_name}: Need at least 4 QP points, found {len(data['qps'])}")
        elif len(data['qps']) == 4:
            warnings.append(f"⚠️  {seq_name}: Only 4 QP points (minimum), 5+ recommended")
        else:
            print(f"   ✅ Sufficient QP points")
        
        # Check for original directory
        orig_seq_dir = orig_root / seq_name
        if not orig_seq_dir.exists():
            issues.append(f"❌ {seq_name}: Original directory not found: {orig_seq_dir}")
            print(f"   ❌ Original directory missing")
        else:
            print(f"   ✅ Original directory found")
            
            # Count frames in first QP directory
            first_dir = data['dirs'][0]
            decoded_frames = list(first_dir.glob("frame_*.pt"))
            original_frames = list(orig_seq_dir.glob("frame_*.pt"))
            
            print(f"   📊 Frames: {len(decoded_frames)} decoded, {len(original_frames)} original")
            
            if len(decoded_frames) == 0:
                issues.append(f"❌ {seq_name}: No decoded frames found in {first_dir}")
            
            if len(original_frames) == 0:
                issues.append(f"❌ {seq_name}: No original frames found in {orig_seq_dir}")
            
            if abs(len(decoded_frames) - len(original_frames)) > 5:
                warnings.append(f"⚠️  {seq_name}: Frame count mismatch "
                               f"({len(decoded_frames)} vs {len(original_frames)})")
        
        print()
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    if not issues and not warnings:
        print("✅ All checks passed! Data structure is ready for BD-rate analysis.")
        print()
        print("You can now run:")
        print(f"  ./run_bd_analysis.sh checkpoints_v2/epoch=998.ckpt")
        return [], []
    
    if warnings:
        print(f"⚠️  {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"   {w}")
        print()
    
    if issues:
        print(f"❌ {len(issues)} CRITICAL ISSUE(S):")
        for i in issues:
            print(f"   {i}")
        print()
        print("Please fix these issues before running BD-rate analysis.")
        return issues, warnings
    
    print("⚠️  Found warnings but no critical issues.")
    print("You can proceed, but results may be suboptimal.")
    print()
    
    return issues, warnings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify BD-rate test data structure")
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
        help='Root directory with original frames'
    )
    
    args = parser.parse_args()
    
    issues, warnings = verify_test_structure(Path(args.test_root), Path(args.orig_root))
    
    # Exit code
    if issues:
        sys.exit(1)
    elif warnings:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
