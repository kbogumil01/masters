#!/usr/bin/env python3
"""
Check which QP/sequence combinations are missing decoded frames in test_frames_REAL.
Generate commands to decode missing combinations.
"""

from pathlib import Path
import re
from collections import defaultdict

def main():
    decoded_test = Path("videos_test/decoded_test")
    test_frames_real = Path("videos_test/test_frames_REAL")
    
    # Expected QP values
    expected_qps = [28, 32, 37, 42, 47]
    
    # Find all sequences in decoded_test
    sequences = defaultdict(set)
    for d in sorted(decoded_test.iterdir()):
        if not d.is_dir():
            continue
        match = re.match(r'(.+?)_(AI|RA)_QP(\d+)', d.name)
        if match:
            seq_name = match.group(1)
            qp = int(match.group(3))
            sequences[seq_name].add(qp)
    
    print("=" * 80)
    print("MISSING TEST FRAMES ANALYSIS")
    print("=" * 80)
    print()
    
    missing_total = 0
    commands = []
    
    for seq_name in sorted(sequences.keys()):
        available_qps = sequences[seq_name]
        
        # Check which are already decoded in test_frames_REAL
        decoded_qps = set()
        for d in test_frames_real.iterdir():
            if d.is_dir() and d.name.startswith(seq_name):
                match = re.search(r'QP(\d+)', d.name)
                if match:
                    qp = int(match.group(1))
                    # Check if has frames
                    if list(d.glob("*.pt")):
                        decoded_qps.add(qp)
        
        missing = available_qps - decoded_qps
        
        if missing:
            print(f"📹 {seq_name}")
            print(f"   Available QPs: {sorted(available_qps)}")
            print(f"   Already decoded: {sorted(decoded_qps)}")
            print(f"   ❌ Missing: {sorted(missing)}")
            missing_total += len(missing)
            
            # Generate decode commands for missing QPs
            for qp in sorted(missing):
                # Find the directory with this QP (take first ALF/DB/SAO combination)
                for d in decoded_test.iterdir():
                    if d.name.startswith(f"{seq_name}_AI_QP{qp}_"):
                        commands.append(f"# Decode {seq_name} QP{qp}")
                        commands.append(f"# Source: {d.name}")
                        print(f"   → Need to decode: {d.name}")
                        break
            print()
    
    print("=" * 80)
    print(f"SUMMARY: {missing_total} missing QP/sequence combinations")
    print("=" * 80)
    print()
    
    if missing_total == 0:
        print("✅ All QP points are already decoded!")
    else:
        print(f"You need to run your decode script for {missing_total} combinations.")
        print()
        print("Tip: If you have a script like 'bin/split_to_frames.py', run it for:")
        for seq_name in sorted(sequences.keys()):
            available_qps = sequences[seq_name]
            decoded_qps = set()
            for d in test_frames_real.iterdir():
                if d.is_dir() and d.name.startswith(seq_name):
                    match = re.search(r'QP(\d+)', d.name)
                    if match and list(d.glob("*.pt")):
                        decoded_qps.add(int(match.group(1)))
            
            missing = available_qps - decoded_qps
            if missing:
                for qp in sorted(missing):
                    for d in decoded_test.iterdir():
                        if d.name.startswith(f"{seq_name}_AI_QP{qp}_"):
                            print(f"  videos_test/decoded_test/{d.name}")
                            break

if __name__ == '__main__':
    main()
