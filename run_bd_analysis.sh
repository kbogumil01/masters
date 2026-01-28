#!/bin/bash
# Quick script to run BD-rate analysis on trained models

set -e

echo "=========================================="
echo "BD-RATE ANALYSIS FOR VIDEO ENHANCEMENT"
echo "=========================================="
echo ""

# Model checkpoint to analyze
CHECKPOINT="${1:-checkpoints_v2/epoch=998.ckpt}"
CONFIG="${2:-experiments/enhancer/dense_high_QP.yaml}"

echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo ""
echo "⏱️  This will take 5-10 minutes..."
echo ""

# Create results directory
mkdir -p results/bd_rate_analysis

# Run analysis (without timeout - let it finish naturally)
echo "Running BD-rate calculation..."
python3 bin/calculate_bd_rate.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --test-root "videos_test/test_frames_REAL" \
    --orig-root "videos_test/test_orig_frames_pt" \
    --output "results/bd_rate_analysis/bd_report_$(basename $CHECKPOINT .ckpt).txt" \
    --device cuda

echo ""
echo "=========================================="
echo "✅ Analysis complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  results/bd_rate_analysis/bd_report_$(basename $CHECKPOINT .ckpt).txt"
echo "  results/bd_rate_analysis/bd_report_$(basename $CHECKPOINT .ckpt).csv"
