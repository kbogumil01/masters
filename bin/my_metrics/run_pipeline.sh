#!/bin/bash
#
# Full pipeline for calculating BD-rate metrics (matching Piotr's approach)
#
# Usage:
#   ./bin/my_metrics/run_pipeline.sh <checkpoint> <config>
#
# Example:
#   ./bin/my_metrics/run_pipeline.sh checkpoints/epoch=787.ckpt experiments/enhancer/baseline_0501.yaml
#

set -e

CHECKPOINT=$1
CONFIG=$2

if [ -z "$CHECKPOINT" ] || [ -z "$CONFIG" ]; then
    echo "Usage: $0 <checkpoint> <config>"
    echo "Example: $0 checkpoints/epoch=787.ckpt experiments/enhancer/baseline_0501.yaml"
    exit 1
fi

echo "=============================================="
echo "BD-RATE METRICS PIPELINE"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo ""

# Directories
INPUT_DIR="data/videos_test/test_frames_REAL"
ORIG_DIR="data/videos_test/test_dataset"
ENCODER_LOGS="data/videos_test/encoded_test"

ENHANCED_PT="data/videos_test/enhanced_pt"
ENHANCED_YUV444="data/videos_test/enhanced_yuv444"
ENHANCED_YUV420="data/videos_test/enhanced_yuv420"
DECODED_YUV420="data/videos_test/decoded_yuv420"
METRICS_DIR="data/videos_test/metrics"
OUTPUT_REPORT="results/bd_report.csv"

echo "Step 1/6: Running inference on test frames..."
python -m bin.my_metrics.run_inference \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$ENHANCED_PT"

echo ""
echo "Step 2/6: Converting enhanced frames to YUV 4:4:4..."
python -m bin.my_metrics.pt_to_yuv \
    --input-dir "$ENHANCED_PT" \
    --output-dir "$ENHANCED_YUV444"

echo ""
echo "Step 3/6: Converting enhanced YUV 4:4:4 to YUV 4:2:0..."
python -m bin.my_metrics.convert_to_420 \
    --input-dir "$ENHANCED_YUV444" \
    --output-dir "$ENHANCED_YUV420"

echo ""
echo "Step 4/6: Converting decoded frames to YUV 4:2:0..."
python -m bin.my_metrics.convert_decoded_to_420 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$DECODED_YUV420"

echo ""
echo "Step 5/6: Calculating PSNR/SSIM metrics..."
python -m bin.my_metrics.calculate_metrics \
    --enhanced-dir "$ENHANCED_YUV420" \
    --decoded-dir "$DECODED_YUV420" \
    --original-dir "$ORIG_DIR" \
    --output-dir "$METRICS_DIR"

echo ""
echo "Step 6/6: Generating BD-rate report..."
python -m bin.my_metrics.generate_report \
    --metrics-dir "$METRICS_DIR" \
    --encoder-logs "$ENCODER_LOGS" \
    --output "$OUTPUT_REPORT"

echo ""
echo "=============================================="
echo "PIPELINE COMPLETE!"
echo "=============================================="
echo "Results saved to: $OUTPUT_REPORT"
