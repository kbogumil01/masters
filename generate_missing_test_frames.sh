#!/bin/bash
# Generate missing test frames for BD-rate analysis
# Uses prepare_test_frames.py to convert YUV files to PT format

set -e

echo "========================================"
echo "Generating Missing Test Frames"
echo "========================================"
echo ""

# Use prepare_test_frames.py to process all decoded sequences
# It will automatically handle all sequences in decoded_test
echo "Running prepare_test_frames.py for all sequences..."
echo "This will process all QP values (28, 32, 37, 42, 47)"
echo ""

python3 bin/prepare_test_frames.py \
    videos_test/decoded_test \
    videos_test/test_dataset \
    videos_test/test_frames_REAL \
    --workers 8

echo ""
echo "========================================"
echo "All frames generated!"
echo "========================================"
echo ""
echo "Verify with:"
echo "  python3 bin/verify_bd_data.py"
echo ""
echo "Then run BD-rate analysis with:"
echo "  ./run_bd_analysis.sh"
