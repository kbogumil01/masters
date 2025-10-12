#! /bin/bash
# bin/decode_data.sh
# Użycie:
#   bin/decode_data.sh <BITSTREAM.vvc> <OUTDIR_ROOT> [OPTIONS]
# 
# Opcje środowiskowe:
#   SKIP_FUSED_MAPS=1      - nie generuj fused maps (użyj separate maps)
#   GENERATE_PNG_PREVIEWS=1 - generuj podglądy PNG (domyślnie wyłączone)
#   KEEP_ORIGINAL_FILES=1  - zachowaj oryginalne .csv/.bin
#   FORCE_RESOLUTION=WxH   - wymuś rozdzielczość (np. 1280x720)

set -euo pipefail

BITSTREAM="${1:-}"
OUTROOT="${2:-}"

if [[ -z "$BITSTREAM" || -z "$OUTROOT" ]]; then
  echo "USAGE: bin/decode_data.sh <BITSTREAM.vvc> <OUTDIR_ROOT> [OPTIONS]"
  echo ""
  echo "Environment options:"
  echo "  SKIP_FUSED_MAPS=1       - generate separate maps instead of fused"
  echo "  GENERATE_PNG_PREVIEWS=1 - generate PNG previews (disabled by default)"
  echo "  KEEP_ORIGINAL_FILES=1   - keep original .csv/.bin files"
  echo "  FORCE_RESOLUTION=WxH    - force resolution (e.g., 1280x720)"
  echo ""
  echo "Examples:"
  echo "  bin/decode_data.sh video.vvc outputs/"
  echo "  FORCE_RESOLUTION=1280x720 bin/decode_data.sh video.vvc outputs/"
  echo "  GENERATE_PNG_PREVIEWS=1 bin/decode_data.sh video.vvc outputs/"
  exit 1
fi

# --- ścieżki absolutne ---
BITSTREAM_ABS="$(readlink -f "$BITSTREAM")"

# katalog projektu (…/new_PC)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# VTM (możesz nadpisać: VTM_DIR=/twoja/sciezka bin/decode_data.sh …)
VTM_DIR_DEFAULT="${PROJECT_ROOT}/VVCSoftware_VTM"
VTM_DIR="${VTM_DIR:-$VTM_DIR_DEFAULT}"
DECODER="${VTM_DIR}/bin/DecoderAppStatic"
[[ -x "$DECODER" ]] || { echo "Decoder not found/executable: $DECODER"; exit 1; }

# Python (spróbuj użyć .venv jeśli jest, inaczej systemowego)
if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

MAPS_SCRIPT="${PROJECT_ROOT}/bin/build_dequant_maps.py"
BOUNDARY_SCRIPT="${PROJECT_ROOT}/bin/build_boundary_maps.py"
FUSE_SCRIPT="${PROJECT_ROOT}/bin/fuse_maps.py"
[[ -f "$MAPS_SCRIPT" ]] || { echo "build_dequant_maps.py not found at: $MAPS_SCRIPT"; exit 1; }
[[ -f "$BOUNDARY_SCRIPT" ]] || { echo "build_boundary_maps.py not found at: $BOUNDARY_SCRIPT"; exit 1; }
[[ -f "$FUSE_SCRIPT" ]] || { echo "fuse_maps.py not found at: $FUSE_SCRIPT"; exit 1; }

# --- katalogi wyjściowe ---
mkdir -p "$OUTROOT"
BASE="$(basename "$BITSTREAM_ABS" .vvc)"   # np. husky_cif_AI_QP32_ALF0_DB0_SAO0
OUTDIR="${OUTROOT}/${BASE}"
mkdir -p "$OUTDIR"

echo "Decoding ${BITSTREAM_ABS}"
echo " -> ${OUTDIR}"

# Dequanty lądują w CWD, więc wchodzimy do OUTDIR
pushd "$OUTDIR" >/dev/null

# --- dekodowanie + tracing ---
# UWAGA: 8-bitowy recon, bo wejście/enkodowanie było 8-bitowe
"$DECODER" \
  --BitstreamFile="$BITSTREAM_ABS" \
  --ReconFile="recon.yuv" \
  --OutputBitDepth=8 \
  --OutputBitDepthC=8 \
  --TraceFile="block_stats.csv" \
  --TraceRule="D_BLOCK_STATISTICS_ALL:poc>=0" \
  > "decode.log" 2>&1

# --- Generate neural network maps directly from dequant data ---
if compgen -G "dequant_poc_*.bin" > /dev/null; then
  echo "Building neural network maps from dequant data..."
  
  # Create maps subdirectory
  mkdir -p "neural_maps"
  
  # Build command arguments
  MAP_ARGS=(. --outdir "neural_maps" --with-dc --debug)
  
  # Add optional arguments based on environment
  if [[ "${GENERATE_PNG_PREVIEWS:-}" == "1" ]]; then
    MAP_ARGS+=(--save-png)
  fi
  
  if [[ "${KEEP_ORIGINAL_FILES:-}" != "1" ]]; then
    MAP_ARGS+=(--cleanup)
  fi
  
  # Force resolution if specified
  if [[ -n "${FORCE_RESOLUTION:-}" ]]; then
    if [[ "$FORCE_RESOLUTION" =~ ^([0-9]+)x([0-9]+)$ ]]; then
      MAP_ARGS+=(--width "${BASH_REMATCH[1]}" --height "${BASH_REMATCH[2]}")
      echo "  Forced resolution: $FORCE_RESOLUTION"
    else
      echo "WARN: Invalid FORCE_RESOLUTION format: $FORCE_RESOLUTION (expected WxH)"
    fi
  fi
  
  # Build maps directly
  "$PYTHON" "$MAPS_SCRIPT" "${MAP_ARGS[@]}" > "maps.log" 2>&1 || {
    echo "ERROR: Neural map generation failed, see ${OUTDIR}/maps.log"
    exit 1
  }
else
  echo "WARN: no dequant_poc_*.bin files found to process."
fi

# --- Generate boundary maps from block statistics ---
if [[ -f "block_stats.csv" ]]; then
  if [[ "${SKIP_FUSED_MAPS:-}" == "1" ]]; then
    echo "Generating separate boundary maps (SKIP_FUSED_MAPS=1)..."
    
    # Create boundary maps subdirectory
    mkdir -p "boundary_maps"
    
    # Build boundary command arguments
    BOUNDARY_ARGS=(. --outdir "boundary_maps" --debug)
    
    # Add PNG previews if explicitly enabled
    if [[ "${GENERATE_PNG_PREVIEWS:-}" == "1" ]]; then
      BOUNDARY_ARGS+=(--save-png)
    fi
    
    # Force resolution if specified
    if [[ -n "${FORCE_RESOLUTION:-}" ]]; then
      if [[ "$FORCE_RESOLUTION" =~ ^([0-9]+)x([0-9]+)$ ]]; then
        BOUNDARY_ARGS+=(--width "${BASH_REMATCH[1]}" --height "${BASH_REMATCH[2]}")
      fi
    fi
    
    # Generate boundary maps
    "$PYTHON" "$BOUNDARY_SCRIPT" "${BOUNDARY_ARGS[@]}" > "boundary.log" 2>&1 || {
      echo "WARN: boundary map generation failed, see ${OUTDIR}/boundary.log"
    }
  else
    echo "Building boundary maps for fusion..."
    
    # Create boundary maps subdirectory (temporary for fused maps)
    mkdir -p "boundary_maps"
    
    # Build boundary command arguments (no PNG unless explicitly requested)
    BOUNDARY_ARGS=(. --outdir "boundary_maps" --debug)
    
    # Force resolution if specified
    if [[ -n "${FORCE_RESOLUTION:-}" ]]; then
      if [[ "$FORCE_RESOLUTION" =~ ^([0-9]+)x([0-9]+)$ ]]; then
        BOUNDARY_ARGS+=(--width "${BASH_REMATCH[1]}" --height "${BASH_REMATCH[2]}")
      fi
    fi
    
    # Generate boundary maps (for fusion)
    "$PYTHON" "$BOUNDARY_SCRIPT" "${BOUNDARY_ARGS[@]}" > "boundary.log" 2>&1 || {
      echo "ERROR: boundary map generation failed, see ${OUTDIR}/boundary.log"
      exit 1
    }
  fi
else
  echo "WARN: block_stats.csv not found, skipping boundary maps."
fi

# --- Generate fused maps (default behavior) ---
if [[ "${SKIP_FUSED_MAPS:-}" != "1" && -d "neural_maps" && -d "boundary_maps" && -f "block_stats.csv" ]]; then
  echo "Creating fused maps from neural and boundary data..."
  
  # Build fuse command arguments
  FUSE_ARGS=(. --outdir "fused_maps" --debug)
  
  # Add visualization if PNG previews are enabled
  if [[ "${GENERATE_PNG_PREVIEWS:-}" == "1" ]]; then
    FUSE_ARGS+=(--visualize)
  fi
  
  # Generate fused maps
  "$PYTHON" "$FUSE_SCRIPT" "${FUSE_ARGS[@]}" > "fuse.log" 2>&1 || {
    echo "ERROR: fused map generation failed, see ${OUTDIR}/fuse.log"
    exit 1
  }
  
  # Clean up intermediate maps unless explicitly keeping them
  if [[ "${KEEP_ORIGINAL_FILES:-}" != "1" ]]; then
    if [[ -d "neural_maps" ]]; then
      echo "Cleaning up intermediate neural maps..."
      rm -rf "neural_maps"
    fi
    if [[ -d "boundary_maps" ]]; then
      echo "Cleaning up intermediate boundary maps..."
      rm -rf "boundary_maps"
    fi
  fi
else
  if [[ "${SKIP_FUSED_MAPS:-}" == "1" ]]; then
    echo "Skipping fused maps generation (SKIP_FUSED_MAPS=1)"
  else
    echo "WARN: Cannot generate fused maps - missing required components"
  fi
fi

popd >/dev/null

echo "OK:"
echo " - $OUTDIR/recon.yuv"
echo " - $OUTDIR/block_stats.csv"

if [[ "${SKIP_FUSED_MAPS:-}" == "1" ]]; then
  echo " - $OUTDIR/neural_maps/dequant_maps_poc*.npz (neural network data)"
  echo " - $OUTDIR/boundary_maps/boundary_maps_poc*.npz (block boundary data)"
  if [[ "${GENERATE_PNG_PREVIEWS:-}" == "1" ]]; then
    echo " - $OUTDIR/neural_maps/*_energy.png, *_nz.png, *_dc.png (previews)"
    echo " - $OUTDIR/boundary_maps/*_boundary.png (previews)"
  fi
  echo " - logi: $OUTDIR/decode.log, $OUTDIR/maps.log, $OUTDIR/boundary.log"
else
  echo " - $OUTDIR/fused_maps/fused_maps_poc*.npz (13-channel enhanced data)"
  if [[ "${GENERATE_PNG_PREVIEWS:-}" == "1" ]]; then
    echo " - $OUTDIR/fused_maps/*_visualization.png (comprehensive previews)"
  fi
  echo " - logi: $OUTDIR/decode.log, $OUTDIR/maps.log, $OUTDIR/boundary.log, $OUTDIR/fuse.log"
fi
 