#!/usr/bin/env bash
# Script for encoding dataset (vvenc, bez plików .cfg)

PROFILE=${1}   # ai | ra
QP=${2}
ALF=${3}       # 0/1
DB=${4}        # 0/1  (0=on, 1=off) -> LoopFilterDisable
SAO=${5}       # 0/1
FILE="${6}"
ODIR="${7}"

print_usage() {
  echo "USAGE:"
  echo "./bin/encode_data.sh PROFILE(Q: ai|ra) QP ALF(0|1) DB(0|1) SAO(0|1) FILE ODIR"
}

[[ -z "$PROFILE" || -z "$QP" || -z "$ALF" || -z "$DB" || -z "$SAO" || -z "$FILE" || -z "$ODIR" ]] && { print_usage; exit 1; }

# Wybór binarki – użyj tej, którą masz
ENCODER_APP="./vvenc/bin/release-static/vvencFFapp"

# Check if the input file directory exists instead of hardcoded "data"
INPUT_DIR=$(dirname "${FILE}")
[[ -d "${INPUT_DIR}" ]] || { echo "Make sure input directory ${INPUT_DIR} exists first"; exit 1; }
[[ -d "${ODIR}" ]] || { echo "Make sure to create ${ODIR} first"; exit 1; }

BASENAME_NOEXT="${FILE%.*}"
INFO_FILE=""
if [[ -f "${BASENAME_NOEXT}.yuv.info" ]]; then
    INFO_FILE="${BASENAME_NOEXT}.yuv.info"
elif [[ -f "${BASENAME_NOEXT}.y4m.info" ]]; then
    INFO_FILE="${BASENAME_NOEXT}.y4m.info"
else
    echo "Info file not found for ${FILE}"
    exit 1
fi

vval() {
  grep "$1" $INFO_FILE | head -n1 | \
    python -c "import sys; print(round(float(sys.stdin.read().split(':')[-1])))"
}

WIDTH=$(vval "Width")
HEIGHT=$(vval "Height")
FRAMERATE=$(vval "Frame rate")
END_FRAME=64

# Nazwy plików jak wcześniej
SUFFIX="${PROFILE}_QP${QP}_ALF${ALF}_DB${DB}_SAO${SAO}"
BASENAME="$(basename "$FILE")"
BASENAME="${BASENAME%.*}"
DESTINATION="${BASENAME}_${SUFFIX}.vvc"
RECON_FILE="${BASENAME}_${SUFFIX}_rec.yuv"
LOG_FILE="${BASENAME}_${SUFFIX}.log"

echo "Processing ${DESTINATION}..."

# Wspólne opcje
COMMON_OPTS=(
  --preset medium
  --InputFile="$FILE"
  --BitstreamFile="$ODIR/$DESTINATION"
  --ReconFile="$ODIR/$RECON_FILE"
  --FrameRate "$FRAMERATE"
  --FramesToBeEncoded "$END_FRAME"
  --SourceWidth "$WIDTH"
  --SourceHeight "$HEIGHT"
  --QP "$QP"
  --ALF "$ALF"
  --SAO "$SAO"
  --LoopFilterDisable "$DB"
  --InternalBitDepth 8
)

if [[ "$DB" -eq 1 ]]; then
  COMMON_OPTS+=( --EncDbOpt 0 )
fi

# Profil: AI vs RA
case "$PROFILE" in
  ai|AI)
    MODE_OPTS=( --IntraPeriod 1 --DecodingRefreshType 2 --GOPSize 1 )
    ;;
  ra|RA)
    # klasyczne RA 32
    MODE_OPTS=( --GOPSize 32 --IntraPeriod 32 --DecodingRefreshType 1 )
    ;;
  *)
    echo "Unknown PROFILE: $PROFILE (use ai or ra)"; exit 1;;
esac

# Uruchomienie
"$ENCODER_APP" "${COMMON_OPTS[@]}" "${MODE_OPTS[@]}" > "$ODIR/$LOG_FILE"
