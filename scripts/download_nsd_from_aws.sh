#!/usr/bin/env bash
set -euo pipefail

# Download NSD data from the public AWS bucket.
# Default behavior is dry-run for safety.
#
# Usage examples:
#   bash scripts/download_nsd_from_aws.sh --dest "/Users/hl/Documents/nsd" --mode minimal --run
#   bash scripts/download_nsd_from_aws.sh --dest "/Users/hl/Documents/nsd" --mode full --run

DEST=""
MODE="minimal"
DO_RUN="false"
INCLUDE_ALLSTIM="false"
SUBJECTS="subj01"
ALL_SUBJECTS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --run)
      DO_RUN="true"
      shift 1
      ;;
    --include-allstim)
      INCLUDE_ALLSTIM="true"
      shift 1
      ;;
    --subjects)
      SUBJECTS="$2"
      shift 2
      ;;
    --all-subjects)
      ALL_SUBJECTS="true"
      shift 1
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$DEST" ]]; then
  echo "Missing required --dest"
  exit 1
fi

if [[ "$MODE" != "minimal" && "$MODE" != "full" ]]; then
  echo "--mode must be one of: minimal, full"
  exit 1
fi

AWS=(python -m awscli s3)
SYNC_FLAGS=(--no-sign-request --exact-timestamps --size-only)
if [[ "$DO_RUN" != "true" ]]; then
  SYNC_FLAGS+=(--dryrun)
fi

mkdir -p "$DEST"

echo "Destination: $DEST"
echo "Mode: $MODE"
echo "Run: $DO_RUN"
echo "Include allstim: $INCLUDE_ALLSTIM"
echo "All subjects: $ALL_SUBJECTS"
echo "Subjects: $SUBJECTS"

# Small metadata + behavior files.
"${AWS[@]}" sync "${SYNC_FLAGS[@]}" \
  "s3://natural-scenes-dataset/nsddata/experiments/" \
  "$DEST/nsddata/experiments/"

"${AWS[@]}" sync "${SYNC_FLAGS[@]}" \
  "s3://natural-scenes-dataset/nsddata/information/" \
  "$DEST/nsddata/information/"

# Minimal stimuli needed for NSD indexing/mapping.
"${AWS[@]}" sync "${SYNC_FLAGS[@]}" \
  "s3://natural-scenes-dataset/nsddata_stimuli/stimuli/" \
  "$DEST/nsddata_stimuli/stimuli/" \
  --exclude "*" \
  --include "nsd/nsd_stimuli.hdf5"

if [[ "$INCLUDE_ALLSTIM" == "true" ]]; then
  "${AWS[@]}" sync "${SYNC_FLAGS[@]}" \
    "s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsdimagery/allstim/" \
    "$DEST/nsddata_stimuli/stimuli/nsdimagery/allstim/"
fi

if [[ "$MODE" == "minimal" ]]; then
  # Minimal pull: only 1.8mm HDF5 session betas, excluding synthetic and NIfTI.
  # This avoids allstim/png churn and huge side products you likely won't use in first pass.
  if [[ "$ALL_SUBJECTS" == "true" ]]; then
    "${AWS[@]}" sync "${SYNC_FLAGS[@]}" \
      "s3://natural-scenes-dataset/nsddata_betas/" \
      "$DEST/nsddata_betas/" \
      --exclude "*" \
      --include "ppdata/subj*/func1pt8mm/betas_fithrf/betas_session*.hdf5"
  else
    IFS=',' read -r -a SUBJ_ARR <<< "$SUBJECTS"
    for subj in "${SUBJ_ARR[@]}"; do
      subj_trimmed="$(echo "$subj" | tr -d '[:space:]')"
      "${AWS[@]}" sync "${SYNC_FLAGS[@]}" \
        "s3://natural-scenes-dataset/nsddata_betas/" \
        "$DEST/nsddata_betas/" \
        --exclude "*" \
        --include "ppdata/${subj_trimmed}/func1pt8mm/betas_fithrf/betas_session*.hdf5"
    done
  fi
else
  # Full beta sync (very large).
  "${AWS[@]}" sync "${SYNC_FLAGS[@]}" \
    "s3://natural-scenes-dataset/nsddata_betas/" \
    "$DEST/nsddata_betas/"
fi

echo
echo "Done."
if [[ "$DO_RUN" != "true" ]]; then
  echo "This was a dry-run. Re-run with --run to actually download."
fi
