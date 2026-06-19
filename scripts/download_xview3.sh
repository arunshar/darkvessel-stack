#!/bin/bash
# Parameterized xView3-SAR puller. Two modes:
#   (A) S3 sync  : XVIEW3_S3=s3://<bucket>/<prefix> with AWS creds in the env
#                  (run `aws configure` or export AWS_ACCESS_KEY_ID / _SECRET / _SESSION_TOKEN first).
#   (B) manifest : XVIEW3_MANIFEST=<file of pre-signed https URLs, one per line>
#                  (the iuu.xview.us per-scene tar.gz download list).
#
# Subset selection is by S3 prefix or by which URLs are in the manifest. Start
# with the validation or tiny subset to prove the pipeline before the ~2TB train set.
#
#   XVIEW3_S3=s3://bucket/validation XVIEW3_DEST=/scratch.global/arunshar/xview3/validation ./scripts/download_xview3.sh
#   XVIEW3_MANIFEST=val_urls.txt     XVIEW3_DEST=/scratch.global/arunshar/xview3/validation ./scripts/download_xview3.sh
set -euo pipefail

DEST="${XVIEW3_DEST:-/scratch.global/arunshar/xview3}"
mkdir -p "$DEST"
echo "[xview3-dl] dest=$DEST"

if [[ -n "${XVIEW3_S3:-}" ]]; then
  echo "[xview3-dl] S3 sync from $XVIEW3_S3"
  command -v aws >/dev/null || { echo "aws CLI not found on PATH"; exit 2; }
  aws sts get-caller-identity >/dev/null 2>&1 || { echo "ERROR: AWS creds not valid; run 'aws configure' or export AWS_ACCESS_KEY_ID/_SECRET_ACCESS_KEY first"; exit 3; }
  aws s3 sync "$XVIEW3_S3" "$DEST" --no-progress
elif [[ -n "${XVIEW3_MANIFEST:-}" ]]; then
  echo "[xview3-dl] downloading from manifest $XVIEW3_MANIFEST"
  [[ -f "$XVIEW3_MANIFEST" ]] || { echo "manifest not found: $XVIEW3_MANIFEST"; exit 2; }
  while IFS= read -r url; do
    [[ -z "$url" || "$url" == \#* ]] && continue
    fname="$(basename "${url%%\?*}")"
    echo "[xview3-dl]   $fname"
    curl -fSL --retry 3 -o "$DEST/$fname" "$url"
    case "$fname" in
      *.tar.gz|*.tgz) tar -xzf "$DEST/$fname" -C "$DEST" && rm -f "$DEST/$fname" ;;
      *.tar)          tar -xf  "$DEST/$fname" -C "$DEST" && rm -f "$DEST/$fname" ;;
    esac
  done < "$XVIEW3_MANIFEST"
else
  echo "ERROR: set XVIEW3_S3=s3://... (with AWS creds) or XVIEW3_MANIFEST=urls.txt"
  exit 1
fi

echo "[xview3-dl] done. scenes under $DEST:"
find "$DEST" -maxdepth 2 -type d | head -20
