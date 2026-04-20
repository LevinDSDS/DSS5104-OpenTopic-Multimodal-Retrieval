#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Upload a SCAN-format precomp/ bundle to a Hugging Face *dataset* repo so that
# scripts/download_precomp.sh can pull it from a stable mirror.
#
# Design: one ``.tar.gz`` per dataset, so scripts/download_precomp.sh can
# consume it with its existing `unpack()` + `move_dir_into()` flow.
# The tarball unpacks to `f30k_precomp/` or `coco_precomp/` (matching SCAN's
# naming), with `{split}_ims.npy` + `{split}_caps.txt` inside.
#
# You run this **once** from a machine that already has the canonical BUTD
# features (e.g. VSE++ or SCAN).  After the upload, copy the URL the script
# prints into `scripts/download_precomp.sh` (one sed command -- the script
# tells you exactly which).
#
# Prereqs:
#   pip install huggingface_hub
#   hf auth login                   # paste a "write"-scoped token
#   # (on huggingface_hub < 1.0 use `huggingface-cli login` instead)
#
# Usage:
#   bash scripts/upload_to_hf.sh flickr30k /data/scan/f30k_precomp \\
#                                [<HF_REPO_ID>]
#   bash scripts/upload_to_hf.sh coco      /data/scan/coco_precomp
#
# If HF_REPO_ID is not given, we default to
#   {your-HF-username}/aitr-scan-precomp
# (a single repo for both f30k and coco tarballs).
# -----------------------------------------------------------------------------
set -euo pipefail

DATASET="${1:?usage: $0 <flickr30k|coco> <path/to/precomp> [<HF_REPO_ID>]}"
SRC_DIR="${2:?usage: $0 <flickr30k|coco> <path/to/precomp> [<HF_REPO_ID>]}"
REPO_ID="${3:-}"

if [[ -z "${REPO_ID}" ]]; then
  # huggingface_hub >= 1.0 renamed the CLI from `huggingface-cli` to `hf`.
  # We support both and degrade gracefully.
  if command -v hf >/dev/null 2>&1; then
      HF_USER="$(hf auth whoami --format json 2>/dev/null | \
                 python3 -c 'import json,sys; \
                 print(json.load(sys.stdin).get("name",""))' \
                 2>/dev/null || true)"
  else
      HF_USER="$(huggingface-cli whoami 2>/dev/null | head -n1 || true)"
  fi
  if [[ -z "${HF_USER:-}" ]]; then
    echo "[!!] could not determine HF username. Run 'hf auth login' first"
    echo "     (or 'huggingface-cli login' on older huggingface_hub)."
    exit 2
  fi
  REPO_ID="${HF_USER}/aitr-scan-precomp"
fi

case "${DATASET}" in
  flickr30k) SPLITS=(train dev test);            MARKER=f30k_precomp ;;
  coco)      SPLITS=(train dev test testall);    MARKER=coco_precomp ;;
  *) echo "unknown dataset '${DATASET}' (expected flickr30k|coco)"; exit 2 ;;
esac

echo "[*] dataset       : ${DATASET}"
echo "[*] src precomp   : ${SRC_DIR}"
echo "[*] target HF repo: ${REPO_ID}"
echo "[*] archive marker: ${MARKER}/"

# Verify source is AITR-compatible *before* paying upload bandwidth.
python -m data.verify_precomp --precomp "${SRC_DIR}" --splits "${SPLITS[@]}"

# Stage a clean "${MARKER}/" directory next to the source, then tar.gz.
STAGE=$(mktemp -d)
trap "rm -rf ${STAGE}" EXIT
mkdir -p "${STAGE}/${MARKER}"
for split in "${SPLITS[@]}"; do
  for ext in _ims.npy _caps.txt; do
    cp -v "${SRC_DIR}/${split}${ext}" "${STAGE}/${MARKER}/${split}${ext}"
  done
done

ARCHIVE="${STAGE}/${MARKER}.tar.gz"
echo "[*] compressing tarball -> ${ARCHIVE}"
tar -C "${STAGE}" -czvf "${ARCHIVE}" "${MARKER}"
ls -lh "${ARCHIVE}"

# Create the remote dataset repo (idempotent).
python - <<PYEOF
from huggingface_hub import create_repo
create_repo("${REPO_ID}", repo_type="dataset", exist_ok=True, private=False)
print("[ok] repo ${REPO_ID} created / already exists")
PYEOF

# Upload the single .tar.gz. The CLI handles LFS automatically for large
# files and chunks them transparently. We prefer the new `hf` binary but
# fall back to the legacy `huggingface-cli` if only the older package is
# installed.
echo "[.] uploading ${MARKER}.tar.gz"
if command -v hf >/dev/null 2>&1; then
    hf upload "${REPO_ID}" "${ARCHIVE}" "${MARKER}.tar.gz" \
        --repo-type dataset \
        --commit-message "upload ${MARKER}.tar.gz (AITR Tables 1-2 BUTD features)"
else
    huggingface-cli upload "${REPO_ID}" "${ARCHIVE}" "${MARKER}.tar.gz" \
        --repo-type dataset \
        --commit-message "upload ${MARKER}.tar.gz (AITR Tables 1-2 BUTD features)"
fi

CANONICAL_URL="https://huggingface.co/datasets/${REPO_ID}/resolve/main/${MARKER}.tar.gz"
TAG_UP=${DATASET^^}

echo
echo "[OK] upload finished. Canonical tarball URL:"
echo
echo "     ${CANONICAL_URL}"
echo
echo "Now wire it into scripts/download_precomp.sh as the *primary* mirror"
echo "for ${DATASET}:"
echo
echo "  # one-liner (requires GNU sed; on macOS use \`gsed\`):"
echo "  sed -i.bak \"s|__HF_MIRROR_${TAG_UP}__|${CANONICAL_URL}|g\" \\"
echo "      scripts/download_precomp.sh"
echo
echo "  # or just export it at runtime (no file edits needed):"
echo "  export AITR_PRECOMP_URL_${TAG_UP}='${CANONICAL_URL}'"
echo "  bash scripts/download_precomp.sh ${DATASET} \$DATA_ROOT"
