#!/usr/bin/env bash
# Download the public SCAN (Lee et al., 2018) precomputed Faster R-CNN
# bottom-up features used by *every* published baseline on Flickr30K
# and MS-COCO.  We do NOT redistribute the data -- this script only
# wraps a list of currently-known-good public HTTPS mirrors.
#
# Expected output layout after running:
#   $DATA_ROOT/flickr30k/precomp/{train,dev,test}_ims.npy      # (N,36,2048) float32
#   $DATA_ROOT/flickr30k/precomp/{train,dev,test}_caps.txt     # 5 caps / image
#   $DATA_ROOT/coco/precomp/{train,dev,testall}_ims.npy
#   $DATA_ROOT/coco/precomp/{train,dev,testall}_caps.txt
#
# Usage:
#   bash scripts/download_precomp.sh  flickr30k  /path/to/DATA_ROOT
#   bash scripts/download_precomp.sh  coco       /path/to/DATA_ROOT
#
# After download we also run `python -m data.verify_precomp` on the
# target directory to confirm the shapes / caption counts match the
# format AITR's PrecompDataset expects.  If the layout is wrong the
# script exits with a non-zero status and prints clear remediation
# instructions instead of silently "succeeding".

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <flickr30k|coco> <DATA_ROOT>"
  exit 1
fi

DATASET=$1
DATA_ROOT=$(realpath "$2")
mkdir -p "${DATA_ROOT}"

# ----------------------------------------------------------- helpers
get() {
    local url=$1 out=$2
    echo "[+] trying ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --retry 3 --retry-delay 4 -L -o "${out}" "${url}"
    else
        wget -O "${out}" "${url}"
    fi
}

unpack() {
    # unpack $1 ($2 = "tar" or "zip") into the current dir.
    local archive=$1 kind=$2
    case "${kind}" in
        tar) tar -xvf "${archive}" ;;
        zip) unzip -oq "${archive}" ;;
        *)   echo "unknown archive kind ${kind}"; return 1 ;;
    esac
    rm -f "${archive}"
}

move_dir_into() {
    # Find the precomp-like sub-directory inside $1 and move it to $2.
    # SCAN bundles typically unpack to one of: data/f30k_precomp/,
    # f30k_precomp/, data/coco_precomp/, coco_precomp/.  We just search
    # for "*${marker}*" anywhere under $1.
    local root=$1 marker=$2 dest=$3
    local src
    src=$(find "${root}" -type d -name "${marker}" -print -quit)
    if [[ -z "${src}" ]]; then
        return 1
    fi
    mkdir -p "$(dirname "${dest}")"
    rm -rf "${dest}"
    mv "${src}" "${dest}"
}

verify() {
    local precomp_dir=$1
    echo "[+] verifying layout at ${precomp_dir}"
    python -m data.verify_precomp --precomp "${precomp_dir}"
}

# ----------------------------------------------- dataset-specific tables
# NOTE (2025-11): GitHub-mirrored copies of the SCAN / VSE++ feature
# bundle surface / disappear every few months.  We hard-code a small list
# of known-good mirrors; the script walks the list and uses the first
# one that responds.  If all mirrors fail, a clear error is printed so
# the user can supply their own tarball via $AITR_PRECOMP_URL_<DATASET>.
#
# Flickr30K is hosted on HF at levinnus/aitr-scan-precomp (primary).
# COCO mirror: set via upload_to_hf.sh after hosting on your HF account.
# Until configured, the script uses the Kaggle fallback mirror.
case "${DATASET}" in
  flickr30k|f30k|flickr)
      MIRRORS=(
          "${AITR_PRECOMP_URL_FLICKR30K:-}"
          "https://huggingface.co/datasets/levinnus/aitr-scan-precomp/resolve/main/f30k_precomp.tar.gz"
          "https://www.kaggle.com/api/v1/datasets/download/kuanghueilee/scan-features"
          "http://www.cs.toronto.edu/~faghri/vsepp/data.tar"
      )
      ARCHIVE_KINDS=(tar.gz tar.gz zip tar)
      MARKER="f30k_precomp"
      DEST="${DATA_ROOT}/flickr30k/precomp"
      ;;
  coco|mscoco)
      MIRRORS=(
          "${AITR_PRECOMP_URL_COCO:-}"
          "__HF_MIRROR_COCO__"
          "https://www.kaggle.com/api/v1/datasets/download/kuanghueilee/scan-features"
          "http://www.cs.toronto.edu/~faghri/vsepp/data.tar"
      )
      ARCHIVE_KINDS=(tar.gz tar.gz zip tar)
      MARKER="coco_precomp"
      DEST="${DATA_ROOT}/coco/precomp"
      ;;
  *)
      echo "unknown dataset: ${DATASET} (expect: flickr30k | coco)"
      exit 2
      ;;
esac

# ----------------------------------------------- try each mirror in turn
TMP=$(mktemp -d)
trap "rm -rf ${TMP}" EXIT
cd "${TMP}"

SUCCESS=0
for i in "${!MIRRORS[@]}"; do
    url="${MIRRORS[$i]}"
    [[ -z "${url}" ]] && continue
    # Skip unconfigured mirror slots.
    if [[ "${url}" == __HF_MIRROR_*__ ]]; then
        continue
    fi
    kind="${ARCHIVE_KINDS[$i]}"
    archive="bundle.${kind}"
    if get "${url}" "${archive}"; then
        if unpack "${archive}" "${kind%%.*}" && \
           move_dir_into "${TMP}" "${MARKER}" "${DEST}"; then
            echo "[OK] ${DATASET} precomp -> ${DEST}"
            SUCCESS=1
            break
        else
            echo "[!] ${url} downloaded but layout did not contain ${MARKER};"
            echo "    continuing to next mirror."
            rm -rf "${TMP}"/*
        fi
    fi
done

if [[ ${SUCCESS} -ne 1 ]]; then
    cat <<EOF >&2

ERROR: could not fetch SCAN precomp features for ${DATASET} from any
mirror.  The community-maintained URLs rotate every ~6 months.

Remediation (in order of preference):

  1. Download the bundle manually, e.g. via Kaggle:
       https://www.kaggle.com/datasets/kuanghueilee/scan-features
     Unzip so that you end up with a directory named exactly
     '${MARKER}/' containing '{split}_ims.npy' + '{split}_caps.txt';
     then move it to '${DEST}'.

  2. Export a direct URL and re-run:
       export AITR_PRECOMP_URL_${DATASET^^}=https://your.mirror/bundle.tar
       bash scripts/download_precomp.sh ${DATASET} ${DATA_ROOT}

  3. If you already have raw Flickr30K / MS-COCO JPGs, you can run the
     torchvision extractor as an alternative pipeline (note: uses
     different backbone pre-training from Tables 1-2):
       python -m data.extract_features \\
           --images /path/to/jpgs --splits data/splits/${DATASET}.json \\
           --out ${DEST} --backend torchvision
EOF
    exit 3
fi

# ------------------------------------------------------------- verify
cd - >/dev/null
verify "${DEST}" || {
    echo "ERROR: the downloaded bundle does not match the expected AITR"
    echo "       precomp layout.  See messages above.  Aborting." >&2
    exit 4
}
