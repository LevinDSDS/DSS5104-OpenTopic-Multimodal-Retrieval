#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Download canonical SCAN-format Faster R-CNN region features from Kaggle
# (dataset: kuanghueilee/scan-features).
#
# Auth (pick one):
#   export KAGGLE_API_TOKEN='KGAT_xxxxxxxx...'   # recommended (Kaggle 新 token)
#   # legacy:
#   echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
#   chmod 600 ~/.kaggle/kaggle.json
#
# Usage:
#   bash scripts/download_scan_features_kaggle.sh flickr30k /path/to/out
#   bash scripts/download_scan_features_kaggle.sh coco      /path/to/out
#
# Writes:  ${OUT}/f30k_precomp/  or  ${OUT}/coco_precomp/
# with      {train,dev,test}_ims.npy  +  _caps.txt  (+ coco: testall_*)
# -----------------------------------------------------------------------------
set -euo pipefail

DATASET="${1:?usage: $0 <flickr30k|coco> <output_dir>}"
OUT="${2:?usage: $0 <flickr30k|coco> <output_dir>}"
DS="kuanghueilee/scan-features"

if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
  export KAGGLE_API_TOKEN
elif [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  :
else
  echo "[!!] Set KAGGLE_API_TOKEN or place ~/.kaggle/kaggle.json"
  exit 2
fi

case "${DATASET}" in
  flickr30k|f30k)
    SUB="f30k_precomp"
    FILES=(
      train_caps.txt train_ids.txt train_ims.npy
      dev_caps.txt dev_ids.txt dev_ims.npy
      test_caps.txt test_ids.txt test_ims.npy
    )
    ;;
  coco)
    SUB="coco_precomp"
    FILES=(
      train_caps.txt train_ids.txt train_ims.npy
      dev_caps.txt dev_ids.txt dev_ims.npy
      test_caps.txt test_ids.txt test_ims.npy
      testall_caps.txt testall_ids.txt testall_ims.npy
    )
    ;;
  *)
    echo "unknown dataset '${DATASET}' (use flickr30k or coco)"
    exit 2
    ;;
esac

MARKER="${SUB}"
BUILD="${OUT}/.${MARKER}_build_$$"
FINAL="${OUT}/${MARKER}"

mkdir -p "${BUILD}"
trap 'rm -rf "${BUILD}"' EXIT

echo "[*] Kaggle dataset : ${DS}"
echo "[*] remote subdir  : data/data/${SUB}/"
echo "[*] staging build  : ${BUILD}"
echo "[*] final precomp  : ${FINAL}"

# Prefer user-local pip kaggle if present
export PATH="${HOME}/Library/Python/3.12/bin:${PATH}"

for f in "${FILES[@]}"; do
  REMOTE="data/data/${SUB}/${f}"
  echo "[.] kaggle datasets download -f ${REMOTE}"
  kaggle datasets download -f "${REMOTE}" -d "${DS}" -p "${BUILD}"
done

shopt -s nullglob
for z in "${BUILD}"/*.zip; do
  echo "[.] unzip $(basename "${z}")"
  unzip -o -q "${z}" -d "${BUILD}"
  rm -f "${z}"
done
shopt -u nullglob

# Flatten: Kaggle may unpack into nested paths (macOS bash 3.2: use find).
while IFS= read -r -d '' npy; do
  base="$(basename "${npy}")"
  case "${base}" in
    train_ims.npy|dev_ims.npy|test_ims.npy|testall_ims.npy)
      mv -f "${npy}" "${BUILD}/${base}"
      ;;
  esac
done < <(find "${BUILD}" -name '*_ims.npy' -print0 2>/dev/null)

while IFS= read -r -d '' txt; do
  base="$(basename "${txt}")"
  case "${base}" in
    train_caps.txt|train_ids.txt|dev_caps.txt|dev_ids.txt|test_caps.txt|test_ids.txt|testall_caps.txt|testall_ids.txt|*_tags.txt)
      mv -f "${txt}" "${BUILD}/${base}"
      ;;
  esac
done < <(find "${BUILD}" \( -name '*.txt' -o -name '*.TXT' \) -print0 2>/dev/null)

# Move loose files in BUILD root (unzip sometimes drops here already).
for f in "${FILES[@]}"; do
  if [[ -f "${BUILD}/${f}" ]]; then
    :
  elif [[ -f "${BUILD}/data/data/${SUB}/${f}" ]]; then
    mv -f "${BUILD}/data/data/${SUB}/${f}" "${BUILD}/${f}"
  fi
done

mkdir -p "${FINAL}"
for f in "${FILES[@]}"; do
  if [[ ! -f "${BUILD}/${f}" ]]; then
    echo "[!!] missing after download: ${f}"
    find "${BUILD}" -maxdepth 4 -name "${f}" -print || true
    exit 3
  fi
  mv -f "${BUILD}/${f}" "${FINAL}/${f}"
done

# SCAN convention: dev/test images are repeated 5x (one per caption).
# AITR expects N unique images + 5*N captions. Convert in-place.
echo "[.] converting SCAN format -> AITR format (deduplicate dev/test images)"
python3 - "${FINAL}" <<'PYEOF'
import numpy as np, os, sys, shutil
DATA = sys.argv[1]
for split in ("dev", "test", "testall"):
    ims_path = os.path.join(DATA, f"{split}_ims.npy")
    cap_path = os.path.join(DATA, f"{split}_caps.txt")
    if not os.path.isfile(ims_path):
        continue
    ims = np.load(ims_path)
    with open(cap_path) as f:
        caps = [l.strip() for l in f if l.strip()]
    if len(caps) == ims.shape[0] and ims.shape[0] % 5 == 0:
        n_img = ims.shape[0] // 5
        if np.array_equal(ims[0], ims[1]):
            shutil.copy2(ims_path, ims_path + ".scan_orig")
            np.save(ims_path, ims[::5].copy())
            print(f"  {split}: {ims.shape[0]} -> {n_img} images (dedup)")
        else:
            print(f"  {split}: already in AITR format ({ims.shape[0]} imgs)")
    else:
        ratio = len(caps) / max(ims.shape[0], 1)
        if abs(ratio - 5.0) < 0.01:
            print(f"  {split}: already in AITR format ({ims.shape[0]} imgs, {len(caps)} caps)")
        else:
            print(f"  [WARN] {split}: unexpected shape ims={ims.shape[0]} caps={len(caps)}")
PYEOF

echo "[OK] precomp ready at ${FINAL}"
echo "     python -m data.verify_precomp --precomp ${FINAL} --splits ..."
