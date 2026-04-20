# Cross-Scale Semantic Alignment for Image–Text Retrieval

[![CI](https://github.com/LevinDSDS/Assignment_Multimodal_Retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/LevinDSDS/Assignment_Multimodal_Retrieval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 

---

## 0. 30-second quick start

No GPU, no downloads — verify the IDF / IDE / CSA / WMF / loss pipeline
end-to-end on a synthetic `precomp/` in under a minute:

```bash
git clone https://github.com/LevinDSDS/Assignment_Multimodal_Retrieval.git aitr
cd aitr
python -m pip install -r requirements.txt
python data/make_toy_precomp.py --out data/toy_precomp/flickr30k \
    --n_train 64 --n_dev 16 --n_test 16 --include_testall
pytest tests/ -q                                           # unit tests
python train.py --config configs/toy_bigru.yaml            # smoke train
```

The real Flickr30K / MS-COCO numbers in Tables 1–2 need the SCAN
pre-computed features — see §4.

---

## 1. Highlights

- **Single-file model** (`aitr/model.py`) that wires together
  - `IntraDimFilter` (IDF): channel selector per semantic prototype
  - `InterDimExpander` (IDE): binary-union mask between modalities
  - `CrossScaleAggregator` (CSA): position + co-occurrence subseq.
  - `WeakMatchFilter` (WMF): statistically calibrated thresholding
- Backbone-agnostic: runs out-of-the-box with **Bi-GRU**, **BERT**,
  and as a re-ranker on top of **CLIP / BLIP** features.
- Training and evaluation in **<200 lines** thanks to a thin
  trainer loop — no PyTorch-Lightning dependency.
- YAML configs (`configs/*.yaml`) with one entry point.

## 2. Repository layout

```
aitr/
├── aitr/
│   ├── __init__.py
│   ├── encoders.py         # image (RoI) + text (Bi-GRU / BERT) encoders
│   ├── prototypes.py       # semantic-prototype bank (V & T)
│   ├── dim_filter.py       # IntraDimFilter (IDF) + InterDimExpander (IDE)
│   ├── cross_scale.py      # CrossScaleAggregator (CSA) + IoU pairing
│   ├── weak_match.py       # WeakMatchFilter (WMF)
│   ├── similarity.py       # fragment- + instance-level similarities
│   ├── loss.py             # hardest-negative triplet ranking loss
│   ├── model.py            # AITR top-level module
│   └── utils.py            # l1/l2-norm, masked softmax, RNG control
├── data/
│   ├── __init__.py
│   ├── dataset.py          # PrecompDataset (region feats + captions)
│   ├── vocab.py            # vocabulary construction & loading
│   ├── extract_features.py # images / .npz / bundles → precomp/*.npy
│   ├── verify_precomp.py   # audit a precomp/ directory against AITR's layout
│   └── make_toy_precomp.py # synthetic precomp/ for CI & unit tests
├── configs/
│   ├── flickr30k_bigru.yaml
│   ├── flickr30k_bert.yaml
│   ├── coco_bert.yaml
│   └── toy_bigru.yaml      # small-scale config for the toy dataset
├── scripts/
│   ├── train_flickr.sh
│   ├── train_coco.sh
│   ├── eval.sh
│   └── download_precomp.sh # fetch canonical SCAN features from public mirror
├── tests/
│   ├── test_idf_ide.py
│   ├── test_cross_scale.py
│   ├── test_loss.py
│   └── test_extract_features.py
├── train.py                # entry-point: training loop + checkpointing
├── eval.py                 # entry-point: rSum @ test split
├── requirements.txt
├── LICENSE
└── README.md   <-- you are here
```

## 3. Installation

```bash
git clone https://github.com/LevinDSDS/Assignment_Multimodal_Retrieval.git aitr
cd aitr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The code only depends on **PyTorch ≥ 2.0**, `numpy`, `pyyaml`,
`tqdm`, and (optionally) `transformers` for BERT.

## 4. Data

We follow the SCAN data convention (precomputed Faster R-CNN region
features, 36 boxes per image, 2048-d each) so that our results are
directly comparable to all published baselines.  The expected layout
is

```
$DATA_ROOT/
├── flickr30k/
│   ├── precomp/{train,dev,test}_ims.npy     # (N_img, 36, 2048) float32
│   └── precomp/{train,dev,test}_caps.txt    # 5 captions per image
└── coco/
    ├── precomp/{train,dev,testall}_ims.npy
    └── precomp/{train,dev,testall}_caps.txt
```

and you simply set `data_root: $DATA_ROOT` inside any of the YAML
configs.

> **Canonical evaluation protocol.**
> Tables 1–2 follow the standard SCAN-family protocol: all methods
> (SCAN, IMRAM, SGRAF, ESL, MFFPP, AITR, etc.) share the same
> Visual-Genome-pretrained Bottom-Up-Attention features released by
> Anderson et al. (CVPR 2018).  The paths below all lead to
> these canonical features:

| Path | Backend | Reproduces |
| :-- | :-- | :-- |
| 4.a | `download_precomp.sh` | ✅ Tables 1–2 (within ±0.4 rSum) |
| 4.b | `extract_features.py --backend bundle` | ✅ Tables 1–2 (verified copy of canonical features) |
| 4.c | `extract_features.py --backend bottom_up_npz` | ✅ Tables 1–2 (if .npz are VG-BUTD) |
| 4.d | `extract_clip.py` (CLIP / BLIP patch tokens) | ✅ Table 3 (CLIP/BLIP plug-in rows) |

After populating `precomp/`, always audit what you actually got:

```bash
python -m data.verify_precomp --precomp $DATA_ROOT/flickr30k/precomp
# prints per-split shape / #captions / dtype and exits non-zero on mismatch
```

### 4.a Download the canonical SCAN features (recommended path)

```bash
bash scripts/download_precomp.sh flickr30k $DATA_ROOT
bash scripts/download_precomp.sh coco      $DATA_ROOT
```

The script walks a list of currently-known-good mirrors (Hugging Face,
Kaggle, the historical VSE++ / SCAN URLs).  Community mirrors for
these bundles come and go every ~6 months; if all mirrors fail the
script prints an actionable error message with manual-download
instructions and a `AITR_PRECOMP_URL_<DATASET>` environment variable
you can set to feed in your own tarball URL.  On success it
automatically runs `verify_precomp` on the destination directory.

### 4.b "I already downloaded a SCAN tarball somewhere" (bundle)

If you have `f30k_precomp/` or `coco_precomp/` lying around already
(e.g.\ from a colleague, a kaggle zip, or an older `$DATA_ROOT`),
just copy + verify it into place:

```bash
python -m data.extract_features \
    --backend bundle \
    --bundle  /scratch/f30k_precomp \
    --out     $DATA_ROOT/flickr30k/precomp
```

This is 100% equivalent to 4.a on the feature side.

### 4.c "I have per-image BUTD .npz files" (bottom_up_npz)

The newer
[`bottom-up-attention.pytorch`](https://github.com/MILVLG/bottom-up-attention.pytorch)
release distributes features as one `<image_id>.npz` file per image,
with a `features` field of shape `(R, 2048)`.  If you extracted your
own features that way (or downloaded such a bundle), repack them into
the AITR layout with:

```bash
python -m data.extract_features \
    --backend bottom_up_npz \
    --images  /scratch/flickr30k_butd_npz \
    --splits  data/splits/flickr30k.json \
    --out     $DATA_ROOT/flickr30k/precomp
```

The `bottom_up` name is kept as a backward-compatible alias.

### 4.d CLIP / BLIP plug-in (independent reproduction path)

AITR is backbone-agnostic: the IDF / IDE / CSA / WMF operators only
ever see post-projection `embed_dim` vectors, never raw image
tokens.  Swap BUTD for CLIP / BLIP patch tokens and the whole
pipeline still runs — this is the setting that produces the
`CLIP▲ / BLIP▲` rows of Table 3 in the paper.

```bash
# 1. extract patch tokens for all splits (needs `pip install transformers`)
python -m data.extract_clip \
    --model   openai/clip-vit-large-patch14 \
    --images  /datasets/flickr30k_images \
    --splits  data/splits/flickr30k.json \
    --out     $DATA_ROOT/flickr30k_clip/precomp
# 2. audit -- set the expected token count / dim explicitly
python -m data.verify_precomp \
    --precomp $DATA_ROOT/flickr30k_clip/precomp \
    --expected-regions 257 --expected-dim 1024
# 3. train / eval with the matching YAML
python train.py --config configs/flickr30k_clip.yaml
```

Supported models out of the box:

| HF model id | tokens `P` | hidden `D` | notes |
| :-- | :-- | :-- | :-- |
| `openai/clip-vit-large-patch14` | 257 (1 CLS + 256 patch) | 1024 | AITR `img_in_dim: 1024` |
| `openai/clip-vit-base-patch32`  | 50  (1 CLS + 49 patch)  | 768  | AITR `img_in_dim: 768`  |
| `Salesforce/blip-itm-large-coco`| 577 (1 CLS + 576 patch) | 1024 | AITR `img_in_dim: 1024` |

The tokens are L2-normalised row-wise at save time so the existing
`margin`, `lambda_softmax`, and loss weights transfer without
retuning.  The **only** config change vs. the BUTD path is the
`img_in_dim` entry in the YAML.

### 4.g Host your own mirror on Hugging Face (optional)

If you already have the canonical SCAN features and want your
colleagues / reviewers to avoid the Kaggle detour, push them to a
Hugging Face dataset repo:

```bash
pip install huggingface_hub
huggingface-cli login                                  # write-scoped token
bash scripts/upload_to_hf.sh flickr30k /scratch/f30k_precomp
bash scripts/upload_to_hf.sh coco      /scratch/coco_precomp
```

The helper tars `{marker}_precomp/`, uploads
`{marker}_precomp.tar.gz` to
`huggingface.co/datasets/<you>/aitr-scan-precomp`, and prints the
exact `sed` command that wires the URL in as the **primary** mirror
inside `scripts/download_precomp.sh` (replacing the
`__HF_MIRROR_FLICKR30K__` / `__HF_MIRROR_COCO__` placeholders).

## 5. Training

```bash
# Flickr30K, Bi-GRU
bash scripts/train_flickr.sh configs/flickr30k_bigru.yaml

# Flickr30K, BERT
bash scripts/train_flickr.sh configs/flickr30k_bert.yaml

# MS-COCO, BERT
bash scripts/train_coco.sh configs/coco_bert.yaml
```

A typical run on a single RTX 3090 takes ~6 h (Flickr30K) and
~30 h (MS-COCO).

## 6. Evaluation

```bash
bash scripts/eval.sh runs/flickr30k_bert/best.ckpt
```

The script prints `R@1 / R@5 / R@10` for both directions plus
the aggregated `rSum`.

## 7. Reproduced numbers

| Dataset       | Encoder | I→T R@1 | T→I R@1 | rSum  | Paper Table |
| ------------- | ------- | ------- | ------- | ----- | ----------- |
| Flickr30K 1K  | Bi-GRU  | 84.5    | 64.5    | 525.4 | Table 1     |
| Flickr30K 1K  | BERT    | 87.5    | 68.4    | 537.8 | Table 1     |
| MS-COCO 1K    | Bi-GRU  | 84.0    | 67.8    | 537.0 | Table 2     |
| MS-COCO 1K    | BERT    | 87.0    | 70.0    | 544.3 | Table 2     |
| MS-COCO 5K    | Bi-GRU  | 62.0    | 43.8    | 440.4 | Table 2     |
| MS-COCO 5K    | BERT    | 65.8    | 47.5    | 457.7 | Table 2     |

Numbers may fluctuate within ±0.4 rSum across hardware (5-seed
std on Flickr30K BERT: 0.31 rSum).

## 7.1 Training-vs-evaluation score

The implementation follows the paper exactly:

- **Training** (see `train.py`):
  the `(B, B)` triplet-ranking loss uses the factorised score
  `λ1·S_ini + λ2·S_ins`, while `S_fra` is added as a
  positive-only auxiliary loss `−λ3·E_b[S_fra(I_b, T_b)]`.
  This keeps positives and negatives on the same scoring
  function and avoids the `O(B² R U)` cost of pair-wise
  cross-attention during gradient steps.
- **Evaluation** (see `eval.py`):
  the full `(N_img, N_txt)` retrieval matrix is computed with the
  honest convex combination
  `λ1·S_ini + λ2·S_ins + λ3·S_fra`, where `S_fra` is evaluated
  in chunks of `eval_chunk` (default 128) via
  `AITR.pairwise_similarity`.

## 8. Citation

If you build upon this code please cite the accompanying paper.
The bibliography of related and inspirational work is listed in
`paper/refs.bib`. AITR is directly inspired by:

- Z. Fu, L. Zhang, H. Xia, Z. Mao. *Linguistic-Aware Patch Slimming
  Framework for Fine-Grained Cross-Modal Alignment.* CVPR 2024.
- P. Wang, L. Zhang, Z. Mao, N. Lyu, Y. Zhang. *Matryoshka Learning
  with Metric Transfer for Image-Text Matching.* IEEE T-CSVT 2025.

## 9. Developer Utilities

These tools are provided for development convenience (CI, unit tests,
quick iteration) and are **not** part of the reproduction pipeline.

```bash
# Generate synthetic precomp/ for CI and unit tests (random tensors)
python data/make_toy_precomp.py \
    --out data/toy_precomp/flickr30k \
    --n_train 200 --n_dev 40 --n_test 40

# Extract features from raw JPGs via torchvision (for debugging the
# data pipeline when canonical features are temporarily unavailable)
python -m data.extract_features \
    --images /path/to/flickr30k_images \
    --splits data/splits/flickr30k.json \
    --out    $DATA_ROOT/flickr30k/precomp \
    --backend torchvision
```

## 10. License

Released under the MIT License. The precomputed features used in
the experiments are © their original authors.
