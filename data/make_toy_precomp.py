"""Generate a tiny synthetic ``precomp/`` directory for CI and unit tests.

Produces the exact layout consumed by :class:`data.dataset.PrecompDataset`
(``train/dev/test_ims.npy`` + ``train/dev/test_caps.txt``) with randomly-
sampled image features and short captions.  Used to verify that the full
training and evaluation pipeline runs correctly without downloading the
canonical features (see ``download_precomp.sh`` for the canonical path).

Usage::

    python -m data.make_toy_precomp \\
        --out        data/toy_precomp/flickr30k \\
        --n_train    200  --n_dev 40  --n_test 40  --cap_per_img 5

The output tree is::

    data/toy_precomp/flickr30k/
    └── precomp/
        ├── train_ims.npy       # (n_train, 36, 2048) float32
        ├── train_caps.txt      # 5 captions per image, one per line
        ├── dev_ims.npy, dev_caps.txt
        └── test_ims.npy, test_caps.txt

Set ``data_root: data/toy_precomp`` in any YAML config to run
``python train.py`` end-to-end on a laptop.
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List

import numpy as np


_SEED_POOL = [
    "a dog runs across the grass",
    "a child holds a red balloon on the beach",
    "a man rides a snowboard down a mountain slope",
    "a woman reads a book on a wooden bench",
    "two cats play near a window in the afternoon",
    "an elderly couple walks along a quiet river",
    "a red car drives through a rainy city street",
    "a white horse grazes peacefully in a field",
    "a boy kicks a football on a sunlit pitch",
    "a group of friends laughs around a dinner table",
]


def _rand_caption(rng: random.Random) -> str:
    base = rng.choice(_SEED_POOL)
    # light jitter so every caption is unique (captions must be unique
    # for the retrieval task to be non-degenerate).
    suffix = rng.choice(["", " at sunset", " in winter", " near a lake",
                         " under a bridge", " with friends"])
    return (base + suffix).strip()


def _write_split(out_dir: str,
                 split: str,
                 n_img: int,
                 cap_per_img: int,
                 rng: random.Random,
                 n_regions: int = 36,
                 feat_dim: int = 2048) -> None:
    rng_np = np.random.default_rng(rng.randint(0, 2**31 - 1))
    # low-rank structure so two captions of the same image are
    # slightly correlated, giving the model a learnable signal.
    basis = rng_np.standard_normal((n_img, 8, feat_dim)).astype(np.float32)
    mix = rng_np.standard_normal((n_img, n_regions, 8)).astype(np.float32)
    feats = np.einsum("nrk,nkd->nrd", mix, basis) / np.sqrt(8.0)
    feats += 0.1 * rng_np.standard_normal(
        (n_img, n_regions, feat_dim)).astype(np.float32)
    captions: List[str] = []
    for _ in range(n_img):
        captions.extend(_rand_caption(rng) for _ in range(cap_per_img))

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split}_ims.npy"),
            feats.astype(np.float32))
    with open(os.path.join(out_dir, f"{split}_caps.txt"),
              "w", encoding="utf-8") as f:
        for c in captions:
            f.write(c + "\n")
    print(f"  [{split}] images={n_img} captions={len(captions)} "
          f"→ {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",          required=True,
                    help="output dataset folder (precomp/ will be created "
                         "inside).")
    ap.add_argument("--n_train",      type=int, default=200)
    ap.add_argument("--n_dev",        type=int, default=40)
    ap.add_argument("--n_test",       type=int, default=40)
    ap.add_argument("--cap_per_img",  type=int, default=5)
    ap.add_argument("--seed",         type=int, default=7)
    ap.add_argument("--include_testall", action="store_true",
                    help="also emit testall_ims.npy/testall_caps.txt "
                         "for MS-COCO-style 5K evaluation.")
    args = ap.parse_args()

    precomp = os.path.join(args.out, "precomp")
    os.makedirs(precomp, exist_ok=True)
    rng = random.Random(args.seed)

    _write_split(precomp, "train",  args.n_train, args.cap_per_img, rng)
    _write_split(precomp, "dev",    args.n_dev,   args.cap_per_img, rng)
    _write_split(precomp, "test",   args.n_test,  args.cap_per_img, rng)
    if args.include_testall:
        _write_split(precomp, "testall",
                     args.n_test * 5, args.cap_per_img, rng)

    print(f"done. set `data_root: {os.path.dirname(args.out) or '.'}`"
          f" and `dataset: {os.path.basename(args.out)}` "
          "in your YAML config.")


if __name__ == "__main__":
    main()
