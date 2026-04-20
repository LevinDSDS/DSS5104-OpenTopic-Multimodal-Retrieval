"""Verify a ``precomp/`` directory against AITR's expected layout.

AITR's :class:`data.dataset.PrecompDataset` expects, for every split
(``train``, ``dev``, ``test`` and additionally ``testall`` on COCO)::

    precomp/{split}_ims.npy     # (N_img, K, 2048)  float32,  K == 36
    precomp/{split}_caps.txt    # exactly 5 * N_img lines, 1 caption per line

This CLI reports exactly *which* of those files are present / missing /
mis-shaped, so that users can tell at a glance whether the bundle they
just downloaded or extracted is actually compatible.  It is invoked
automatically by ``scripts/download_precomp.sh`` after unpacking, and
can be run manually whenever you suspect something is off::

    python -m data.verify_precomp --precomp $DATA_ROOT/flickr30k/precomp

Exit codes:
  * 0  -- every expected split is present and well-shaped.
  * 1  -- at least one expected file is missing or has the wrong shape.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


EXPECTED_REGIONS = 36          # BUTD / SCAN convention
EXPECTED_DIM     = 2048
EXPECTED_DTYPE   = np.float32
CAPS_PER_IMAGE   = 5


# Which splits should exist for each known dataset.  A dataset whose
# directory name we do not recognise is audited with the generic
# (``train``, ``dev``, ``test``) triplet.
_SPLITS_BY_DATASET = {
    "flickr30k":    ("train", "dev", "test"),
    "f30k_precomp": ("train", "dev", "test"),
    "coco":         ("train", "dev", "test", "testall"),
    "coco_precomp": ("train", "dev", "test", "testall"),
}


@dataclass
class SplitReport:
    split: str
    ok: bool
    messages: List[str] = field(default_factory=list)
    n_img: Optional[int] = None
    n_cap: Optional[int] = None


def _expected_splits(precomp_dir: str) -> Tuple[str, ...]:
    """Guess which splits to require from the directory / parent name."""
    parent = os.path.basename(os.path.normpath(precomp_dir))
    grand  = os.path.basename(os.path.dirname(os.path.normpath(precomp_dir)))
    for key in (parent, grand):
        if key.lower() in _SPLITS_BY_DATASET:
            return _SPLITS_BY_DATASET[key.lower()]
    return ("train", "dev", "test")


def _verify_split(precomp_dir: str, split: str,
                  expected_regions: int = EXPECTED_REGIONS,
                  expected_dim: int = EXPECTED_DIM) -> SplitReport:
    rep = SplitReport(split=split, ok=True)
    ims = os.path.join(precomp_dir, f"{split}_ims.npy")
    cps = os.path.join(precomp_dir, f"{split}_caps.txt")

    if not os.path.isfile(ims):
        rep.ok = False
        rep.messages.append(f"missing image features: {ims}")
    else:
        arr = np.load(ims, mmap_mode="r")
        rep.n_img = int(arr.shape[0])
        if arr.ndim != 3:
            rep.ok = False
            rep.messages.append(
                f"{ims}: expected 3-D (N, K, D) tensor, got ndim={arr.ndim} "
                f"shape={tuple(arr.shape)}")
        else:
            _, k, d = arr.shape
            if k != expected_regions:
                rep.ok = False
                rep.messages.append(
                    f"{ims}: expected {expected_regions} tokens per image, "
                    f"got {k}. For BUTD/SCAN set --expected-regions 36. "
                    f"For CLIP ViT-L/14 use 257 (256 patches + 1 CLS). "
                    f"If you have a BUTD bundle with variable #boxes per "
                    f"image, re-pack it with data/extract_features.py "
                    f"--backend bottom_up_npz first.")
            if d != expected_dim:
                rep.ok = False
                rep.messages.append(
                    f"{ims}: expected {expected_dim}-D features, got {d}. "
                    f"Common hidden sizes: 2048 (BUTD/R-101), 1024 "
                    f"(CLIP ViT-L/14, BLIP-large), 768 (CLIP ViT-B/16, "
                    f"ViT-B/32), 512 (CLIP ViT-B/32 proj). Pass the "
                    f"matching `--expected-dim` or check your config's "
                    f"`img_in_dim`.")
        if arr.dtype != EXPECTED_DTYPE:
            rep.messages.append(
                f"{ims}: dtype is {arr.dtype}, not float32 -- AITR will cast "
                f"at load time, but consider resaving to save memory.")

    if not os.path.isfile(cps):
        rep.ok = False
        rep.messages.append(f"missing captions file: {cps}")
    else:
        with open(cps, encoding="utf-8") as f:
            n_cap = sum(1 for _ in f if _.strip())
        rep.n_cap = n_cap
        if rep.n_img is not None and n_cap != CAPS_PER_IMAGE * rep.n_img:
            rep.ok = False
            rep.messages.append(
                f"{cps}: expected {CAPS_PER_IMAGE} captions per image "
                f"({CAPS_PER_IMAGE}*{rep.n_img}={CAPS_PER_IMAGE*rep.n_img}), "
                f"got {n_cap}.")

    return rep


def verify(precomp_dir: str,
           splits: Optional[Tuple[str, ...]] = None,
           expected_regions: int = EXPECTED_REGIONS,
           expected_dim: int = EXPECTED_DIM) -> List[SplitReport]:
    if splits is None:
        splits = _expected_splits(precomp_dir)
    return [_verify_split(precomp_dir, s,
                          expected_regions=expected_regions,
                          expected_dim=expected_dim)
            for s in splits]


def _print_report(reports: List[SplitReport]) -> None:
    ok = all(r.ok for r in reports)
    for r in reports:
        prefix = "OK " if r.ok else "XX "
        n_img = r.n_img if r.n_img is not None else "?"
        n_cap = r.n_cap if r.n_cap is not None else "?"
        print(f"  [{prefix}] {r.split:<8} N_img={n_img}  N_cap={n_cap}")
        for m in r.messages:
            print(f"         ! {m}")
    if ok:
        print("[OK] precomp layout matches AITR's expectations.")
    else:
        print("[!!] precomp layout is incompatible with AITR -- see messages "
              "above.  Fix with data/extract_features.py or by re-downloading "
              "the canonical SCAN bundle (scripts/download_precomp.sh).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precomp", required=True,
                    help="path to a precomp/ directory to audit")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="override the auto-detected split list")
    ap.add_argument("--expected-regions", type=int, default=EXPECTED_REGIONS,
                    help=f"tokens per image; default {EXPECTED_REGIONS} "
                         f"(BUTD). CLIP ViT-L/14 -> 257.")
    ap.add_argument("--expected-dim", type=int, default=EXPECTED_DIM,
                    help=f"feature hidden size; default {EXPECTED_DIM} "
                         f"(BUTD). CLIP ViT-L/14 -> 1024.")
    args = ap.parse_args()

    splits = tuple(args.splits) if args.splits else None
    reports = verify(args.precomp, splits,
                     expected_regions=args.expected_regions,
                     expected_dim=args.expected_dim)
    _print_report(reports)
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
