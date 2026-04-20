"""Smoke tests for ``data.extract_features`` and ``data.verify_precomp``.

Four paths are exercised:

* ``bottom_up`` backend (backward-compatible alias for ``bottom_up_npz``):
  pure NumPy re-packing of per-image .npz files.  Runs unconditionally,
  including on the CPU-only CI runner.

* ``bottom_up_npz`` backend: canonical name for the same path; verified
  in a dedicated test to protect against typo-breakage.

* ``bundle`` backend: copy an already-assembled SCAN-format directory
  into place.  No detector needed.

* ``torchvision`` backend: only runs when ``torchvision`` is available.
  Skipped otherwise to keep CI fast and deterministic.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))


def _pkg_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


_HAS_TORCHVISION = _pkg_available("torchvision")
_IN_CI = os.environ.get("CI", "").lower() in ("true", "1")


# ------------------------------------------------------------ helpers
def _make_splits(root: str,
                 n_train: int = 3,
                 n_dev: int = 2,
                 n_test: int = 2) -> str:
    def _entries(prefix: str, n: int):
        return [
            {"image_id": f"{prefix}{i:04d}.jpg",
             "captions": [f"caption {prefix} {i} {k}" for k in range(5)]}
            for i in range(n)
        ]

    splits = {
        "train": _entries("tr", n_train),
        "dev":   _entries("dv", n_dev),
        "test":  _entries("te", n_test),
    }
    path = os.path.join(root, "splits.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f)
    return path


def _write_synthetic_butd(feat_dir: str,
                     splits_path: str,
                     top_k: int = 36,
                     dim: int = 2048) -> None:
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    os.makedirs(feat_dir, exist_ok=True)
    rng = np.random.default_rng(123)
    for entries in splits.values():
        for e in entries:
            base = os.path.splitext(e["image_id"])[0]
            feats = rng.standard_normal((top_k, dim)).astype(np.float32)
            np.savez(os.path.join(feat_dir, base + ".npz"),
                     features=feats)


def _write_synthetic_bundle(bundle_dir: str,
                       n_by_split: dict,
                       top_k: int = 36,
                       dim: int = 2048) -> None:
    """Create a SCAN-format bundle directory (bulk _ims.npy + _caps.txt)."""
    os.makedirs(bundle_dir, exist_ok=True)
    rng = np.random.default_rng(321)
    for split, n in n_by_split.items():
        feats = rng.standard_normal((n, top_k, dim)).astype(np.float32)
        np.save(os.path.join(bundle_dir, f"{split}_ims.npy"), feats)
        with open(os.path.join(bundle_dir, f"{split}_caps.txt"),
                  "w", encoding="utf-8") as f:
            for i in range(n):
                for k in range(5):
                    f.write(f"bundle {split} img{i} cap{k}\n")


# -------------------------------------------------------- bottom-up path
@pytest.mark.parametrize("backend", ["bottom_up", "bottom_up_npz"])
def test_bottom_up_roundtrip(backend):
    """Both aliases must produce the exact precomp layout."""
    with tempfile.TemporaryDirectory() as tmp:
        feat_dir    = os.path.join(tmp, "butd")
        out_dir     = os.path.join(tmp, "precomp")
        splits_path = _make_splits(tmp)
        _write_synthetic_butd(feat_dir, splits_path)

        cmd = [sys.executable, "-m", "data.extract_features",
               "--images",  feat_dir,
               "--splits",  splits_path,
               "--out",     out_dir,
               "--backend", backend]
        subprocess.check_call(cmd, cwd=REPO_ROOT)

        for split, n_img in [("train", 3), ("dev", 2), ("test", 2)]:
            feats = np.load(os.path.join(out_dir, f"{split}_ims.npy"))
            assert feats.shape == (n_img, 36, 2048), \
                f"{split} feats have wrong shape: {feats.shape}"
            assert feats.dtype == np.float32
            caps = open(
                os.path.join(out_dir, f"{split}_caps.txt"),
                encoding="utf-8").read().strip().split("\n")
            assert len(caps) == n_img * 5, \
                f"{split} expects 5 caps/img, got {len(caps)/n_img:.1f}"


def test_bottom_up_raises_on_missing_npz():
    """Missing .npz files must surface a FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        feat_dir    = os.path.join(tmp, "butd")
        os.makedirs(feat_dir, exist_ok=True)
        splits_path = _make_splits(tmp, n_train=1, n_dev=0, n_test=0)
        out_dir     = os.path.join(tmp, "precomp")

        cmd = [sys.executable, "-m", "data.extract_features",
               "--images",  feat_dir,
               "--splits",  splits_path,
               "--out",     out_dir,
               "--backend", "bottom_up"]
        result = subprocess.run(cmd, cwd=REPO_ROOT,
                                capture_output=True, text=True)
        assert result.returncode != 0, \
            "expected non-zero exit on missing .npz"
        assert "missing bottom-up feature file" in (result.stderr
                                                     + result.stdout)


# ------------------------------------------------------------ bundle path
def test_bundle_copy(tmp_path):
    """``--backend bundle`` must copy & verify a SCAN-format dir."""
    bundle_dir = tmp_path / "scan_bundle"
    out_dir    = tmp_path / "precomp"
    _write_synthetic_bundle(str(bundle_dir),
                       {"train": 4, "dev": 2, "test": 2})

    cmd = [sys.executable, "-m", "data.extract_features",
           "--backend", "bundle",
           "--bundle",  str(bundle_dir),
           "--out",     str(out_dir)]
    subprocess.check_call(cmd, cwd=REPO_ROOT)

    for split, n_img in [("train", 4), ("dev", 2), ("test", 2)]:
        feats = np.load(out_dir / f"{split}_ims.npy")
        assert feats.shape == (n_img, 36, 2048)
        assert feats.dtype == np.float32
        n_lines = sum(1 for _ in open(out_dir / f"{split}_caps.txt",
                                      encoding="utf-8"))
        assert n_lines == 5 * n_img


def test_bundle_rejects_empty_dir(tmp_path):
    bundle_dir = tmp_path / "empty"
    bundle_dir.mkdir()
    out_dir = tmp_path / "precomp"
    cmd = [sys.executable, "-m", "data.extract_features",
           "--backend", "bundle",
           "--bundle",  str(bundle_dir),
           "--out",     str(out_dir)]
    result = subprocess.run(cmd, cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0, \
        "expected non-zero exit on empty bundle dir"


# ------------------------------------------------------- torchvision path
@pytest.mark.skipif(
    not _HAS_TORCHVISION or _IN_CI,
    reason="Skipped: torchvision not installed or CI (downloads ~500 MB models)")
def test_torchvision_one_image(tmp_path):
    """Run the torchvision backend end-to-end on a single tiny image."""
    import torch                            # noqa: F401 (import-time check)
    from PIL import Image

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    img_path = img_dir / "tr0000.jpg"
    # Use a 320x240 image so BUTD resize lands on ~600x800.
    arr = (np.random.RandomState(0).rand(240, 320, 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(img_path)

    splits = {"train": [{"image_id": "tr0000.jpg",
                         "captions": ["dummy"] * 5}]}
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))

    out_dir = tmp_path / "precomp"
    cmd = [sys.executable, "-m", "data.extract_features",
           "--images",  str(img_dir),
           "--splits",  str(splits_path),
           "--out",     str(out_dir),
           "--backend", "torchvision",
           "--device",  "cpu",
           "--top_k",   "36"]
    subprocess.check_call(cmd, cwd=REPO_ROOT)

    feats = np.load(out_dir / "train_ims.npy")
    assert feats.shape == (1, 36, 2048)
    assert feats.dtype == np.float32


# ----------------------------------------------------------- verify_precomp
def test_verify_precomp_passes_on_toy(tmp_path):
    """verify_precomp must exit 0 on a well-formed bundle."""
    _write_synthetic_bundle(str(tmp_path),
                       {"train": 2, "dev": 2, "test": 2})
    cmd = [sys.executable, "-m", "data.verify_precomp",
           "--precomp", str(tmp_path)]
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def test_verify_precomp_detects_wrong_dim(tmp_path):
    """verify_precomp must exit non-zero if region dim is wrong (e.g. CLIP)."""
    # write 777-D features (not a standard size) instead of the
    # 2048-D BUTD features AITR expects.
    rng = np.random.default_rng(0)
    np.save(tmp_path / "train_ims.npy",
            rng.standard_normal((2, 36, 777)).astype(np.float32))
    with open(tmp_path / "train_caps.txt", "w", encoding="utf-8") as f:
        for i in range(2):
            for k in range(5):
                f.write(f"bad dim img{i} cap{k}\n")

    cmd = [sys.executable, "-m", "data.verify_precomp",
           "--precomp", str(tmp_path),
           "--splits",  "train"]
    result = subprocess.run(cmd, cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "2048" in (result.stdout + result.stderr)


def test_verify_precomp_accepts_clip_path(tmp_path):
    """verify_precomp with --expected-regions / --expected-dim must
    accept a CLIP-style 257-token, 1024-D bundle."""
    rng = np.random.default_rng(0)
    for split, n in [("train", 2), ("dev", 2), ("test", 2)]:
        np.save(tmp_path / f"{split}_ims.npy",
                rng.standard_normal((n, 257, 1024)).astype(np.float32))
        with open(tmp_path / f"{split}_caps.txt", "w",
                  encoding="utf-8") as f:
            for i in range(n):
                for k in range(5):
                    f.write(f"clip {split} img{i} cap{k}\n")

    cmd = [sys.executable, "-m", "data.verify_precomp",
           "--precomp", str(tmp_path),
           "--expected-regions", "257",
           "--expected-dim",     "1024"]
    subprocess.check_call(cmd, cwd=REPO_ROOT)
