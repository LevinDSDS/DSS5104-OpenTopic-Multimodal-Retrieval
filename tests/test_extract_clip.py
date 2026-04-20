"""Lightweight tests for ``data.extract_clip``.

We do **not** run the real CLIP / BLIP forward pass here (it would
download ~1-3 GB of weights and dominate CI time).  We only exercise
the parts that run without a HuggingFace hub round-trip:

* ``_l2_normalise``   -- ensures saved features are unit-norm.
* ``_auto_device``    -- CPU fallback when CUDA is unavailable.
* CLI surface         -- ``python -m data.extract_clip --help`` exits
  0 and prints the supported backbones in the docstring.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from data.extract_clip import _auto_device, _l2_normalise


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def test_l2_normalise_is_unit_norm():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 257, 1024)).astype(np.float32)
    y = _l2_normalise(x)
    assert y.dtype == np.float32
    # every token should have unit norm (up to FP tolerance)
    norms = np.linalg.norm(y, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), (norms.max(), norms.min())


def test_auto_device_falls_back_to_cpu():
    # When we ask for a CUDA device but PyTorch reports no CUDA, the
    # helper must silently downgrade to cpu. We cannot actually
    # simulate "CUDA present" here, so we only assert the negative
    # branch is wired up correctly.
    import torch
    if torch.cuda.is_available():
        # best we can do is confirm the happy path still returns cuda.
        assert _auto_device("cuda:0") == "cuda:0"
    else:
        assert _auto_device("cuda") == "cpu"
        assert _auto_device("cuda:0") == "cpu"
        # Non-CUDA strings must round-trip unchanged.
        assert _auto_device("cpu") == "cpu"
        assert _auto_device("mps") == "mps"


def test_cli_help_smoke():
    """`python -m data.extract_clip --help` must exit zero and mention
    the plug-in nature of this path."""
    result = subprocess.run(
        [sys.executable, "-m", "data.extract_clip", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert "clip-vit-large-patch14" in result.stdout
    assert "--splits" in result.stdout
