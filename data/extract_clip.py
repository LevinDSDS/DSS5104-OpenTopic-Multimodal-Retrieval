"""Extract CLIP / BLIP patch-token features in the AITR precomp layout.

This is a fully independent reproduction path that does **not** depend
on any Visual-Genome-pretrained Bottom-Up-Attention tensors.  It
replaces the 36 BUTD region features with the ``P`` patch tokens
produced by a pre-trained vision transformer:

* ``openai/clip-vit-large-patch14``  ->  ``P = 256 + 1 = 257`` tokens,
  ``D = 1024``
* ``openai/clip-vit-base-patch32``   ->  ``P =  49 + 1 =  50`` tokens,
  ``D =  768``
* ``Salesforce/blip-itm-large-coco`` ->  ``P = 576 + 1 = 577`` tokens,
  ``D = 1024``   (from ``vision_model.last_hidden_state``)

The saved ``{split}_ims.npy`` tensor therefore has shape
``(N_img, P, D)``, with ``D`` matching the ``img_in_dim`` entry in the
YAML config (see ``configs/flickr30k_clip.yaml`` for an example).
Captions are written byte-identically to the BUTD path.

Why this works with AITR without any model changes
---------------------------------------------------
AITR treats the visual side as a bag of tokens of shape ``(B, P, D)``
with an ``img_in_dim -> embed_dim`` linear projector (see
``aitr/encoders.py``).  The specific backbone, the meaning of each
token (object region vs. patch) and the token count ``P`` are all
backbone-agnostic -- the IDF / IDE / CSA / WMF operators never look at
tokens directly, they operate on the post-projection ``embed_dim``
vectors.  Setting ``img_in_dim`` to the vision encoder's hidden size
is the only config knob required.

This path is *exactly* the plug-in recipe used by LAPS
(Fu et al., CVPR 2024), X-Dim, and MAMET when running their
re-ranker on top of CLIP / BLIP visual features.

Caveats
-------
1. The text side also needs CLIP-style tokens when
   ``text_encoder: clip`` is set in the YAML; here we only handle the
   visual side (the BUTD families of AITR use Bi-GRU or BERT on the
   text side, which keeps backbones symmetric with the vision stream).
2. Patch tokens are L2-normalised row-wise at save time to match the
   scale of BUTD features (~unit-norm), so existing hyper-parameters
   (``lambda_softmax``, ``margin``, ``lambdas``) transfer without
   retuning.
3. We only emit the ``{split}_ims.npy`` + ``{split}_caps.txt`` files;
   running ``data/verify_precomp.py`` after extraction will flag a
   ``--D != 2048`` mismatch -- that is expected and benign, AITR still
   consumes the data as long as the YAML's ``img_in_dim`` matches.

Usage
-----
::

    python -m data.extract_clip \\
        --model   openai/clip-vit-large-patch14 \\
        --images  /datasets/flickr30k_images \\
        --splits  data/splits/flickr30k.json \\
        --out     $DATA_ROOT/flickr30k_clip/precomp

and then train with::

    python train.py --config configs/flickr30k_clip.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np


def _encode_batch_clip(model, processor, pil_images, device):
    """Return ``(B, P, D)`` patch tokens from an HF CLIP vision tower."""
    import torch
    with torch.no_grad():
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        out = model.vision_model(**inputs,
                                 output_hidden_states=False,
                                 return_dict=True)
        # last_hidden_state: (B, P, D) with P = 1 CLS + H*W patches
        return out.last_hidden_state.float().cpu().numpy()


def _encode_batch_blip(model, processor, pil_images, device):
    import torch
    with torch.no_grad():
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        out = model.vision_model(**inputs, return_dict=True)
        return out.last_hidden_state.float().cpu().numpy()


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalise each token so the magnitude matches BUTD."""
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return (x / n).astype(np.float32)


def _auto_device(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("  [!] CUDA not available; falling back to CPU.")
                return "cpu"
        except ImportError:
            return "cpu"
    return device


def _load_model(name: str, device: str):
    """Load an HF CLIP- or BLIP-family vision model + image processor."""
    try:
        from transformers import (AutoProcessor, CLIPModel, BlipForImageTextRetrieval)
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "extract_clip requires `transformers`.  Install via "
            "`pip install transformers`.") from exc

    lname = name.lower()
    processor = AutoProcessor.from_pretrained(name)
    if "blip" in lname:
        model = BlipForImageTextRetrieval.from_pretrained(name).to(device).eval()
        return model, processor, _encode_batch_blip
    # default to CLIP (covers ViT-B/32, ViT-L/14, etc.)
    model = CLIPModel.from_pretrained(name).to(device).eval()
    return model, processor, _encode_batch_clip


def extract(image_files: List[str],
            model_name: str,
            device: str,
            batch_size: int = 16) -> np.ndarray:
    from PIL import Image
    device = _auto_device(device)
    model, processor, encode = _load_model(model_name, device)

    out = None
    N = len(image_files)
    for i0 in range(0, N, batch_size):
        batch = [Image.open(f).convert("RGB")
                 for f in image_files[i0:i0 + batch_size]]
        feats = encode(model, processor, batch, device)          # (b, P, D)
        feats = _l2_normalise(feats)
        if out is None:
            out = np.zeros((N, feats.shape[1], feats.shape[2]),
                           dtype=np.float32)
        out[i0:i0 + feats.shape[0]] = feats
        if (i0 // batch_size) % 20 == 0:
            print(f"    {model_name} {i0 + feats.shape[0]}/{N}")
    return out


def _write_split(out_dir: str, split: str,
                 feats: np.ndarray, caps: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split}_ims.npy"), feats)
    with open(os.path.join(out_dir, f"{split}_caps.txt"),
              "w", encoding="utf-8") as f:
        for c in caps:
            f.write(c.strip() + "\n")
    print(f"  [{split}] images={feats.shape[0]} "
          f"tokens={feats.shape[1]} dim={feats.shape[2]} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. "
                         "openai/clip-vit-large-patch14 or "
                         "Salesforce/blip-itm-large-coco.")
    ap.add_argument("--images", required=True,
                    help="folder with raw JPGs matching the splits JSON.")
    ap.add_argument("--splits", required=True,
                    help="SCAN-style splits JSON (see data/splits/).")
    ap.add_argument("--out", required=True,
                    help="destination precomp/ directory.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda",
                    help="torch device; auto-falls-back to CPU.")
    args = ap.parse_args()

    with open(args.splits, "r", encoding="utf-8") as f:
        splits: Dict[str, List[dict]] = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    for split_name, entries in splits.items():
        image_files = [os.path.join(args.images, e["image_id"])
                       for e in entries]
        caps: List[str] = []
        for e in entries:
            caps.extend(e["captions"][:5])
        feats = extract(image_files, args.model,
                        device=args.device,
                        batch_size=args.batch_size)
        _write_split(args.out, split_name, feats, caps)

    # A verify step here would flag the intentional D != 2048 mismatch;
    # we instead print an informative notice and exit cleanly.
    print("done. Note: tokens are CLIP / BLIP patches with a non-2048 "
          "hidden dim, so `verify_precomp.py` will flag the dim. Set "
          "the matching `img_in_dim` in your YAML (e.g. 1024 for "
          "clip-vit-large-patch14) and AITR will consume the data "
          "natively.")


if __name__ == "__main__":
    sys.exit(main() or 0)
