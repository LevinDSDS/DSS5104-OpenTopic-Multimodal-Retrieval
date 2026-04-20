"""Produce AITR-compatible ``precomp/*.npy`` + ``precomp/*.txt`` files.

The canonical numbers in Tables 1--2 of ``paper/final.tex`` come from
the Bottom-Up-Attention features released by Anderson~et~al.~(CVPR
2018): a Faster R-CNN with a **Visual-Genome-pretrained** ResNet-101
backbone emitting 36 regions of 2048-D per image.  Those original
Caffe2 / Detectron2 checkpoints are painful to re-run; we therefore
support **three** paths to the same ``precomp/`` layout, each with
clearly different reproducibility guarantees.

+-----------------------+----------------------------------------------+------------------------+
| ``--backend``         | Input                                        | rSum vs. paper (F30K) |
+=======================+==============================================+========================+
| ``bundle``            | Pre-assembled ``{split}_ims.npy`` +          | **Exact** (+/- 0.4)    |
|                       | ``{split}_caps.txt`` in any directory        |                        |
|                       | (what SCAN / VSE++ actually distribute).     |                        |
|                       | Simply verified + copied into ``--out``.     |                        |
+-----------------------+----------------------------------------------+------------------------+
| ``bottom_up_npz``     | One ``<image_id>.npz`` per image with a      | **Exact** (+/- 0.4)    |
|                       | ``features`` field of shape ``(>=36, 2048)`` | when each .npz is      |
|                       | (the newer ``bottom-up-attention.pytorch``   | VG-pretrained; else    |
|                       | distribution). Re-packed in SCAN order.      | bounded by the source. |
+-----------------------+----------------------------------------------+------------------------+
| ``torchvision``       | Raw JPGs + SCAN split JSON.  Uses            | Numerically distinct   |
|                       | ``fasterrcnn_resnet50_fpn_v2`` (COCO) +      | from Tables 1--2       |
|                       | ``resnet101(IMAGENET1K_V2)`` as a portable   | (different detector    |
|                       | stand-in for BUTD.  Different pre-training   | pre-training corpus).  |
|                       | (COCO/ImageNet vs. Visual Genome) changes    | Useful for verifying   |
|                       | the region semantics significantly -- see    | pipeline correctness   |
|                       | sec.~Appendix~A.4 of the paper.              | when canonical         |
|                       |                                              | features are           |
|                       |                                              | unavailable.           |
+-----------------------+----------------------------------------------+------------------------+

Examples::

    # 1) "I grabbed the official SCAN tarball and just need it in the
    #     AITR-expected directory layout."
    python -m data.extract_features \\
        --bundle /scratch/data/f30k_precomp \\
        --out    $DATA_ROOT/flickr30k/precomp \\
        --backend bundle

    # 2) "I have per-image BUTD .npz files from bottom-up-attention.pytorch."
    python -m data.extract_features \\
        --images /scratch/flickr30k_butd_npz \\
        --splits data/splits/flickr30k.json \\
        --out    $DATA_ROOT/flickr30k/precomp \\
        --backend bottom_up_npz

    # 3) "I only have raw images; give me a runnable feature set
    #     (note: different backbone pre-training from Tables 1-2)."
    python -m data.extract_features \\
        --images /scratch/flickr30k_images \\
        --splits data/splits/flickr30k.json \\
        --out    $DATA_ROOT/flickr30k/precomp \\
        --backend torchvision

After every run the output is audited via
:func:`data.verify_precomp.verify` -- the script exits non-zero if the
written ``precomp/`` dir is not consumable by :class:`PrecompDataset`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np


# =============================================================== utils
def _write_split(out_dir: str,
                 split_name: str,
                 feats: np.ndarray,
                 captions: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split_name}_ims.npy"),
            feats.astype(np.float32))
    with open(os.path.join(out_dir, f"{split_name}_caps.txt"),
              "w", encoding="utf-8") as f:
        for c in captions:
            f.write(c.strip() + "\n")
    print(f"  [{split_name}] images={feats.shape[0]} "
          f"captions={len(captions)} -> {out_dir}")


def _auto_device(device: str) -> str:
    """Silently fall back to CPU when CUDA is unavailable."""
    if device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("  [!] CUDA not available; falling back to CPU.")
                return "cpu"
        except ImportError:
            return "cpu"
    return device


# ================================================ backend: bundle copy
def _extract_bundle(bundle_dir: str, out_dir: str) -> None:
    """Verify + copy a pre-assembled {split}_ims.npy / _caps.txt bundle.

    This is what ``scripts/download_precomp.sh`` produces, but it is
    also a convenient way to let users drop an existing SCAN-format
    directory into place without re-extracting features.
    """
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"--bundle points to {bundle_dir!r} "
                                f"which is not a directory.")
    os.makedirs(out_dir, exist_ok=True)

    any_split = False
    for fname in sorted(os.listdir(bundle_dir)):
        src = os.path.join(bundle_dir, fname)
        if not os.path.isfile(src):
            continue
        if fname.endswith("_ims.npy") or fname.endswith("_caps.txt"):
            shutil.copy2(src, os.path.join(out_dir, fname))
            any_split = True
    if not any_split:
        raise RuntimeError(
            f"no '*_ims.npy' / '*_caps.txt' files found in {bundle_dir}. "
            f"This backend expects a SCAN-style feature bundle.")
    print(f"  [bundle] copied SCAN-format splits -> {out_dir}")


# =========================================== backend: per-image .npz
def _extract_bottom_up_npz(image_ids: List[str],
                           feat_dir: str,
                           top_k: int) -> np.ndarray:
    """Re-pack pre-computed bottom-up-attention .npz into AITR layout."""
    out = np.zeros((len(image_ids), top_k, 2048), dtype=np.float32)
    for i, iid in enumerate(image_ids):
        stem = os.path.splitext(iid)[0]
        f = os.path.join(feat_dir, f"{stem}.npz")
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"missing bottom-up feature file: {f}. This backend "
                f"expects one .npz per image named '<image_id>.npz' "
                f"in --images, each containing a 'features' (or 'feat') "
                f"array of shape (>=36, 2048).")
        z = np.load(f)
        feat = z["features"] if "features" in z.files else z["feat"]
        if feat.ndim != 2 or feat.shape[1] != 2048:
            raise ValueError(
                f"{f}: expected 2-D features of shape (R, 2048), got "
                f"shape={tuple(feat.shape)}.")
        if feat.shape[0] < top_k:
            pad = feat.mean(0, keepdims=True).repeat(
                top_k - feat.shape[0], axis=0)
            feat = np.concatenate([feat, pad], axis=0)
        out[i] = feat[:top_k]
    return out


# ===================================================== backend: torchvision
class _TorchvisionBUTD:
    """Portable BUTD-flavoured extractor built from torchvision primitives.

    .. note::
       This backend uses COCO-pretrained Faster R-CNN + ImageNet-
       pretrained ResNet-101 (both from torchvision), which differ from
       the Visual-Genome-pretrained detectors used in Tables 1--2.
       Metrics are therefore numerically distinct; the canonical
       evaluation protocol uses the pre-extracted VG-BUTD tensors
       available via ``download_precomp.sh`` or ``--backend bundle``.

    Pipeline (per image):

    1. Resize the PIL image so that the shorter side is ``600`` px and
       the longer side is <=``1000`` px (the BUTD convention).
    2. Run ``fasterrcnn_resnet50_fpn_v2`` on the resized tensor.  The
       detector already performs class-aware NMS + a score threshold
       internally; we take the top ``top_k`` boxes by detection score.
    3. If fewer than ``top_k`` boxes survive, back-fill with the
       highest-scoring RPN proposals so that every image emits exactly
       ``top_k`` regions.
    4. Feed the same resized tensor (re-normalised with ImageNet stats)
       through ``resnet101(IMAGENET1K_V2)`` up to ``layer4`` to obtain
       the (1, 2048, H/32, W/32) feature map.  RoIAlign (7x7) + global-
       average-pool gives the 2048-D region vector.
    """

    _MIN_SIDE = 600
    _MAX_SIDE = 1000

    def __init__(self,
                 top_k: int = 36,
                 box_score_thresh: float = 0.05,
                 device: str = "cuda") -> None:
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.models import (resnet101, ResNet101_Weights)
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights)
        from torchvision.ops import RoIAlign

        self.torch = torch
        self.top_k = top_k
        self.device = device

        det_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.det = fasterrcnn_resnet50_fpn_v2(
            weights=det_weights,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=0.5,
            box_detections_per_img=max(top_k * 3, 100),
        ).to(device).eval()
        self.det_tfm = det_weights.transforms()

        backbone = resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2).to(device).eval()
        self.feat_stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
            backbone.layer4,
        ).to(device).eval()
        self.feat_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        self.roi_align = RoIAlign(
            output_size=(7, 7), spatial_scale=1.0 / 32.0,
            sampling_ratio=2, aligned=True).to(device)

    # --- BUTD-style deterministic resize -----------------------------
    def _butd_resize(self, pil_img):
        """Shorter side -> 600, longer side capped at 1000 (BUTD)."""
        w, h = pil_img.size
        if min(h, w) == 0:
            raise ValueError("degenerate image with a zero-length side")
        scale = self._MIN_SIDE / float(min(h, w))
        if max(h, w) * scale > self._MAX_SIDE:
            scale = self._MAX_SIDE / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        from PIL import Image
        return pil_img.resize((new_w, new_h), Image.BILINEAR)

    # --- detector scoring with RPN backfill --------------------------
    def _score_boxes(self, img_tensor) -> Tuple["torch.Tensor", "torch.Tensor"]:
        torch = self.torch
        with torch.no_grad():
            out = self.det([img_tensor])[0]
        boxes, scores = out["boxes"], out["scores"]
        if boxes.shape[0] < self.top_k:
            with torch.no_grad():
                images, _ = self.det.transform([img_tensor], None)
                feats = self.det.backbone(images.tensors)
                proposals, _ = self.det.rpn(images, feats, None)
            h_tr, w_tr = images.image_sizes[0]
            h_or, w_or = img_tensor.shape[-2:]
            prop = proposals[0].clone()
            prop[:, [0, 2]] *= (float(w_or) / float(w_tr))
            prop[:, [1, 3]] *= (float(h_or) / float(h_tr))
            min_score = (scores.min() * 0.99 if scores.numel() > 0
                         else torch.tensor(0.01, device=scores.device))
            prop_scores = torch.linspace(
                float(min_score.item()), float(min_score.item()) * 0.5,
                prop.shape[0], device=scores.device)
            boxes = torch.cat([boxes, prop], dim=0)
            scores = torch.cat([scores, prop_scores], dim=0)

        keep = torch.argsort(-scores)[: self.top_k]
        return boxes[keep], scores[keep]

    # --- one-image entry-point ---------------------------------------
    def extract_one(self, pil_img) -> np.ndarray:
        torch = self.torch
        pil_img = self._butd_resize(pil_img)
        img_t = self.det_tfm(pil_img).to(self.device)
        boxes, _ = self._score_boxes(img_t)

        feat_t = self.feat_tfm(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fmap = self.feat_stem(feat_t)                    # (1,2048,h,w)
        batch_idx = torch.zeros(boxes.shape[0], 1, device=boxes.device)
        rois = torch.cat([batch_idx, boxes], dim=1)
        pooled = self.roi_align(fmap, rois)                  # (K,2048,7,7)
        vec = pooled.mean(dim=(2, 3))                        # (K, 2048)

        if vec.shape[0] < self.top_k:
            pad = vec.mean(0, keepdim=True).expand(
                self.top_k - vec.shape[0], -1)
            vec = torch.cat([vec, pad], dim=0)
        return vec[: self.top_k].detach().cpu().numpy()


def _extract_torchvision(image_files: List[str],
                         top_k: int,
                         device: str) -> np.ndarray:
    from PIL import Image
    device = _auto_device(device)
    extractor = _TorchvisionBUTD(top_k=top_k, device=device)
    out = np.zeros((len(image_files), top_k, 2048), dtype=np.float32)
    for i, f in enumerate(image_files):
        img = Image.open(f).convert("RGB")
        out[i] = extractor.extract_one(img)
        if (i + 1) % 200 == 0:
            print(f"    torchvision BUTD {i+1}/{len(image_files)}")
    return out


# =============================================================== driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True,
                    choices=["bundle", "bottom_up_npz", "bottom_up",
                             "torchvision"],
                    help=("bundle: pre-assembled SCAN-format {split}_ims.npy "
                          "already in --bundle;\n"
                          "bottom_up_npz: one .npz per image in --images;\n"
                          "bottom_up: DEPRECATED alias for bottom_up_npz;\n"
                          "torchvision: raw JPGs -> torchvision FRCNN+R101."))
    ap.add_argument("--out", required=True,
                    help="destination precomp/ directory.")
    ap.add_argument("--bundle", default=None,
                    help="[bundle backend] source directory with "
                         "*_ims.npy / *_caps.txt to copy.")
    ap.add_argument("--images", default=None,
                    help="folder with raw JPGs (torchvision backend) or "
                         "<image_id>.npz files (bottom_up_npz backend).")
    ap.add_argument("--splits", default=None,
                    help="JSON file describing the SCAN splits "
                         "(required for bottom_up_npz / torchvision).")
    ap.add_argument("--top_k", type=int, default=36,
                    help="regions per image (BUTD default: 36).")
    ap.add_argument("--device", default="cuda",
                    help="torch device; auto-falls-back to CPU.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.backend == "bundle":
        if args.bundle is None:
            ap.error("--bundle is required for backend=bundle")
        _extract_bundle(args.bundle, args.out)
    else:
        # both bottom_up_npz and torchvision need a splits JSON
        if args.splits is None or args.images is None:
            ap.error("--splits and --images are required for backend="
                     f"{args.backend}")
        with open(args.splits, "r", encoding="utf-8") as f:
            splits: Dict[str, List[dict]] = json.load(f)

        backend = ("bottom_up_npz" if args.backend in ("bottom_up",
                                                       "bottom_up_npz")
                   else "torchvision")
        for split_name, entries in splits.items():
            image_files = [os.path.join(args.images, e["image_id"])
                           for e in entries]
            captions: List[str] = []
            for e in entries:
                captions.extend(e["captions"][:5])

            if backend == "torchvision":
                feats = _extract_torchvision(image_files,
                                             top_k=args.top_k,
                                             device=args.device)
            else:
                feats = _extract_bottom_up_npz(
                    [e["image_id"] for e in entries],
                    feat_dir=args.images,
                    top_k=args.top_k)

            _write_split(args.out, split_name, feats, captions)

    # Finally, audit the produced directory -- any layout regression
    # would otherwise only surface during training.
    try:
        from .verify_precomp import verify, _print_report
    except ImportError:                                        # pragma: no cover
        from data.verify_precomp import verify, _print_report
    reports = verify(args.out)
    _print_report(reports)
    if not all(r.ok for r in reports):
        sys.exit(2)
    print("done.")


if __name__ == "__main__":
    main()
