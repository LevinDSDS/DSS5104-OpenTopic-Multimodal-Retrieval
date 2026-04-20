"""Dataset wrapper for the SCAN precomputed Faster R-CNN features.

Expected layout (Flickr30K example)::

    $DATA_ROOT/flickr30k/precomp/
      ├── train_ims.npy      # (N_img, 36, 2048)  float32
      ├── train_caps.txt     # 5 captions per image, one per line
      ├── dev_ims.npy
      ├── dev_caps.txt
      ├── test_ims.npy
      └── test_caps.txt
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .vocab import Vocabulary


class PrecompDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 dataset: str,
                 split: str,
                 vocab: Optional[Vocabulary] = None,
                 max_len: int = 64) -> None:
        super().__init__()
        precomp = os.path.join(data_root, dataset, "precomp")
        ims_path = os.path.join(precomp, f"{split}_ims.npy")
        cap_path = os.path.join(precomp, f"{split}_caps.txt")
        if not os.path.isfile(ims_path) or not os.path.isfile(cap_path):
            raise FileNotFoundError(
                f"PrecompDataset could not locate {split}_ims.npy and/or "
                f"{split}_caps.txt under {precomp}. "
                f"Expected layout:\n"
                f"  {precomp}/\n"
                f"    \u251c\u2500\u2500 {{train,dev,test}}_ims.npy  "
                f"# (N_img, 36, 2048) float32\n"
                f"    \u2514\u2500\u2500 {{train,dev,test}}_caps.txt "
                f"# exactly 5*N_img lines\n"
                f"Three ways to populate it:\n"
                f"  (1) bash scripts/download_precomp.sh {dataset} "
                f"{data_root}\n"
                f"  (2) python -m data.extract_features --backend bundle "
                f"--bundle <your_scan_tarball_dir> --out {precomp}\n"
                f"  (3) python -m data.extract_features --backend "
                f"torchvision --images <jpgs> --splits "
                f"data/splits/{dataset}.json --out {precomp} "
                f"(uses different backbone pre-training from Tables 1-2).\n"
                f"Run 'python -m data.verify_precomp --precomp {precomp}' "
                f"to audit what you have.")
        self.images = np.load(ims_path, mmap_mode="r")
        if self.images.ndim != 3:
            raise ValueError(
                f"{ims_path}: expected a 3-D (N, K, D) tensor, got "
                f"shape={tuple(self.images.shape)}. Run "
                f"'python -m data.verify_precomp --precomp {precomp}' "
                f"for details.")
        if self.images.shape[-1] not in (512, 768, 1024, 2048):
            # common hidden sizes: BUTD (2048), CLIP ViT-L / BLIP-large
            # (1024), CLIP ViT-B/16 / ViT-B/32 (768), CLIP projection head
            # (512). Anything else is almost certainly a wrong bundle.
            raise ValueError(
                f"{ims_path}: hidden dim {self.images.shape[-1]} is not "
                f"a common vision backbone size (expected 512/768/1024/"
                f"2048). Check your bundle, and make sure the config's "
                f"`model.img_in_dim` matches. Run "
                f"'python -m data.verify_precomp --precomp {precomp} "
                f"--expected-dim <your_dim>' for a full audit.")
        with open(cap_path, encoding="utf-8") as f:
            self.captions = [c.strip() for c in f.readlines() if c.strip()]
        if len(self.captions) % self.images.shape[0] != 0:
            raise ValueError(
                f"{cap_path}: got {len(self.captions)} captions for "
                f"{self.images.shape[0]} images -- not a multiple. "
                f"Expected exactly 5 captions per image (SCAN convention).")
        self.cap_per_img = len(self.captions) // self.images.shape[0]
        if self.cap_per_img != 5:
            raise ValueError(
                f"expected 5 captions per image (SCAN convention), "
                f"got {self.cap_per_img} at {precomp}.")
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int):
        img_idx = idx // self.cap_per_img
        regions = torch.from_numpy(np.array(self.images[img_idx])).float()
        cap = self.captions[idx]
        if self.vocab is not None:
            cap_ids = torch.tensor(self.vocab.encode(cap, self.max_len),
                                   dtype=torch.long)
        else:
            cap_ids = cap                              # raw string for BERT
        return regions, cap_ids, img_idx


def _pad(seqs, pad_id: int = 0):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : lengths[i]] = s
    return out, torch.tensor(lengths, dtype=torch.long)


def collate_bigru(batch):
    regions = torch.stack([b[0] for b in batch], dim=0)
    cap_ids, lengths = _pad([b[1] for b in batch])
    img_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    text_args = {"token_ids": cap_ids, "lengths": lengths}
    return regions, text_args, img_idx


def collate_bert(batch, tokenizer, max_len: int = 64):
    regions = torch.stack([b[0] for b in batch], dim=0)
    enc = tokenizer([b[1] for b in batch],
                    padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    img_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    text_args = {"token_ids": enc["input_ids"],
                 "attn_mask": enc["attention_mask"]}
    return regions, text_args, img_idx


def get_loaders(cfg, vocab: Optional[Vocabulary] = None,
                tokenizer=None):
    train_set = PrecompDataset(cfg["data_root"], cfg["dataset"], "train",
                               vocab=vocab)
    dev_set   = PrecompDataset(cfg["data_root"], cfg["dataset"], "dev",
                               vocab=vocab)
    test_set  = PrecompDataset(cfg["data_root"], cfg["dataset"], "test",
                               vocab=vocab)
    if cfg["text_encoder"] == "bigru":
        coll = collate_bigru
    else:
        coll = lambda b: collate_bert(b, tokenizer, cfg.get("max_len", 64))
    nw = cfg.get("num_workers", 4)
    bs = cfg["batch_size"]
    loaders = (
        DataLoader(train_set, batch_size=bs, shuffle=True,
                   num_workers=nw, collate_fn=coll),
        DataLoader(dev_set,  batch_size=bs, shuffle=False,
                   num_workers=nw, collate_fn=coll),
        DataLoader(test_set, batch_size=bs, shuffle=False,
                   num_workers=nw, collate_fn=coll),
    )

    testall_path = os.path.join(cfg["data_root"], cfg["dataset"],
                                "precomp", "testall_ims.npy")
    if os.path.isfile(testall_path):
        testall_set = PrecompDataset(cfg["data_root"], cfg["dataset"],
                                     "testall", vocab=vocab)
        testall_loader = DataLoader(testall_set, batch_size=bs,
                                    shuffle=False, num_workers=nw,
                                    collate_fn=coll)
        return loaders + (testall_loader,)
    return loaders
