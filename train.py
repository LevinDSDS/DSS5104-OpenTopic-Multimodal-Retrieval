"""AITR training entry-point.

Usage::

    python train.py --config configs/flickr30k_bert.yaml

Implements the factorised training-time score (paper Eq.~train):

    L = TripletRanking(lambda1 * S_ini + lambda2 * S_ins)
        - lambda3 * mean_b S_fra(I_b, T_b)

The (B, B) similarity matrix is built directly from the model's
forward output (no proxy), which keeps positives and negatives on the
same scoring function.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

from aitr import AITR, AITRConfig, TripletRankingLoss
from data import build_or_load_vocab, get_loaders
from eval import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, vocab_size: int) -> AITR:
    m = cfg["model"]
    config = AITRConfig(
        img_in_dim=m["img_in_dim"],
        embed_dim=m["embed_dim"],
        text_encoder=cfg["text_encoder"],
        vocab_size=vocab_size,
        bert_name=cfg.get("bert_name", "bert-base-uncased"),
        n_proto=m["n_proto"],
        tau=m["tau"],
        csa_windows=tuple(m["csa_windows"]),
        csa_strides=tuple(m["csa_strides"]),
        csa_alpha=m["csa_alpha"],
        csa_top_pairs=m["csa_top_pairs"],
        z_balanced=m["z_balanced"],
        z_unbalanced=m["z_unbalanced"],
        lambdas=tuple(m["lambdas"]),
        eval_chunk=m.get("eval_chunk", 128),
    )
    return AITR(config)


def build_text_aux(cfg):
    if cfg["text_encoder"] == "bigru":
        cap_path = os.path.join(cfg["data_root"], cfg["dataset"],
                                "precomp", "train_caps.txt")
        sentences = None
        if not os.path.isfile(cfg["vocab_path"]):
            with open(cap_path, encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
        vocab = build_or_load_vocab(cfg["vocab_path"], sentences=sentences)
        return vocab, None
    from transformers import BertTokenizerFast
    return None, BertTokenizerFast.from_pretrained(cfg["bert_name"])


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["logging"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["logging"]["ckpt_dir"], exist_ok=True)

    vocab, tokenizer = build_text_aux(cfg)
    loaders = get_loaders(cfg, vocab, tokenizer)
    train_loader, dev_loader, test_loader = loaders[:3]
    testall_loader = loaders[3] if len(loaders) > 3 else None

    model = build_model(cfg, vocab_size=(len(vocab) if vocab else 0)).to(device)
    criterion = TripletRankingLoss(margin=cfg["optim"]["margin"]).to(device)
    optim = torch.optim.AdamW(model.parameters(),
                              lr=cfg["optim"]["lr"],
                              weight_decay=cfg["optim"]["weight_decay"])

    l1, l2, l3 = cfg["model"]["lambdas"]

    best_rsum = -1.0
    for epoch in range(cfg["optim"]["epochs"]):
        if epoch == cfg["optim"]["lr_decay_epoch"]:
            for pg in optim.param_groups:
                pg["lr"] *= cfg["optim"]["lr_decay_factor"]

        model.train()
        t0 = time.time()
        for step, (regions, text_args, _) in enumerate(
                tqdm(train_loader, desc=f"epoch {epoch}")):
            regions = regions.to(device)
            text_args = {k: v.to(device) for k, v in text_args.items()}

            out = model(regions, text_args)

            sim_matrix = l1 * out["s_ini_mat"] + l2 * out["s_ins_mat"]
            loss_tri = criterion(sim_matrix)
            loss_aux = -l3 * out["s_fra_diag"].mean()
            loss = loss_tri + loss_aux

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["optim"]["grad_clip"])
            optim.step()

            if step % cfg["logging"]["log_every"] == 0:
                print(f"  step {step:4d}  loss {loss.item():.4f} "
                      f"(tri {loss_tri.item():.4f}, aux {loss_aux.item():.4f})")

        rsum = evaluate(model, dev_loader, device)
        elapsed = time.time() - t0
        print(f"epoch {epoch}  dev rSum = {rsum:.2f}  ({elapsed:.0f}s)")

        if rsum > best_rsum:
            best_rsum = rsum
            torch.save({"model": model.state_dict(),
                        "config": cfg, "epoch": epoch, "rsum": rsum},
                       os.path.join(cfg["logging"]["ckpt_dir"], "best.ckpt"))
            print(f"  * new best (rSum {rsum:.2f}) saved.")

    print("Training done.  Best dev rSum =", best_rsum)

    if testall_loader is not None:
        ckpt = torch.load(
            os.path.join(cfg["logging"]["ckpt_dir"], "best.ckpt"),
            map_location=device)
        model.load_state_dict(ckpt["model"])
        print("--- COCO 5K (testall) evaluation ---")
        evaluate(model, testall_loader, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
