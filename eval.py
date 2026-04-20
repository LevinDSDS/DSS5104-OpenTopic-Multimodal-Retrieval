"""AITR evaluation entry-point.

Computes Recall@K and rSum on the precomputed test split using the
honest convex-combination score (paper Eq.~final) with chunked
fragment-level cross-attention.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from aitr import AITR, AITRConfig


@torch.no_grad()
def encode_split(model: AITR, loader, device) -> tuple[torch.Tensor,
                                                       torch.Tensor,
                                                       torch.Tensor]:
    """Cache the (token-level) image and text features for the whole split."""
    img_chunks, txt_chunks, idx_chunks = [], [], []
    for regions, text_args, img_idx in loader:
        regions = regions.to(device)
        text_args = {k: v.to(device) for k, v in text_args.items()}
        v_tok = model.encode_image(regions)               # (B, R, d)
        t_tok = model.encode_text(**text_args)            # (B, U, d)
        img_chunks.append(v_tok.cpu())
        txt_chunks.append(t_tok.cpu())
        idx_chunks.append(img_idx)
    return (torch.cat(img_chunks, dim=0),
            torch.cat(txt_chunks, dim=0),
            torch.cat(idx_chunks, dim=0))


def recall_at_k(sim_matrix: np.ndarray, k_list=(1, 5, 10),
                cap_per_img: int = 5) -> dict:
    """Compute Recall@K for both directions and rSum.

    sim_matrix shape: (N_img, N_txt) with N_txt = N_img * cap_per_img.
    """
    N_img = sim_matrix.shape[0]
    out = {}

    rank_i2t = np.zeros(N_img, dtype=int)
    for i in range(N_img):
        order = np.argsort(-sim_matrix[i])
        positive = set(range(i * cap_per_img, (i + 1) * cap_per_img))
        rank = N_img * cap_per_img
        for r, idx in enumerate(order):
            if int(idx) in positive:
                rank = r
                break
        rank_i2t[i] = rank

    sim_t2i = sim_matrix.T
    N_txt = sim_t2i.shape[0]
    rank_t2i = np.zeros(N_txt, dtype=int)
    for j in range(N_txt):
        order = np.argsort(-sim_t2i[j])
        positive = j // cap_per_img
        rank_t2i[j] = int(np.where(order == positive)[0][0])

    for k in k_list:
        out[f"i2t_R@{k}"] = 100.0 * (rank_i2t < k).mean()
        out[f"t2i_R@{k}"] = 100.0 * (rank_t2i < k).mean()
    out["rSum"] = sum(out[f"i2t_R@{k}"] + out[f"t2i_R@{k}"] for k in k_list)
    return out


@torch.no_grad()
def evaluate(model: AITR, loader, device,
             cap_per_img: int = 5,
             chunk: int | None = None) -> float:
    """Run a full evaluation pass and print per-direction Recall@K."""
    model.eval()
    v_tok_all, t_tok_all, _ = encode_split(model, loader, device)

    # de-duplicate images: each image is paired with cap_per_img captions
    v_unique = v_tok_all[::cap_per_img].to(device)
    t_all = t_tok_all.to(device)

    sim = model.pairwise_similarity(v_unique, t_all, chunk=chunk).cpu().numpy()
    metrics = recall_at_k(sim, cap_per_img=cap_per_img)
    print(" | ".join(f"{k}: {v:.2f}" for k, v in metrics.items()))
    return float(metrics["rSum"])


def _build_model_from_ckpt(ckpt_path: str) -> tuple[AITR, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = payload["config"]
    cfg_obj = AITRConfig(
        img_in_dim=cfg["model"]["img_in_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        text_encoder=cfg["text_encoder"],
        n_proto=cfg["model"]["n_proto"],
        tau=cfg["model"]["tau"],
        csa_windows=tuple(cfg["model"]["csa_windows"]),
        csa_strides=tuple(cfg["model"]["csa_strides"]),
        csa_alpha=cfg["model"]["csa_alpha"],
        csa_top_pairs=cfg["model"]["csa_top_pairs"],
        z_balanced=cfg["model"]["z_balanced"],
        z_unbalanced=cfg["model"]["z_unbalanced"],
        lambdas=tuple(cfg["model"]["lambdas"]),
        eval_chunk=cfg["model"].get("eval_chunk", 128),
    )
    model = AITR(cfg_obj)
    model.load_state_dict(payload["model"])
    return model, cfg


def main(ckpt_path: str) -> None:
    from data import build_or_load_vocab, get_loaders

    model, cfg = _build_model_from_ckpt(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if cfg["text_encoder"] == "bigru":
        vocab = build_or_load_vocab(cfg["vocab_path"])
        loaders = get_loaders(cfg, vocab, None)
    else:
        from transformers import BertTokenizerFast
        tok = BertTokenizerFast.from_pretrained(cfg["bert_name"])
        loaders = get_loaders(cfg, None, tok)

    test_loader = loaders[2]
    print("--- 1K test ---")
    evaluate(model, test_loader, device)

    if len(loaders) > 3:
        print("--- 5K testall ---")
        evaluate(model, loaders[3], device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    main(p.parse_args().ckpt)
