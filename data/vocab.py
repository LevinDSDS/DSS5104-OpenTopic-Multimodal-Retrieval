"""Tiny vocabulary for the Bi-GRU branch.

Skipped automatically when ``text_encoder = bert`` because BERT
brings its own WordPiece tokenizer.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Iterable, List

try:
    import nltk
    _HAVE_NLTK = True
except ImportError:                                    # pragma: no cover
    _HAVE_NLTK = False


_SIMPLE_TOK = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    """Tokenise a sentence.

    Prefer NLTK's Punkt tokenizer when both the library and the
    ``punkt_tab`` / ``punkt`` resource are available (same tokens the
    SCAN codebase uses). Otherwise fall back to a pure-regex word
    splitter so that the toy smoke test and CI do not require any
    NLTK data downloads.
    """
    if _HAVE_NLTK:
        try:
            return nltk.word_tokenize(text.lower())
        except LookupError:                            # resource missing
            pass
    return _SIMPLE_TOK.findall(text.lower())


class Vocabulary:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        for tok in (self.PAD, self.BOS, self.EOS, self.UNK):
            self.add(tok)

    def add(self, word: str) -> None:
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode(self, sentence: str, max_len: int = 64) -> List[int]:
        toks = _tokenize(sentence)
        ids = [self.word2idx.get(self.BOS)]
        for t in toks[: max_len - 2]:
            ids.append(self.word2idx.get(t, self.word2idx[self.UNK]))
        ids.append(self.word2idx[self.EOS])
        return ids

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.idx2word, f, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "Vocabulary":
        v = cls()
        with open(path, "r", encoding="utf-8") as f:
            words = json.load(f)
        v.word2idx, v.idx2word = {}, []
        for w in words:
            v.add(w)
        return v


def build_or_load_vocab(path: str,
                        sentences: Iterable[str] | None = None,
                        min_freq: int = 4) -> Vocabulary:
    if os.path.exists(path):
        return Vocabulary.from_json(path)
    if sentences is None:
        raise FileNotFoundError(f"vocab not found at {path} and no "
                                "sentences provided to build one")
    counter: Counter[str] = Counter()
    for s in sentences:
        counter.update(_tokenize(s))
    vocab = Vocabulary()
    for w, c in counter.most_common():
        if c < min_freq:
            break
        vocab.add(w)
    vocab.to_json(path)
    return vocab
