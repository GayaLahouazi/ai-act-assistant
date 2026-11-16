"""Okapi BM25 with a tokenizer adapted to French legal text."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np

STOPWORDS = frozenset(
    """a au aux avec ce ces cet cette dans de des du elle en est et il ils la le les leur leurs lui ne ni nous on ou
    par pas pour qu que quel quelle quelles quels qui sa se ses son sont sur un une vous y dont ainsi etre ete
    quoi comment quand selon entre plus tout tous toute toutes si afin lorsque""".split()
)
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, drop stopwords and elided articles (l', d'), strip plural marks."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = []
    for tok in TOKEN.findall(text):
        if tok in STOPWORDS or (len(tok) == 1 and not tok.isdigit()):
            continue
        if len(tok) > 3 and tok[-1] in "sx":  # crude plural stemming: systemes -> systeme
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        tokenized = [tokenize(d) for d in documents]
        self.lengths = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(tokenized) else 0.0
        self.n = len(tokenized)
        # Inverted index: term -> (doc ids, term frequencies)
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, tokens in enumerate(tokenized):
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, []).append((doc_id, tf))
        self.postings = {
            term: (np.array([d for d, _ in plist]), np.array([tf for _, tf in plist], dtype=np.float32))
            for term, plist in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            docs, tf = self.postings[term]
            idf = math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[docs] / self.avg_len)
            scores[docs] += idf * tf * (self.k1 + 1) / norm
        return scores
