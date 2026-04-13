from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def train(path: Path) -> dict[str, Counter[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    model: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for token in _tokens(row["text"]):
            model[row["label"]][token] += 1
    return model


def predict(model: dict[str, Counter[str]], text: str) -> str:
    toks = _tokens(text)
    scores = {}
    for label, counts in model.items():
        scores[label] = sum(counts[t] for t in toks)
    return max(scores, key=scores.get)


def predict_with_score(model: dict[str, Counter[str]], text: str) -> tuple[str, int]:
    toks = _tokens(text)
    scores = {}
    for label, counts in model.items():
        scores[label] = sum(counts[t] for t in toks)
    best = max(scores, key=scores.get)
    return best, scores[best]


def evaluate(model: dict[str, Counter[str]], dataset: list[dict[str, str]]) -> float:
    if not dataset:
        return 0.0
    correct = 0
    for row in dataset:
        if predict(model, row["text"]) == row["label"]:
            correct += 1
    return correct / len(dataset)


def label_counts(dataset: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in dataset:
        counts[row["label"]] += 1
    return dict(counts)
