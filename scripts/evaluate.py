"""Zero-shot accuracy and calibration of any adapter on a labelled task, through /v1/systemone requests.

Every item becomes one request with one ``choice`` question over all the task's labels, so each
model sees exactly the same input.

    uv run python scripts/evaluate.py --model alibiserikbay/JevK5 --task banking77 --limit 200
    uv run python scripts/evaluate.py --model ../clef-mlx/clef-flash-8bit --task banking77 --limit 200
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from decisions_mlx import load, parse_request

TASKS = {
    "banking77": {
        "repo": "mteb/banking77",
        "file": "test.jsonl",
        "text": "text",
        "label": "label_text",
        "instructions": "Which banking support intent does the customer message express?",
    },
}


def items(task: dict, limit: int, seed: int) -> tuple[list[dict], list[str]]:
    path = hf_hub_download(task["repo"], task["file"], repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    labels = sorted({row[task["label"]] for row in rows})
    random.Random(seed).shuffle(rows)
    return rows[:limit], labels


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    gold, predicted = Counter(g for g, _ in pairs), Counter(p for _, p in pairs)
    hits = Counter(g for g, p in pairs if g == p)
    scores = []
    for label in set(gold) | set(predicted):
        precision = hits[label] / predicted[label] if predicted[label] else 0.0
        recall = hits[label] / gold[label] if gold[label] else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def ece(confidences: list[float], correct: list[bool], bins: int = 15) -> float:
    """Top-label expected calibration error over equal-width confidence bins."""
    total = 0.0
    for b in range(bins):
        members = [i for i, c in enumerate(confidences) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if members:
            gap = sum(confidences[i] for i in members) / len(members) - sum(correct[i] for i in members) / len(members)
            total += len(members) * abs(gap)
    return total / len(confidences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--task", choices=sorted(TASKS), default="banking77")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write per-item predictions here (JSONL)")
    args = parser.parse_args()

    task = TASKS[args.task]
    rows, labels = items(task, args.limit, args.seed)
    decider = load(args.model, args.adapter)
    pairs, confidences, correct, brier, seconds, records = [], [], [], [], [], []
    for row in rows:
        request = parse_request(
            {
                "model": "eval",
                "state": row[task["text"]],
                "questions": {"label": {"type": "choice", "instructions": task["instructions"], "criteria": labels}},
            }
        )
        start = time.perf_counter()
        probabilities = decider.decide(request).probabilities["label"]
        seconds.append(time.perf_counter() - start)
        gold = row[task["label"]]
        predicted = max(probabilities, key=probabilities.get)
        pairs.append((gold, predicted))
        confidences.append(probabilities[predicted])
        correct.append(gold == predicted)
        brier.append(sum((p - (label == gold)) ** 2 for label, p in probabilities.items()))
        records.append({"text": row[task["text"]], "gold": gold, "predicted": predicted, "p_gold": probabilities[gold]})

    summary = {
        "model": args.model,
        "task": args.task,
        "items": len(rows),
        "options": len(labels),
        "accuracy": round(sum(correct) / len(correct), 4),
        "macro_f1": round(macro_f1(pairs), 4),
        "brier": round(statistics.fmean(brier), 4),
        "ece": round(ece(confidences, correct), 4),
        "median_seconds": round(statistics.median(seconds[1:] or seconds), 3),
    }
    if args.out:
        args.out.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
