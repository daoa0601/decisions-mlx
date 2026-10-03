"""SemIf's option-letter prompt and the many-option readout, with no model dependencies.

Ported from ``jevk5/prompt.py`` in JevK5 (allebee/jevk5, Apache-2.0), which follows SemIf
(TheoLeeCJ/SemIf, MIT). The decision goes to the model as JSON with lettered options; the answer is
a softmax over the letters' next-token logits. Up to 16 options take one read; more are read in
groups and combined by :func:`spread`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

LETTERS = "ABCDEFGHIJKLMNOP"
METHODS = ("knockout", "tree")
# Sharpening of the combined distribution over more than 16 options, per method, when the model
# does not carry its own (JevK5's ``knockout_temperature``).
TEMPERATURES = {"knockout": 0.77, "tree": 1.0}
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def messages(state: Any, criterion: str, options: list[str]) -> list[dict[str, str]]:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def decision_options(question: dict[str, Any]) -> list[tuple[str, str]]:
    """(option id, option text) for a typed question: noul -> true/false, choice -> its
    criteria, score -> level indices. Texts are "id: description", as SemIf's JevBench mapping."""
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


Reader = Callable[[list[str]], Sequence[float]]


def groups(n: int, count: int) -> list[range]:
    """`count` contiguous runs covering range(n), their sizes differing by at most one."""
    base, extra = divmod(n, count)
    runs, start = [], 0
    for g in range(count):
        stop = start + base + (g < extra)
        runs.append(range(start, stop))
        start = stop
    return runs


def spread(
    read: Reader, texts: list[str], method: str = "knockout", temperature: float | None = None
) -> list[float]:
    """A probability for every option, from a reader that answers at most 16 lettered options.

    Up to 16 options the result is ``read(texts)`` itself. Beyond that, "knockout" reads
    ceil(n / 16) near-equal groups and then a final of the groups' best options; "tree" reads one
    pass whose letters stand for whole groups, then one pass per group. The combined distribution
    is then sharpened by ``temperature`` (default: TEMPERATURES).
    """
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    probs = _combine(read, texts, method)
    temperature = TEMPERATURES[method] if temperature is None else temperature
    if temperature != 1.0:
        probs = [q ** (1 / temperature) for q in probs]
        total = sum(probs)
        probs = [q / total for q in probs]
    return probs


def _combine(read: Reader, texts: list[str], method: str) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    if method == "knockout":
        weights = _knockout(read, texts)
    elif method == "tree":
        weights = _tree(read, texts)
    else:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    total = sum(weights)
    return [w / total for w in weights]


def _knockout(read: Reader, texts: list[str]) -> list[float]:
    runs = groups(len(texts), -(-len(texts) // len(LETTERS)))
    inner = [list(read([texts[i] for i in run])) for run in runs]
    inner = [[q / sum(p) for q in p] for p in inner]
    keep = max(1, len(LETTERS) // len(runs))
    # Ties go to the earlier option: the sorts are stable and walk the options in order.
    ranked = [sorted(range(len(p)), key=lambda j: -p[j]) for p in inner]
    chosen = {(g, j) for g, order in enumerate(ranked) for j in order[:keep]}
    rest = sorted(
        ((g, j) for g, order in enumerate(ranked) for j in order[keep:]),
        key=lambda gj: -inner[gj[0]][gj[1]],
    )
    chosen.update(rest[: max(0, len(LETTERS) - len(chosen))])
    tops = [sorted(j for h, j in chosen if h == g) for g in range(len(runs))]
    final = _combine(read, [texts[run[j]] for run, top in zip(runs, tops) for j in top], "knockout")
    shares, at = [], 0
    for top in tops:
        shares.append(dict(zip(top, final[at : at + len(top)])))
        at += len(top)
    in_final = sum(sum(f.values()) * sum(p[j] for j in f) for p, f in zip(inner, shares))
    weights = []
    for p, f in zip(inner, shares):
        mass = sum(f.values())
        weights += [f[j] * in_final if j in f else mass * q for j, q in enumerate(p)]
    return weights


def _tree(read: Reader, texts: list[str]) -> list[float]:
    runs = groups(len(texts), min(len(LETTERS), -(-len(texts) // len(LETTERS))))
    outer = read(["One of: " + "; ".join(texts[i] for i in run) for run in runs])
    weights = []
    for run, share in zip(runs, outer):
        weights += [share * q for q in _combine(read, [texts[i] for i in run], "tree")]
    return weights
