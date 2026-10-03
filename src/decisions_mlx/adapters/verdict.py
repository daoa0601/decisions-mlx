"""Verdict: a GLiClass uni-encoder over ModernBERT that scores label tokens against the text.

Each question is one sequence: ``<<LABEL>>label0<<LABEL>>label1 ... <<SEP>>text``, with an
"insufficient evidence" label appended. The first token's state, projected, is dotted with each
label token's projected state; the logits are divided by a temperature fitted per label count,
and the abstention label's share is dropped.

The prompt and temperatures follow OpenJev's ``VerdictEngine`` (razorback16/openjev,
Apache-2.0), itself from Verdict's v1.4 inference (Heman10x-NGU/Verdict-open-jev); the model is
gliclass's ``GLiClassUniEncoder`` with the "first" pooling and the "simple" (dot) scorer, which
is what the ``heman10x/rlcd-modernbert-151m`` checkpoint uses.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from ..api import Decision, Request, RequestError
from .modernbert import ModernBert

LABEL_MARKER, SEP_MARKER = "<<LABEL>>", "<<SEP>>"
ABSTAIN = "insufficient evidence"
MAX_LEN = 512
# The GLiClass variant this port implements; other configurations fail at load.
EXPECTED = {
    "architecture_type": "uni-encoder",
    "pooling_strategy": "first",
    "scorer_type": "simple",
    "class_token_pooling": "first",
    "embed_class_token": True,
    "normalize_features": False,
    "extract_text_features": False,
    "use_lstm": False,
    "squeeze_layers": False,
    "use_segment_embeddings": False,
    "encoder_layer_id": -1,
}


def prompt(kind: str, instructions: str, choices: list[tuple[str, str]], context: str) -> tuple[str, int]:
    """The model input for one question, and its label count (the caller's options plus abstention)."""
    if kind == "noul":
        labels = [f"true: {instructions}", f"false: not {instructions}"]
        text = f"Context:\n{context}\n\nEvaluate proposition: {instructions}"
    else:
        if kind == "choice":
            labels = [f"It is {desc or name}" for name, desc in choices]
        else:
            labels = [f"{desc} (Value: {float(i)})" for i, (_, desc) in enumerate(choices)]
        text = f"Question: {instructions}\n\nContext:\n{context}" if instructions else context
    labels.append(ABSTAIN)
    return "".join(LABEL_MARKER + label for label in labels) + SEP_MARKER + text, len(labels)


def text_of(value: Any) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class Projector(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(width, hidden)
        self.linear_2 = nn.Linear(hidden, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.gelu(self.linear_1(x)))


class GLiClass(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        width = config["encoder_config"]["hidden_size"]
        self.class_token = config["class_token_index"]
        self.encoder_model = ModernBert(config["encoder_config"])
        self.text_projector = Projector(width, config["hidden_size"])
        self.classes_projector = Projector(width, config["hidden_size"])

    def __call__(self, input_ids: list[int]) -> mx.array:
        """One logit per label token, in order."""
        h = self.encoder_model(mx.array([input_ids]))[0]
        labels = mx.array([i for i, t in enumerate(input_ids) if t == self.class_token])
        return self.classes_projector(h[labels]) @ self.text_projector(h[0])


class VerdictDecider:
    def __init__(self, path: Path, dtype: mx.Dtype = mx.float32) -> None:
        config = json.loads((path / "config.json").read_text())
        mismatched = {k: config.get(k) for k, v in EXPECTED.items() if config.get(k, v) != v}
        if mismatched:
            raise ValueError(f"unsupported GLiClass configuration: {mismatched}")
        self.max_choices = config["max_num_classes"] - 1  # one label is kept for abstention
        calibrator = json.loads((path / "calibrator.json").read_text())
        self.temperature = float(calibrator["temperature"])
        self.per_k = {int(k): float(v) for k, v in calibrator.get("per_k", {}).items()}
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = GLiClass(config)
        weights = {
            k.removeprefix("model."): v.astype(dtype)
            for k, v in mx.load(str(path / "model.safetensors")).items()
            if k != "model.logit_scale"  # only applied with normalize_features
        }
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()

    def question(self, context: str, question_id: str, question: dict[str, Any]) -> tuple[list[float], int]:
        kind, criteria = question["type"], question["criteria"]
        instructions = text_of(question.get("instructions")) or question_id
        if kind == "choice":
            if len(criteria) > self.max_choices:
                raise RequestError(f"{question_id}: at most {self.max_choices} choices")
            choices = [(name, text_of(desc)) for name, desc in criteria.items()]
        elif kind == "score":
            if len(criteria) > self.max_choices:
                raise RequestError(f"{question_id}: at most {self.max_choices} score levels")
            choices = [(str(i), text_of(c)) for i, c in enumerate(criteria)]
        else:
            choices = []
        text, k = prompt(kind, instructions, choices, context)
        ids = self.tokenizer(text, truncation=True, max_length=MAX_LEN)["input_ids"]
        logits = self.model(ids)[:k].astype(mx.float32)
        p = mx.softmax(logits / self.per_k.get(k, self.temperature))[:-1].tolist()
        total = sum(p)
        # the caller's options only: the abstention share is dropped
        p = [v / total for v in p] if math.isfinite(total) and total > 0 else [1.0 / (k - 1)] * (k - 1)
        return p, len(ids)

    def decide(self, request: Request) -> Decision:
        if request.images:
            raise RequestError("this model reads text only; images are not supported")
        state = request.state
        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        probabilities, tokens = {}, 0
        for qid, question in request.questions.items():
            p, used = self.question(context, qid, question)
            tokens += used
            if question["type"] == "noul":
                probabilities[qid] = {"true": p[0], "false": p[1]}
            elif question["type"] == "choice":
                probabilities[qid] = dict(zip(question["criteria"], p))
            else:
                probabilities[qid] = {str(i): v for i, v in enumerate(p)}
        return Decision(probabilities, tokens)
