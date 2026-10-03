"""Laya: a ModernBERT encoder whose head scores a [MASK] marker in front of every option.

Each question is its own sequence, ``[CLS] <type> question: <instructions> [SEP] [MASK] opt0
[MASK] opt1 ... [SEP] <state> [SEP]``. A 2-layer transformer head over the encoder states (plus a
question-type embedding) feeds a scorer at each marker, and the logits are divided by a
temperature fitted per question type and option count.

Ported from Laya (NandhaKishorM/laya, Apache-2.0: ``laya/common.py`` and ``laya/agent.py``) for
the ``convaiinnovations/laya-typed-decisions`` checkpoint. Laya's act head, hooks, custom noul
labels, option shuffling and long-state windowing are not ported; a state longer than the
sequence is cut, as Laya's ``system_one`` does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from ..api import Decision, Request, RequestError
from .modernbert import ModernBert

QTYPES = {"choice": 0, "score": 1, "noul": 2}
TEMP_MIN, TEMP_MAX = 0.5, 5.0  # Laya refuses fitted temperatures outside this range
OPTION_TOKENS = 48


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(kind: str, criteria: Any) -> list[str]:
    """Option texts in label order; a noul question is always [false, true]."""
    if kind == "choice":
        return [str(k) if v is None or v == "" else f"{k}: {render_criterion(v)}" for k, v in criteria.items()]
    if kind == "score":
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(criteria)]
    criteria = criteria or {}
    false, true = criteria.get("false"), criteria.get("true")
    return [
        "false: " + (render_criterion(false) if false not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true) if true not in (None, "") else "yes, the statement holds"),
    ]


def temp_bucket(kind: str, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{kind}:{size}"


def clamp_temperature(t: Any) -> float:
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def encode(
    tok: Any, state: Any, kind: str, instructions: str, criteria: Any, max_len: int, head_max_len: int
) -> tuple[list[int], list[int]]:
    """One question's sequence and the [MASK] position of each option (Laya's ``build_sequence``)."""

    def tokens(text: str, **kwargs: Any) -> list[int]:
        return tok(text, add_special_tokens=False, **kwargs)["input_ids"]

    mask = tok.mask_token
    options = [
        [tok.mask_token_id] + tokens(" " + text.replace(mask, " "), truncation=True, max_length=OPTION_TOKENS)
        for text in render_options(kind, criteria)
    ]
    budget = head_max_len - sum(len(o) for o in options)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(options)))
        options = [o[:per] for o in options]
        budget = head_max_len - sum(len(o) for o in options)
    head = tokens(f"{kind} question: {instructions.replace(mask, ' ')}")[: max(8, budget)]
    ids = [tok.cls_token_id] + head + [tok.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tok.sep_token_id)
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    state_ids = tokens(text.replace(mask, " "))
    room = max(0, max_len - len(ids) - 1)
    # a conversation (a list state) keeps its newest turns
    state_ids = state_ids[max(0, len(state_ids) - room) :] if isinstance(state, list) else state_ids[:room]
    ids = (ids + state_ids + [tok.sep_token_id])[:max_len]
    markers = [m for m in markers if m < max_len]
    if len(markers) != len(options):
        raise RequestError(f"only {len(markers)} of {len(options)} options fit in {max_len} tokens")
    return ids, markers


class TorchMultiheadAttention(nn.Module):
    """``torch.nn.MultiheadAttention`` self-attention (batch_first, packed in-projection), unmasked."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, width = x.shape
        qkv = (x @ self.in_proj_weight.T + self.in_proj_bias).reshape(batch, length, 3, self.heads, -1)
        q, k, v = qkv.transpose(2, 0, 3, 1, 4)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, width))


class HeadLayer(nn.Module):
    """``torch.nn.TransformerEncoderLayer`` with ``norm_first=True`` and its default ReLU."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = TorchMultiheadAttention(width, max(1, width // 64))
        self.linear1 = nn.Linear(width, 4 * width)
        self.linear2 = nn.Linear(4 * width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x))
        return x + self.linear2(nn.relu(self.linear1(self.norm2(x))))


class LayaModel(nn.Module):
    def __init__(self, encoder_config: dict[str, Any], head_layers: int) -> None:
        super().__init__()
        width = encoder_config["hidden_size"]
        self.encoder = ModernBert(encoder_config)
        self.head = nn.Module()
        self.head.layers = [HeadLayer(width) for _ in range(head_layers)]
        self.type_emb = nn.Embedding(3, width)
        self.scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def __call__(self, input_ids: mx.array, markers: mx.array, qtype: int) -> mx.array:
        """Per-option logits for one sequence."""
        h = self.encoder(input_ids[None]) + self.type_emb(mx.array([qtype]))[:, None, :]
        for layer in self.head.layers:
            h = layer(h)
        return self.scorer(h[0, markers]).squeeze(-1).astype(mx.float32)


class LayaDecider:
    def __init__(self, path: Path, dtype: mx.Dtype = mx.float32) -> None:
        self.cfg = json.loads((path / "rl_agent_config.json").read_text())
        encoder_config = json.loads((path / "encoder" / "config.json").read_text())
        self.max_len = self.cfg.get("max_len", 512)
        self.head_max_len = self.cfg.get("head_max_len", 192)
        self.temperature = [clamp_temperature(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {
            k: clamp_temperature(v) for k, v in self.cfg.get("temperature_by_options", {}).items()
        }
        self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        self.model = LayaModel(encoder_config, self.cfg.get("head_layers", 2))
        weights = {
            re.sub(r"^scorer\.(\d+)\.", r"scorer.layers.\1.", k): v.astype(dtype)
            for k, v in mx.load(str(path / "model.safetensors")).items()
            if not k.startswith(("act_head.", "temperature"))
        }
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()

    def encode(self, state: Any, kind: str, instructions: str, criteria: Any) -> tuple[list[int], list[int]]:
        return encode(self.tokenizer, state, kind, instructions, criteria, self.max_len, self.head_max_len)

    def question(self, state: Any, question_id: str, question: dict[str, Any]) -> tuple[list[float], int]:
        kind = question["type"]
        instructions = question.get("instructions") or question_id
        ids, markers = self.encode(state, kind, instructions, question["criteria"])
        logits = self.model(mx.array(ids), mx.array(markers), QTYPES[kind])
        k = len(markers)
        t = self.temperature_by_options.get(temp_bucket(kind, k), self.temperature[QTYPES[kind]])
        return mx.softmax(logits / t).tolist(), len(ids)

    def decide(self, request: Request) -> Decision:
        if request.images:
            raise RequestError("this model reads text only; images are not supported")
        probabilities, tokens = {}, 0
        for qid, question in request.questions.items():
            p, used = self.question(request.state, qid, question)
            tokens += used
            if question["type"] == "noul":
                probabilities[qid] = {"false": p[0], "true": p[1]}
            elif question["type"] == "choice":
                probabilities[qid] = dict(zip(question["criteria"], p))
            else:
                probabilities[qid] = {str(i): v for i, v in enumerate(p)}
        return Decision(probabilities, tokens)
