"""Option-letter readout over any mlx-lm causal LM.

Each question is one read: SemIf's prompt (``semif.py``) with lettered options, then a softmax
over the letters' next-token logits, divided by the model's temperature. This runs JevK5
(Qwen3.5 + a distilled LoRA, temperatures in ``jevk5_config.json``) exactly, and any other mlx-lm
model zero-shot with temperature 1.

Reads share long prompt prefixes (the system prompt and the state come first, so every question
about one state starts the same way), and the model processes a shared prefix once: see
``letter_probabilities``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model
from transformers import AutoTokenizer

from ..api import Decision, Request, RequestError
from .semif import LETTERS, METHODS, decision_options, messages, spread

CONFIG = "jevk5_config.json"
# Snapshot a shared prompt prefix only when it saves at least this many tokens of prefill.
MIN_SHARED_TOKENS = 32


def _model_config(path: Path) -> dict[str, Any] | None:
    """Map a text-only HF model type (``qwen3_5_text``) onto the mlx-lm module that loads it."""
    model_type = json.loads((path / "config.json").read_text())["model_type"]
    if importlib.util.find_spec(f"mlx_lm.models.{model_type}") is None and model_type.endswith("_text"):
        base = model_type.removesuffix("_text")
        if importlib.util.find_spec(f"mlx_lm.models.{base}") is not None:
            return {"model_type": base}
    return None


class LetterDecider:
    def __init__(
        self,
        path: Path,
        temperature: float | None = None,
        knockout_temperature: float | None = None,
        method: str = "knockout",
        lazy: bool = False,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
        self.model, _ = load_model(path, lazy=lazy, model_config=_model_config(path))
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        slots = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(ids) != 1 for ids in slots):
            raise ValueError("every answer letter must be one token for this tokenizer")
        self.slots = mx.array([ids[0] for ids in slots])
        config = json.loads((path / CONFIG).read_text()) if (path / CONFIG).exists() else {}
        self.temperature = float(config.get("temperature", 1.0)) if temperature is None else temperature
        if knockout_temperature is None and "knockout_temperature" in config:
            knockout_temperature = float(config["knockout_temperature"])
        self.method = method
        self.knockout_temperature = knockout_temperature if method == "knockout" else None
        # The last read's prompt, and one snapshot: a prompt prefix with its processed cache.
        self._last: list[int] = []
        self._prefix: tuple[list[int], list[Any]] | None = None

    def encode(self, state: Any, criterion: str, options: list[str]) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            messages(state, criterion, options),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def _prefill(self, ids: list[int], cache: list[Any]) -> None:
        if ids:
            self.model(mx.array([ids]), cache=cache)  # the logits are never evaluated, only the cache
            mx.eval([c.state for c in cache])

    def letter_probabilities(self, ids: list[int], count: int) -> list[float]:
        """The calibrated distribution over the first ``count`` letters after the prompt ``ids``.

        All but the last token are prefilled into a cache and the last runs alone, so only one
        position goes through the output projection. The cache starts from the snapshot when the
        snapshot's prompt is a prefix of ``ids``. When ``ids`` shares a longer prefix with the
        previous read, that prefix is processed first and becomes the new snapshot.
        """
        done, cache = 0, None
        if self._prefix is not None:
            prefix, snapshot = self._prefix
            if len(prefix) < len(ids) and ids[: len(prefix)] == prefix:
                done, cache = len(prefix), copy.deepcopy(snapshot)
        if cache is None:
            cache = make_prompt_cache(self.model)
        shared = 0
        for a, b in zip(ids[:-1], self._last):
            if a != b:
                break
            shared += 1
        if shared - done >= MIN_SHARED_TOKENS:
            self._prefill(ids[done:shared], cache)
            self._prefix, done = (ids[:shared], copy.deepcopy(cache)), shared
        self._prefill(ids[done:-1], cache)
        self._last = ids
        logits = self.model(mx.array([ids[-1:]]), cache=cache)[0, -1, self.slots[:count]]
        return mx.softmax(logits.astype(mx.float32) / self.temperature).tolist()

    def question(self, state: Any, question_id: str, question: dict[str, Any]) -> tuple[dict[str, float], int]:
        """Probability per option id for one question, and the input tokens over all its reads."""
        options = decision_options(question)
        criterion = question.get("instructions") or question_id
        tokens = 0

        def read(texts: list[str]) -> list[float]:
            nonlocal tokens
            ids = self.encode(state, criterion, texts)
            tokens += len(ids)
            return self.letter_probabilities(ids, len(texts))

        probs = spread(read, [text for _, text in options], self.method, self.knockout_temperature)
        return {key: p for (key, _), p in zip(options, probs, strict=True)}, tokens

    def decide(self, request: Request) -> Decision:
        if request.images:
            raise RequestError("this model reads text only; images are not supported")
        probabilities, tokens = {}, 0
        for question_id, question in request.questions.items():
            probabilities[question_id], used = self.question(request.state, question_id, question)
            tokens += used
        return Decision(probabilities, tokens)


def convert(source: Path, out: Path, bits: int = 8, group_size: int = 64) -> Path:
    """Quantize a letters model with mlx-lm, keeping its readout config.

    mlx-lm's own ``convert`` cannot load text-only model types such as ``qwen3_5_text``, so this
    composes its load, quantize and save with the same type mapping ``LetterDecider`` uses.
    """
    from mlx_lm.utils import load, quantize_model, save

    if out.exists():
        raise ValueError(f"{out} already exists")
    model, tokenizer, config = load(str(source), return_config=True, lazy=True, model_config=_model_config(source))
    model, config = quantize_model(model, config, group_size, bits)
    save(out, source, model, tokenizer, config)
    if (source / CONFIG).exists():
        shutil.copyfile(source / CONFIG, out / CONFIG)
    return out
