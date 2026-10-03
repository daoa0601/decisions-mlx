"""DiffusionGemma read as a diffusion canvas, through mlx-vlm.

A discrete diffusion model denoises a whole canvas of tokens per pass. The canvas is seeded with
an answer template ("q1: <slot>\\nq2: <slot>") in which only the label slots are noise; one
read-only decoder pass then gives, at each slot, a distribution over that question's one-token
labels (yes/no, A/B/C, 0/1/2). If any slot is uncertain, the group is read again with fresh noise
and the reads are averaged.

Ported from OpenJev (razorback16/openjev, Apache-2.0: ``openjev/engine.py`` and
``openjev/mlx_backend.py``), which adapts the structured-read example server of
vllm-project/vllm#57250. Only Jev's contract is ported: no extra denoise steps, sample counts,
thoughts or sequential chunks.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import mlx.core as mx
from transformers import AutoTokenizer

from ..api import Decision, Request, RequestError

# DiffusionGemma's vocabulary size and the token ids of its turn close and padding.
VOCAB = 262144
TURN_CLOSE = 106
PAD = 0
TOPK = 20
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"  # the empty thought block the chat template leaves to the model
# (join between questions, text before a label, reply instruction); "indexed" fits more questions per canvas.
FORMATS = {
    "lines": ("\n", "{id}: ", 'Reply with one line per question, in this order, formatted as "id: label".'),
    "indexed": (" ", "{id}", "Reply on one line with each question's id immediately followed by its label, separated by single spaces."),
}  # fmt: skip


def text_of(value: Any) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def seed_of(request: Request) -> int:
    """The canvas noise seed: the same request always gets the same noise, so answers repeat."""
    key = json.dumps([request.state, request.questions, len(request.images)], sort_keys=True, default=str)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def slot_distribution(top: dict[int, float], label_ids: list[int]) -> tuple[list[float], float]:
    """Label probabilities at one slot (temperature-1 logprobs, renormalized over the labels), and
    the entropy over the slot's top-k tokens, which decides whether to read again."""
    floor = min(top.values()) - 5.0
    lp = [top.get(i, floor) for i in label_ids]
    peak = max(lp)
    ex = [math.exp(x - peak) for x in lp]
    z = sum(ex)
    top_p = [math.exp(v) for v in top.values()]
    return [e / z for e in ex], -sum(p * math.log(p) for p in top_p if p > 0)


class CanvasPrompt:
    """What the model is asked and where each answer goes on the canvas; needs only the tokenizer."""

    def __init__(self, tokenizer: Any, canvas: int = 64, canvas_step: int = 16) -> None:
        self.tokenizer = tokenizer
        self.canvas, self.canvas_step = canvas, canvas_step
        self.scaffold = self.enc(SCAFFOLD_TEXT)
        self.choice_labels = self._single_token_labels()
        self._templates: dict[str, tuple[list[int], list[dict[str, Any]]]] = {}

    def enc(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _single_token_labels(self) -> list[str]:
        """Choice labels that stay one token after "q1: ", in a stable order."""
        base = self.enc("q1: A")
        candidates = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        candidates += [a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        out, seen = [], set()
        for c in candidates:
            e = self.enc("q1: " + c)
            if len(e) == len(base) and e[:-1] == base[:-1] and e[-1] not in seen:
                seen.add(e[-1])
                out.append(c)
            if len(out) == MAX_CHOICES:
                break
        return out

    def build_schema(self, questions: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
        """Questions to read, and the answers that need no read (one option or one level). The model
        sees q1, q2, ... rather than the caller's ids."""
        qs, forced = [], {}
        for qid, q in questions.items():
            kind, crit = q["type"], q["criteria"]
            if kind == "noul":
                crit = crit or {}
                choices = [("true", text_of(crit.get("true"))), ("false", text_of(crit.get("false")))]
                labels = ["yes", "no"]
            elif kind == "choice":
                if len(crit) == 1:
                    forced[qid] = {next(iter(crit)): 1.0}
                    continue
                if len(crit) > len(self.choice_labels):
                    raise RequestError(f"{qid}: at most {len(self.choice_labels)} choices")
                choices = [(name, text_of(desc)) for name, desc in crit.items()]
                labels = self.choice_labels[: len(choices)]
            else:
                if len(crit) > MAX_SCORE_LEVELS:
                    raise RequestError(f"{qid}: at most {MAX_SCORE_LEVELS} score levels")
                if len(crit) == 1:
                    forced[qid] = {"0": 1.0}
                    continue
                choices = [(str(i), text_of(c)) for i, c in enumerate(crit)]
                labels = [str(i) for i in range(len(crit))]
            qs.append({"key": qid, "type": kind, "instructions": text_of(q.get("instructions")),
                       "choices": choices, "labels": labels})  # fmt: skip
        for i, q in enumerate(qs):
            q["id"] = f"q{i + 1}"
        return qs, forced

    def system_text(self, qs: list[dict[str, Any]], fmt: str, chunked: bool = False) -> str:
        s = ("Answer a fixed set of questions about the state the user provides. "
             "Each question lists its allowed answers; reply with exactly one label per question.\n")  # fmt: skip
        for q in qs:
            s += f"\nQuestion {q['id']}: {q['instructions'] or 'Answer about the state.'}\n"
            for (name, desc), label in zip(q["choices"], q["labels"]):
                if q["type"] == "noul":
                    s += f"  {label}: {desc}\n" if desc else f"  {label}\n"
                elif q["type"] == "score":
                    s += f"  {label}: {desc}\n"
                else:
                    s += f"  {label}: {name} ({desc})\n" if desc else f"  {label}: {name}\n"
        s += "\n" + FORMATS[fmt][2]
        if chunked:
            s += " A reply may cover only some of the questions; answer every line that is present."
        return s

    def answer_text(self, qs: list[dict[str, Any]], labels: list[int], fmt: str) -> str:
        join, lead, _ = FORMATS[fmt]
        return join.join(lead.format(id=q["id"]) + q["labels"][label] for q, label in zip(qs, labels))

    def resolve_template(self, qs: list[dict[str, Any]], fmt: str) -> tuple[list[int], list[dict[str, Any]]]:
        """Tokenize the answer template and find each question's slot: every label must change
        exactly one token, at the same position for all of a question's labels."""
        key = json.dumps([fmt] + [(q["id"], q["labels"]) for q in qs])
        if key in self._templates:
            return self._templates[key]
        base_labels = [0] * len(qs)
        base = self.scaffold + self.enc(self.answer_text(qs, base_labels, fmt))
        if len(base) + 1 > self.canvas:
            raise RequestError(f"answer template is {len(base)} tokens; the canvas holds {self.canvas - 1}")
        slots = []
        for qi, q in enumerate(qs):
            pos, ids = None, [0] * len(q["labels"])
            for li in range(1, len(q["labels"])):
                labels = list(base_labels)
                labels[qi] = li
                e = self.scaffold + self.enc(self.answer_text(qs, labels, fmt))
                diffs = [i for i in range(min(len(e), len(base))) if e[i] != base[i]]
                if len(e) != len(base) or len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                    raise RequestError(f"question {q['key']!r}: labels do not share one template slot")
                pos = diffs[0]
                ids[li] = e[pos]
            ids[0] = base[pos]
            slots.append({"pos": pos, "label_ids": ids})
        if len(self._templates) > 4096:
            self._templates.clear()
        self._templates[key] = (base, slots)
        return base, slots

    def groups(self, qs: list[dict[str, Any]], fmt: str) -> list[list[dict[str, Any]]]:
        """Split questions, in order, into the fewest groups whose answer templates fit the canvas."""
        out, group = [], []
        for q in qs:
            trial = group + [q]
            rows = len(self.scaffold) + len(self.enc(self.answer_text(trial, [0] * len(trial), fmt))) + 1
            if rows > self.canvas and group:
                out.append(group)
                group = [q]
            else:
                group = trial
        out.append(group)
        return out

    def build_canvas(self, template: list[int], slots: list[dict[str, Any]], seed: int) -> list[int]:
        rng = random.Random(seed)
        width = min(self.canvas, -(-(len(template) + 1) // self.canvas_step) * self.canvas_step)
        canvas = list(template) + [TURN_CLOSE]
        canvas += [PAD] * (width - len(canvas))
        for s in slots:
            canvas[s["pos"]] = rng.randrange(VOCAB)
        return canvas

    def chat_ids(self, sys_text: str, state_text: str) -> list[int]:
        messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": state_text}]
        out = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return [int(t) for t in (out["input_ids"] if hasattr(out, "keys") else out)]


class CanvasDecider:
    def __init__(
        self,
        path: Path,
        canvas: int = 64,
        canvas_step: int = 16,
        rereads: int = 4,
        reread_entropy: float = 0.1,
        max_prompt: int = 32768,
        lazy: bool = False,
    ) -> None:
        from mlx_vlm import load

        self.model, self.processor = load(str(path), lazy=lazy)
        self.prompt = CanvasPrompt(AutoTokenizer.from_pretrained(path), canvas, canvas_step)
        self.rereads, self.reread_entropy, self.max_prompt = rereads, reread_entropy, max_prompt

    def prefill(self, sys_text: str, state_text: str, images: list[Any]) -> tuple[Any, int]:
        """The prompt's prefill cache, shared by every read of one group, and its length."""
        if images:
            from mlx_vlm.utils import prepare_inputs

            # this model's message format puts structured image placeholders before the text
            text = self.processor.apply_chat_template(
                [{"role": "system", "content": sys_text},
                 {"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": state_text}]}],
                add_generation_prompt=True, tokenize=False)  # fmt: skip
            inputs = prepare_inputs(self.processor, images=images, prompts=text)
            kwargs = {"input_ids": inputs["input_ids"], "pixel_values": inputs.get("pixel_values"),
                      "mm_token_type_ids": inputs.get("mm_token_type_ids"),
                      "attention_mask": inputs.get("attention_mask")}  # fmt: skip
            length = int(inputs["input_ids"].shape[-1])
        else:
            ids = self.prompt.chat_ids(sys_text, state_text)
            kwargs, length = {"input_ids": mx.array([ids])}, len(ids)
        if length > self.max_prompt:
            raise RequestError(f"the request is {length} tokens; the limit is {self.max_prompt}")
        return self.model.diffusion_prefill_cache(**kwargs), length

    def read(self, cache: Any, canvas: list[int], slots: list[dict[str, Any]]) -> list[dict[int, float]]:
        """One read-only decoder pass: {token id: logprob} for the top-k tokens and every label, per slot."""
        ids = mx.array([canvas])
        masks = self.model.diffusion_decoder_masks(ids, cache, None)
        logits = self.model.diffusion_decoder_logits(ids, cache=cache, self_conditioning=None, decoder_attention_mask=masks)
        out = []
        for s in slots:
            row = logits[0, s["pos"]].astype(mx.float32)
            lp = row - mx.logsumexp(row)
            keep = sorted(set(mx.argpartition(-lp, TOPK)[:TOPK].tolist()) | set(s["label_ids"]))
            out.append(dict(zip(keep, lp[mx.array(keep)].tolist())))
        return out

    def read_group(self, qs, fmt, sys_text, state_text, images, seed) -> tuple[list[list[float]], int]:
        """Mean label probabilities per question over the group's reads, and the prompt tokens."""
        template, slots = self.prompt.resolve_template(qs, fmt)
        cache, tokens = self.prefill(sys_text, state_text, images)

        def one(k: int) -> list[tuple[list[float], float]]:
            tops = self.read(cache, self.prompt.build_canvas(template, slots, seed + k * 7919), slots)
            return [slot_distribution(top, s["label_ids"]) for top, s in zip(tops, slots)]

        reads = [one(0)]
        if self.rereads > 1 and max(entropy for _, entropy in reads[0]) > self.reread_entropy:
            reads += [one(k) for k in range(1, self.rereads)]
        means = [
            [sum(r[qi][0][label] for r in reads) / len(reads) for label in range(len(q["labels"]))]
            for qi, q in enumerate(qs)
        ]
        return means, tokens

    def decide(self, request: Request) -> Decision:
        qs, probabilities = self.prompt.build_schema(request.questions)
        tokens = 0
        if qs:
            fmt = "lines" if len(qs) <= 10 else "indexed"
            state = request.state
            state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            groups = self.prompt.groups(qs, fmt)
            seed = seed_of(request)
            for k, group in enumerate(groups):
                sys_text = self.prompt.system_text(group, fmt, chunked=len(groups) > 1)
                means, used = self.read_group(group, fmt, sys_text, state_text, request.images, seed + 104729 * k)
                tokens += used
                for q, mean in zip(group, means):
                    probabilities[q["key"]] = {name: p for (name, _), p in zip(q["choices"], mean)}
        return Decision({qid: probabilities[qid] for qid in request.questions}, tokens)
