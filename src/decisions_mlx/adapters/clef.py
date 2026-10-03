"""Clef / Clef-Flash: a joint schema head over a Qwen3.5 backbone, via the sibling ``clef-mlx`` port.

All questions of a request are answered jointly in one forward pass; images need the ``vision``
extra and a checkpoint that kept its vision tower.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ..api import Decision, Request


class ClefDecider:
    def __init__(self, model: str | Path, vision: bool | None = None) -> None:
        from clef_mlx import load

        self.clef = load(model, vision=vision)

    def decide(self, request: Request) -> Decision:
        encoded = self.clef.encode(
            {"state": request.state, "questions": request.questions, "images": request.images}
        )
        probabilities = {
            question.question_id: dict(
                zip(question.option_ids, mx.softmax(logits.astype(mx.float32), axis=-1).tolist())
            )
            for question, logits in zip(encoded.questions, self.clef.logits(encoded))
        }
        return Decision(probabilities, len(encoded.input_ids))
