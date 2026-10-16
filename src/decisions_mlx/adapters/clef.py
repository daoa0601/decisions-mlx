"""Clef / Clef-Flash: a joint schema head over a Qwen3.5 backbone, via the sibling ``clef-mlx`` port.

All questions of a request are answered jointly in one forward pass; images need the ``vision``
extra and a checkpoint that kept its vision tower.
"""

from __future__ import annotations

from pathlib import Path

from ..api import Decision, Request


class ClefDecider:
    def __init__(self, model: str | Path, vision: bool | None = None) -> None:
        from clef_mlx import load

        self.clef = load(model, vision=vision)

    def decide(self, request: Request) -> Decision:
        encoded = self.clef.encode(
            {"state": request.state, "questions": request.questions, "images": request.images}
        )
        return Decision(self.clef.option_probabilities(encoded), len(encoded.input_ids))
