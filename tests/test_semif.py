"""The ported SemIf prompt and many-option readout match JevK5's own ``jevk5.prompt``."""

import json
import random

import pytest

jevk5_prompt = pytest.importorskip("jevk5.prompt")

from decisions_mlx.adapters import semif  # noqa: E402

QUESTIONS = [
    {"type": "noul", "instructions": "Is it urgent?"},
    {"type": "noul", "criteria": {"true": "It is urgent.", "false": "It can wait."}},
    {"type": "choice", "criteria": {"billing": "Payments", "technical": None}},
    {"type": "choice", "criteria": ["a", "b", "c"]},
    {"type": "score", "criteria": ["low", "medium", "high"]},
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_prompt_matches_reference(question):
    options = [text for _, text in semif.decision_options(question)]
    assert semif.decision_options(question) == jevk5_prompt.decision_options(question)
    state = {"ticket": "Ünïcode ticket", "id": 7}
    assert semif.messages(state, "Which?", options) == jevk5_prompt.messages(state, "Which?", options)


def reader(seed: int):
    """A deterministic stand-in for one model read: a distribution keyed on the option texts."""

    def read(texts):
        weights = [random.Random(f"{seed}:{text}").random() ** 3 for text in texts]
        return [w / sum(weights) for w in weights]

    return read


@pytest.mark.parametrize("count", [2, 16, 17, 40, 300])
@pytest.mark.parametrize("method, temperature", [("knockout", None), ("knockout", 0.93), ("tree", None)])
def test_spread_matches_reference(count, method, temperature):
    texts = [f"option {i}: {json.dumps(i)}" for i in range(count)]
    ours = semif.spread(reader(count), texts, method, temperature)
    theirs = jevk5_prompt.spread(reader(count), texts, method, temperature)
    assert ours == pytest.approx(theirs, abs=1e-12)
    assert sum(ours) == pytest.approx(1.0)
