"""The ported prompts and encodings match their reference packages, using only tokenizers.

Model weights are not needed; a test skips when its tokenizer is not in the local Hugging Face
cache. ``scripts/parity.py`` checks the full models.
"""

import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from decisions_mlx import parse_request

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does it need a reply within the hour?"},
    "fraud": {"type": "noul", "criteria": {"true": "Signs of fraud.", "false": "Ordinary."}},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": None}},
    "intent": {"type": "choice", "instructions": "Intent?", "criteria": [f"intent_{i}" for i in range(30)]},
    "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
    "only": {"type": "choice", "criteria": ["just_one"]},
}
STATES = ["Everything is down; we demo at noon.", {"ticket": "Refund ✓ please", "id": 7}, ["hi", "still broken"]]


def cached(repo: str):
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001 - not downloaded
        pytest.skip(f"{repo} is not in the local Hugging Face cache")


def questions():
    return parse_request({"model": "m", "state": "", "questions": QUESTIONS}).questions


def test_canvas_prompt_matches_openjev():
    openjev = pytest.importorskip("openjev.engine")
    from openjev.config import Settings

    from decisions_mlx.adapters.canvas import CanvasPrompt

    tokenizer = AutoTokenizer.from_pretrained(cached("mlx-community/diffusiongemma-26B-A4B-it-4bit"))
    ours, theirs = CanvasPrompt(tokenizer), openjev.Engine(Settings(), tokenizer)
    assert ours.choice_labels == theirs.choice_labels

    qs, forced = ours.build_schema(questions())
    schema = theirs.build_schema(questions())
    assert set(forced) == set(schema["forced"]) == {"only"}
    assert [(q["id"], q["labels"]) for q in qs] == [(q["id"], q["labels"]) for q in schema["questions"]]
    fmt = schema["format"]
    assert [[q["id"] for q in g] for g in ours.groups(qs, fmt)] == [
        [q["id"] for q in g] for g in theirs.groups(schema["questions"], fmt)
    ]
    for group, their_group in zip(ours.groups(qs, fmt), theirs.groups(schema["questions"], fmt)):
        assert ours.system_text(group, fmt, True) == theirs.system_text(their_group, fmt, True)
        template, slots = ours.resolve_template(group, fmt)
        assert (template, slots) == theirs.resolve_template(their_group, fmt)
        assert ours.build_canvas(template, slots, 1234) == theirs.build_canvas(template, slots, 1234)
    assert ours.chat_ids("system", "state") == theirs.chat_prompt_ids("system", "state")


@pytest.mark.parametrize("state", STATES)
def test_laya_sequences_match_laya(state):
    common = pytest.importorskip("laya.common")
    from decisions_mlx.adapters.laya import encode

    path = cached("convaiinnovations/laya-typed-decisions")
    tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")
    for qid, q in questions().items():
        ins = q.get("instructions") or qid
        ours = encode(tokenizer, state, q["type"], ins, q["criteria"], 1024, 256)
        internal = {"t": q["type"], "ins": ins, "crit": q["criteria"]}
        theirs = common.build_sequence(tokenizer, state, internal, 1024, 256, truncate_left=isinstance(state, list))
        assert ours == tuple(theirs), qid


def test_laya_long_states_and_option_budgets_match_laya():
    common = pytest.importorskip("laya.common")
    from decisions_mlx.adapters.laya import encode

    tokenizer = AutoTokenizer.from_pretrained(f"{cached('convaiinnovations/laya-typed-decisions')}/tokenizer")
    long_state = " ".join(f"line {i} of a very long incident log" for i in range(400))
    many = {f"option_{i}": "a fairly long description of this option " * 3 for i in range(20)}
    wordy = {f"option_{i}": "an option whose description runs well past the per-option cap " * 8 for i in range(2)}
    for state in (long_state, [long_state, "newest turn"]):
        for criteria in (many, wordy):
            ours = encode(tokenizer, state, "choice", "Which option?", criteria, 1024, 256)
            theirs = common.build_sequence(
                tokenizer, state, {"t": "choice", "ins": "Which option?", "crit": criteria}, 1024, 256,
                truncate_left=isinstance(state, list),
            )  # fmt: skip
            assert ours == tuple(theirs)


def test_verdict_prompts_match_openjev():
    encoders = pytest.importorskip("openjev.encoders")
    from decisions_mlx.adapters.verdict import prompt, text_of

    for qid, q in questions().items():
        if q["type"] == "choice":
            choices = [(name, text_of(desc)) for name, desc in q["criteria"].items()]
        elif q["type"] == "score":
            choices = [(str(i), text_of(c)) for i, c in enumerate(q["criteria"])]
        else:
            choices = []
        ins = text_of(q.get("instructions")) or qid
        theirs = encoders.verdict_prompt({"type": q["type"], "instructions": ins, "choices": choices}, "ctx")
        assert prompt(q["type"], ins, choices, "ctx") == theirs, qid
