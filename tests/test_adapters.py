import json
from types import SimpleNamespace

import mlx.core as mx
from clef_mlx import Clef, encode_record

from decisions_mlx import parse_request, systemone
from decisions_mlx.adapters import detect
from decisions_mlx.adapters.clef import ClefDecider
from decisions_mlx.adapters.letters import _model_config
from decisions_mlx.cli import parse_model


def test_detect_picks_clef_by_its_head_config(tmp_path):
    assert detect(tmp_path) == "letters"
    (tmp_path / "joint_head_config.json").write_text("{}")
    assert detect(tmp_path) == "clef"


def test_text_only_model_types_load_with_their_base_module(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
    assert _model_config(tmp_path) == {"model_type": "qwen3_5"}
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert _model_config(tmp_path) is None


def test_model_specs():
    assert parse_model("letters:alibiserikbay/JevK5") == ("letters", "alibiserikbay/JevK5")
    assert parse_model("Cloudflare/clef-flash") == (None, "Cloudflare/clef-flash")
    assert parse_model("./clef-flash-8bit") == (None, "./clef-flash-8bit")



class CharTokenizer:
    """One token per character: enough for clef-mlx's record encoding without a real vocabulary."""

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(character) for character in text])


def rising_head(hidden, encoded, rows):
    """Logits 0, 1, 2, ... per question, so each question's last option is the most likely."""
    return [mx.arange(len(question.option_ids), dtype=mx.float32) for question in encoded.questions]


def test_clef_adapter_answers_through_clef_mlx_encoding_and_probabilities():
    tokenizer = CharTokenizer()
    decider = object.__new__(ClefDecider)
    decider.clef = Clef(backbone=lambda encoded: mx.zeros((1, 4)), head=rising_head, tokenizer=tokenizer)
    body = {
        "model": "clef-flash",
        "state": "Checkout is down.",
        "questions": {
            "outage": {"type": "noul"},
            "team": {"type": "choice", "criteria": ["billing", "technical"]},
        },
    }
    response = systemone(decider, body)
    record = {"state": body["state"], "questions": parse_request(body).questions}
    expected_tokens = len(encode_record(tokenizer, record).input_ids)
    assert response == {
        "model": "clef-flash",
        "answers": {
            "outage": {"type": "noul", "noul": 0.2689},
            "team": {
                "type": "choice",
                "choice": "technical",
                "confidence": 0.7311,
                "probabilities": {"billing": 0.2689, "technical": 0.7311},
            },
        },
        "usage": {"input_tokens": expected_tokens, "output_tokens": 0},
    }
