import json

from decisions_mlx.adapters import detect
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
