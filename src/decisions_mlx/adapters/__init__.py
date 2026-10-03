"""One adapter per way a model family turns a request into probabilities.

- ``clef``: a trained joint head over hidden states (Cloudflare Clef / Clef-Flash).
- ``letters``: next-token logits of the answer letters (JevK5, or any mlx-lm model zero-shot).
- ``canvas``: a diffusion model's answer canvas, read in one denoise pass (DiffusionGemma).
- ``laya``: a ModernBERT encoder that scores a marker token in front of every option (Laya).
- ``verdict``: a GLiClass encoder that dots label tokens with the text (Verdict).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from ..api import Decider

ADAPTERS = ("clef", "letters", "canvas", "laya", "verdict")


def resolve(path_or_repo: str | Path) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    # Skip remote code and other formats; every adapter here loads safetensors itself.
    return Path(
        snapshot_download(str(path_or_repo), ignore_patterns=["*.py", "*.gguf", "*.bin", "*.pt", "*.onnx"])
    )


def detect(path: Path) -> str:
    if (path / "joint_head_config.json").exists():
        return "clef"
    if (path / "rl_agent_config.json").exists():
        return "laya"
    config = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    if config.get("model_type") == "diffusion_gemma":
        return "canvas"
    if config.get("model_type") == "GLiClass":
        return "verdict"
    return "letters"


def load(path_or_repo: str | Path, adapter: str | None = None, **options: Any) -> Decider:
    """Load a model as a :class:`~decisions_mlx.api.Decider`; the adapter is detected from its files
    unless given."""
    if adapter is not None and adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter!r}; use one of {ADAPTERS}")
    path = resolve(path_or_repo)
    adapter = adapter or detect(path)
    if adapter == "clef":
        from .clef import ClefDecider

        return ClefDecider(path, **options)
    if adapter == "canvas":
        from .canvas import CanvasDecider

        return CanvasDecider(path, **options)
    if adapter == "laya":
        from .laya import LayaDecider

        return LayaDecider(path, **options)
    if adapter == "verdict":
        from .verdict import VerdictDecider

        return VerdictDecider(path, **options)
    from .letters import LetterDecider

    return LetterDecider(path, **options)


def convert(path_or_repo: str | Path, out: str | Path, adapter: str | None = None, bits: int = 8, group_size: int = 64) -> Path:
    """Quantize a model for its adapter. Clef and letters models are supported; DiffusionGemma
    ships quantized (mlx-community), and the encoders are small enough to run in float32."""
    path, out = resolve(path_or_repo), Path(out)
    adapter = adapter or detect(path)
    if adapter == "clef":
        from clef_mlx import convert as clef_convert

        return clef_convert(path, out, bits=bits, group_size=group_size)
    if adapter == "letters":
        from .letters import convert as letters_convert

        return letters_convert(path, out, bits=bits, group_size=group_size)
    raise ValueError(f"convert does not support the {adapter} adapter")
