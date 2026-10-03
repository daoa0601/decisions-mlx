"""One adapter per way a model family turns a request into probabilities.

- ``clef``: a trained joint head over hidden states (Cloudflare Clef / Clef-Flash).
- ``letters``: next-token logits of the answer letters (JevK5, or any mlx-lm model zero-shot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from ..api import Decider

ADAPTERS = ("clef", "letters")


def resolve(path_or_repo: str | Path) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    # Skip remote code and other formats; every adapter here loads safetensors itself.
    return Path(snapshot_download(str(path_or_repo), ignore_patterns=["*.py", "*.gguf", "*.bin", "*.pt"]))


def detect(path: Path) -> str:
    return "clef" if (path / "joint_head_config.json").exists() else "letters"


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
    from .letters import LetterDecider

    return LetterDecider(path, **options)
