"""ModernBERT in MLX, matching transformers' ``ModernBertModel`` (the encoder under Laya and Verdict).

Parameter names follow transformers, so a checkpoint's ``encoder.*`` weights load as they are.
Sequences run one at a time without padding, so the only mask is the sliding window of the
local-attention layers.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class Embeddings(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.norm = nn.LayerNorm(config["hidden_size"], eps=config["norm_eps"], bias=config["norm_bias"])

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class Attention(nn.Module):
    def __init__(self, config: dict[str, Any], rope_theta: float) -> None:
        super().__init__()
        width = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.head_dim = width // self.heads
        self.Wqkv = nn.Linear(width, 3 * width, bias=config["attention_bias"])
        self.Wo = nn.Linear(width, width, bias=config["attention_bias"])
        # transformers' rotate_half layout, which is MLX's non-traditional RoPE
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_theta)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(batch, length, 3, self.heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = self.rope(qkv[0]), self.rope(qkv[1]), qkv[2]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.Wi = nn.Linear(config["hidden_size"], 2 * config["intermediate_size"], bias=config["mlp_bias"])
        self.Wo = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=config["mlp_bias"])

    def __call__(self, x: mx.array) -> mx.array:
        inputs, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(inputs) * gate)


class Layer(nn.Module):
    def __init__(self, config: dict[str, Any], index: int, sliding: bool, rope_theta: float) -> None:
        super().__init__()
        width, eps, bias = config["hidden_size"], config["norm_eps"], config["norm_bias"]
        self.sliding = sliding
        # Layer 0 has no attention norm (nn.Identity in transformers, so no weights).
        self.attn_norm = nn.LayerNorm(width, eps=eps, bias=bias) if index else None
        self.attn = Attention(config, rope_theta)
        self.mlp_norm = nn.LayerNorm(width, eps=eps, bias=bias)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, window: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x) if self.attn_norm is not None else x, window if self.sliding else None)
        return x + self.mlp(self.mlp_norm(x))


def _rope_theta(config: dict[str, Any], layer_type: str) -> float:
    params = (config.get("rope_parameters") or {}).get(layer_type)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    if layer_type == "sliding_attention":
        return float(config.get("local_rope_theta") or 10000.0)
    return float(config.get("global_rope_theta", 160000.0))


class ModernBert(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        every = config.get("global_attn_every_n_layers", 3)
        layer_types = config.get("layer_types") or [
            "full_attention" if i % every == 0 else "sliding_attention" for i in range(config["num_hidden_layers"])
        ]
        self.half_window = config.get("local_attention", 128) // 2
        self.embeddings = Embeddings(config)
        self.layers = [
            Layer(config, i, kind == "sliding_attention", _rope_theta(config, kind)) for i, kind in enumerate(layer_types)
        ]
        self.final_norm = nn.LayerNorm(config["hidden_size"], eps=config["norm_eps"], bias=config["norm_bias"])

    def __call__(self, input_ids: mx.array) -> mx.array:
        """``last_hidden_state`` for unpadded ``input_ids`` of shape ``[batch, length]``."""
        positions = mx.arange(input_ids.shape[1])
        window = mx.abs(positions[:, None] - positions[None, :]) <= self.half_window
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, window)
        return self.final_norm(x)
