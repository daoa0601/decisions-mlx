# decisions-mlx

Run open Jev / SystemOne decision models on Apple silicon with [MLX](https://github.com/ml-explore/mlx),
behind one `POST /v1/systemone` API.

These models all take the same request (a state plus typed `noul` / `choice` / `score` questions)
and return a probability per option, but they compute those probabilities in different ways. The
request contract, response formatting, server and parity harness are shared here, and each way of
computing the probabilities is one adapter:

| Adapter | How it answers | Models |
|---|---|---|
| `clef` | A trained joint head reads the backbone's hidden states at every question and option span; all questions in one pass | [Cloudflare/clef](https://huggingface.co/Cloudflare/clef), [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (via [clef-mlx](https://github.com/daoa0601/clef-mlx)), text and images |
| `letters` | SemIf's prompt with lettered options; a softmax over the letters' next-token logits, one read per question (several above 16 options) | [JevK5](https://huggingface.co/alibiserikbay/JevK5) and [JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B) exactly; any other mlx-lm model zero-shot |

A model's adapter is detected from its files (`joint_head_config.json` means `clef`) unless you
name it. Formatting mistakes in these models don't raise errors, they just produce wrong
probabilities, so every adapter is checked against its model's own PyTorch code
(see [Parity](#parity)).

Not covered yet: DiffusionGemma read as a diffusion canvas ([OpenJev](https://github.com/razorback16/openjev)
already runs that on MLX), encoder classifiers such as Laya or Verdict, and video.

## Install

This repo depends on a sibling checkout of [clef-mlx](https://github.com/daoa0601/clef-mlx) at
`../clef-mlx` (an explicit uv path source):

```bash
git clone https://github.com/daoa0601/clef-mlx ../clef-mlx
uv sync                       # text
uv sync --extra vision        # + images for Clef (mlx-vlm)
```

## Use

```bash
echo '{"model": "jev-latest", "state": "I was charged twice, please refund the duplicate.",
       "questions": {"refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}}}' \
  | uv run decisions-mlx run --model alibiserikbay/JevK5

uv run decisions-mlx serve --model alibiserikbay/JevK5 --model ../clef-mlx/clef-flash-8bit --port 8080
```

`--model` takes a Hub repo or a local directory, optionally prefixed with the adapter
(`letters:Qwen/Qwen3.5-4B`). The server routes each request by its `model` field to the model
with that directory or repo name (`JevK5`, `clef-flash-8bit`). With a single model loaded it
accepts any name, so clients that send `jev-latest` work unchanged. Images go in `images` as
`data:image/...;base64,` URLs (Clef only). `GET /v1/models` lists the served names.

```python
from decisions_mlx import load, systemone

decider = load("alibiserikbay/JevK5")
print(systemone(decider, {"model": "jevk5", "state": "...", "questions": {...}}))
```

The response has the same shape for every adapter:

- `noul`: `noul`, the probability of true.
- `choice`: `choice`, `confidence` (its probability) and `probabilities`.
- `score`: the expected `score`, `confidence`, `legend` and `probabilities`.
- `usage.input_tokens`: the tokens the model read, summed over reads.

A question without `instructions` uses its id as the instruction.

## Parity

```bash
uv sync --all-extras
uv run pytest                                  # contract, server, and the SemIf port vs jevk5.prompt
uv run python scripts/parity.py --adapter letters --model alibiserikbay/JevK5
uv run python scripts/parity.py --adapter clef --model Cloudflare/clef-flash --mlx-model ../clef-mlx/clef-flash-8bit
```

`parity.py` runs the model's own reference code (JevK5's `jevk5` package, or Clef's
`joint_schema_model.py`; transformers + torch on MPS, bf16) and the adapter on the same four
requests: 8 questions covering all three types, list and dict criteria, a JSON state, and a
20-option choice that the `letters` adapter reads in groups. Measured on an M4 Pro (48 GB):

| Adapter, checkpoint | Argmax agreement | Worst \|Δp\| | Time per request, MPS reference → MLX |
|---|---|---|---|
| `letters`, JevK5 (bf16) | 8/8 | 0.0023 | 1.3–3.6 s → 0.5–1.3 s |
| `clef`, Clef-Flash (8-bit, vs bf16 reference) | 8/8 | 0.0098 | 1.5–4.6 s → 0.6–1.0 s |

## Attribution

`src/decisions_mlx/adapters/semif.py` is ported from `jevk5/prompt.py` in
[JevK5](https://github.com/allebee/jevk5) (Apache-2.0), which follows
[SemIf](https://github.com/TheoLeeCJ/SemIf) (MIT). Clef support comes from clef-mlx, a port of
Cloudflare's Apache-2.0 release code.
