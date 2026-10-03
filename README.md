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
| `canvas` | A diffusion model's answer canvas seeded with a template whose label slots are noise; one read-only denoise pass gives each slot's distribution, re-read with fresh noise when uncertain | [DiffusionGemma 26B-A4B](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit) (4-bit, ~16 GB), zero-shot, text and images |
| `laya` | A ModernBERT encoder plus a 2-layer head scores a `[MASK]` marker in front of every option; temperatures per question type and option count | [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) (421M), text |
| `verdict` | A GLiClass encoder dots each label token with the text's first token; temperatures per label count, an abstention label dropped | [Verdict](https://huggingface.co/heman10x/rlcd-modernbert-151m) (151M), text, up to 24 options |

A model's adapter is detected from its files unless you name it. Formatting mistakes in these
models don't raise errors, they just produce wrong probabilities, so every adapter is checked
against its model's own reference code (see [Parity](#parity)).

Not covered: CLM (contrastive heads over Qwen3-8B embeddings), DiffusionGemma's non-Jev request
options (extra denoise steps, sample counts, thoughts), Laya's long-state windowing, and video.

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

uv run decisions-mlx convert --model alibiserikbay/JevK5 --out ./JevK5-8bit    # clef and letters models
```

`--model` takes a Hub repo or a local directory, optionally prefixed with the adapter
(`letters:Qwen/Qwen3.5-4B`). The server routes each request by its `model` field to the model
with that directory or repo name (`JevK5`, `clef-flash-8bit`). With a single model loaded it
accepts any name, so clients that send `jev-latest` work unchanged. Images go in `images` as
`data:image/...;base64,` URLs (Clef and DiffusionGemma). `GET /v1/models` lists the served names.

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

The `letters` adapter processes a shared prompt prefix once: the system prompt and the state come
first, so the questions about one state differ only after it. On a ~2,800-token state with six
questions, JevK5 answers in 7.2 s instead of 20.2 s; against the PyTorch reference the two modes
are equally close (worst |Δp| 0.025 either way, bf16 noise at that length).

`convert` quantizes Clef (through clef-mlx) and letters models. JevK5 at 8-bit is 4.2 GB; it agrees
with the bf16 reference on 7 of the 8 parity questions (the miss is an exact tie) with a worst
|Δp| of 0.03. DiffusionGemma already ships quantized, and the two encoders are small enough to run
in float32.

## Parity

```bash
uv sync --all-extras
uv run pytest     # contract, server, and every ported prompt/encoding vs its reference package
uv run python scripts/parity.py --adapter letters --model alibiserikbay/JevK5
uv run python scripts/parity.py --adapter clef --model Cloudflare/clef-flash --mlx-model ../clef-mlx/clef-flash-8bit
uv run python scripts/parity.py --adapter canvas --model mlx-community/diffusiongemma-26B-A4B-it-4bit
uv run python scripts/parity.py --adapter laya --model convaiinnovations/laya-typed-decisions --device cpu
uv run python scripts/parity.py --adapter verdict --model heman10x/rlcd-modernbert-151m --device cpu
```

`parity.py` runs each model's reference code and the adapter on the same four requests: 8
questions covering all three types, list and dict criteria, a JSON state, and a 20-option choice
(which the `letters` adapter reads in groups). Adapters that take images also get a receipt image
request. The references:

- `letters`: JevK5's `jevk5` package (transformers + torch on MPS, bf16).
- `clef`: Clef's own `joint_schema_model.py` (MPS, bf16).
- `canvas`: OpenJev's MLX engine, on the same mlx-vlm and with the adapter's noise seed. This
  checks the port; the original vLLM path needs an NVIDIA GPU.
- `laya`, `verdict`: OpenJev's engines over the `laya` and `gliclass` packages (CPU, float32).

Measured on an M4 Pro (48 GB):

| Adapter, checkpoint | Argmax agreement | Worst \|Δp\| | Time per request, reference → MLX |
|---|---|---|---|
| `letters`, JevK5 (bf16) | 8/8 | 0.0023 | 1.3–3.6 s → 0.5–1.3 s |
| `clef`, Clef-Flash (8-bit, vs bf16 reference) | 8/8 | 0.0098 | 1.5–4.6 s → 0.6–1.0 s |
| `canvas`, DiffusionGemma (4-bit) | 10/10, incl. image | 0.0000 | 0.6–2.1 s → 0.6–1.4 s |
| `laya`, Laya (float32) | 8/8 | 0.0001 (Laya rounds to 4 places) | 0.09–0.16 s → 0.03–0.05 s |
| `verdict`, Verdict (float32) | 8/8 | 0.0000 | 0.04–0.64 s → 0.01–0.02 s |

With prefix reuse on (the default), the `letters` row becomes 7/8 and 0.027: the miss is the
20-option question, whose top two options are an exact tie (0.472 each) in the reference.

## Zero-shot evaluation

```bash
uv run python scripts/evaluate.py --model ../clef-mlx/clef-flash-8bit --task banking77 --limit 200
```

`evaluate.py` sends each item of a labelled task through the adapter as a `/v1/systemone` request
with one `choice` question over all the task's labels, so every model sees the same input. It
reports accuracy, macro-F1, multi-class Brier, top-label ECE (15 bins) and median latency. On 200
items of the BANKING77 test split (seed 0, 77 options), M4 Pro:

| Model | Accuracy | Macro-F1 | Brier | ECE | Median per item |
|---|---|---|---|---|---|
| Clef-Flash, 8-bit | 0.935 | 0.911 | 0.101 | 0.054 | 3.7 s (one pass over all 77 options) |
| JevK5 (4B), bf16 | 0.645 | 0.595 | 0.517 | 0.105 | 2.6 s (6 reads of ≤16 options) |
| DiffusionGemma 26B-A4B, 4-bit | 0.645 | 0.591 | 0.590 | 0.265 | 1.3 s (one canvas read, re-read when uncertain) |
| Laya (421M), float32 | 0.320 | 0.236 | 0.854 | 0.206 | 0.06 s |

Verdict is left out: it takes at most 24 options. 200 items carry about ±6 points of sampling
error. For comparison:

- Cloudflare reports BANKING77 macro-F1 of 90.9 for Clef-Flash, 74.3 for DiffusionGemma as Jev
  and 14.3 for Laya, from its own run of the Decision Index.
- JevK5 reports 0.69 accuracy for v0.2 on BANKING77's train split.

Laya gives each question 256 head tokens, so 77 options get about 4 tokens each. The two
large-model latencies are prompt-processing bound: Clef reads every option's description in one
long prompt, and JevK5 reads 77 options as five groups of 15–16 plus a final.

## Attribution

All ported code comes from Apache-2.0 projects:

- `adapters/semif.py`: `jevk5/prompt.py` from [JevK5](https://github.com/allebee/jevk5), which
  follows [SemIf](https://github.com/TheoLeeCJ/SemIf) (MIT).
- `adapters/canvas.py`: [OpenJev](https://github.com/razorback16/openjev)'s engine and MLX
  backend, from the structured-read example in vllm-project/vllm#57250.
- `adapters/laya.py`: [Laya](https://github.com/NandhaKishorM/laya)'s sequence layout and
  decision head.
- `adapters/verdict.py`: OpenJev's Verdict engine (after
  [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev) v1.4) and
  [GLiClass](https://github.com/Knowledgator/GLiClass)'s uni-encoder.
- `adapters/modernbert.py`: transformers' ModernBERT.
- Clef support: clef-mlx, a port of Cloudflare's release code.
