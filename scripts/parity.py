"""Parity: each model's own PyTorch reference vs its decisions-mlx adapter, on the same requests.

The two sides run in separate subprocesses so their weights never share memory.

    uv run python scripts/parity.py --adapter letters --model alibiserikbay/JevK5
    uv run python scripts/parity.py --adapter clef --model Cloudflare/clef-flash --mlx-model ../clef-mlx/clef-flash-8bit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

INTENTS = [
    "activate_card", "card_arrival", "card_not_working", "lost_or_stolen_card", "pin_blocked",
    "change_pin", "exchange_rate", "top_up_failed", "top_up_pending", "transfer_not_received",
    "cancel_transfer", "refund_not_showing_up", "request_refund", "duplicate_charge",
    "cash_withdrawal_fee", "declined_card_payment", "edit_personal_details", "verify_identity",
    "close_account", "terminate_account",
]  # fmt: skip

REQUESTS = [
    {
        "model": "parity",
        "state": "I was charged twice for my March subscription, please refund the duplicate.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should handle the message?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages", "sales": "Pricing"},
            },
            "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["Calm", "Annoyed", "Furious"]},
        },
    },
    {
        "model": "parity",
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "large": {
                "type": "noul",
                "instructions": "Is the total above 1000 USD?",
                "criteria": {"true": "The total exceeds 1000 USD.", "false": "The total is 1000 USD or less."},
            },
            "status": {"type": "choice", "instructions": "What is the invoice status?", "criteria": ["paid", "overdue", "draft"]},
        },
    },
    {
        "model": "parity",
        "state": "My card got swallowed by the ATM in Lisbon last night and I need cash today.",
        # 20 options: more than one read of 16 letters for the letters adapter.
        "questions": {"intent": {"type": "choice", "instructions": "Which banking intent is this?", "criteria": INTENTS}},
    },
    {
        "model": "parity",
        "state": {
            "pr": "Replaces the hand-rolled retry loop in payments/client.py with tenacity; adds no tests.",
            "policy": "Changes to payment code need tests and a second reviewer.",
        },
        "questions": {
            "compliant": {"type": "noul", "instructions": "Does the change follow the policy?"},
            "risk": {"type": "score", "instructions": "How risky is merging as is?", "criteria": ["Low", "Medium", "High", "Critical"]},
        },
    },
]


def reference_letters(model: str, device: str) -> list[dict]:
    from jevk5 import JevK5

    jevk5 = JevK5(model, device=device, graphs=False)
    results = []
    for body in REQUESTS:
        start = time.perf_counter()
        probabilities = {qid: jevk5.probabilities(body["state"], q)[0] for qid, q in body["questions"].items()}
        results.append({"seconds": time.perf_counter() - start, "probabilities": probabilities})
    return results


def reference_clef(model: str, device: str) -> list[dict]:
    import torch
    from huggingface_hub import snapshot_download

    from decisions_mlx import parse_request

    path = Path(model) if Path(model).is_dir() else Path(snapshot_download(model))
    spec = importlib.util.spec_from_file_location("joint_schema_model", path / "joint_schema_model.py")
    release = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release
    spec.loader.exec_module(release)
    clef, processor = release.load_release_model(str(path), device=device)
    results = []
    for body in REQUESTS:
        record = {"state": body["state"], "questions": parse_request(body).questions}
        encoded = release.encode_record(processor.tokenizer, record, processor=processor)
        batch = release.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device(device))
        start = time.perf_counter()
        with torch.inference_mode():
            logits = clef(batch)[0]
        probabilities = {
            q.question_id: dict(zip(q.option_ids, ql.float().softmax(-1).tolist()))
            for q, ql in zip(encoded.questions, logits)
        }
        results.append({"seconds": time.perf_counter() - start, "probabilities": probabilities})
    return results


REFERENCES = {"letters": reference_letters, "clef": reference_clef}


def run_mlx(adapter: str, model: str) -> list[dict]:
    from decisions_mlx import load, parse_request

    decider = load(model, adapter)
    decider.decide(parse_request(REQUESTS[0]))  # warm up kernels
    results = []
    for body in REQUESTS:
        start = time.perf_counter()
        decision = decider.decide(parse_request(body))
        results.append({"seconds": time.perf_counter() - start, "probabilities": decision.probabilities})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=sorted(REFERENCES), required=True)
    parser.add_argument("--model", required=True, help="checkpoint the reference runs")
    parser.add_argument("--mlx-model", help="checkpoint the adapter runs (defaults to --model)")
    parser.add_argument("--device", default="mps", help="torch device for the reference")
    parser.add_argument("--cache", type=Path, help="reuse/write the reference results here")
    parser.add_argument("--side", choices=["reference", "mlx"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.side == "reference":
        print(json.dumps(REFERENCES[args.adapter](args.model, args.device)))
        return
    if args.side == "mlx":
        print(json.dumps(run_mlx(args.adapter, args.mlx_model or args.model)))
        return

    def side(name: str) -> list[dict]:
        command = [sys.executable, __file__, "--side", name, "--adapter", args.adapter, "--model", args.model]
        command += ["--device", args.device] + (["--mlx-model", args.mlx_model] if args.mlx_model else [])
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        return json.loads(output.strip().splitlines()[-1])

    if args.cache and args.cache.exists():
        reference = json.loads(args.cache.read_text())
    else:
        reference = side("reference")
        if args.cache:
            args.cache.write_text(json.dumps(reference))
    ours = side("mlx")

    worst, agree, total = 0.0, 0, 0
    for index, (want, got) in enumerate(zip(reference, ours)):
        print(f"request {index}: reference {want['seconds']:.2f}s, mlx {got['seconds']:.2f}s")
        for question, expected in want["probabilities"].items():
            actual = got["probabilities"][question]
            diff = max(abs(expected[o] - actual[o]) for o in expected)
            worst = max(worst, diff)
            total += 1
            agree += max(expected, key=expected.get) == max(actual, key=actual.get)
            top = sorted(expected, key=expected.get, reverse=True)[:4]
            fmt = lambda p: " ".join(f"{o}={p[o]:.3f}" for o in top)  # noqa: E731
            print(f"  {question:<10} ref  {fmt(expected)}")
            print(f"  {'':<10} mlx  {fmt(actual)}   max|Δp|={diff:.4f}")
    print(f"\nargmax agreement {agree}/{total}, worst max|Δp| {worst:.4f}")


if __name__ == "__main__":
    main()
