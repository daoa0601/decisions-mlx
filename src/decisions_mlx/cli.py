"""``decisions-mlx`` command line: answer one /v1/systemone request, serve the endpoint, or quantize.

A model is given as ``[adapter:]path-or-repo``; without the prefix the adapter is detected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_model(spec: str) -> tuple[str | None, str]:
    from .adapters import ADAPTERS

    adapter, _, model = spec.partition(":")
    return (adapter, model) if adapter in ADAPTERS and model else (None, spec)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="decisions-mlx")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser(
        "run", help="answer a /v1/systemone request body (JSON file or stdin); images are file paths"
    )
    run_parser.add_argument("--model", required=True, help="[adapter:]path-or-repo")
    run_parser.add_argument("request", nargs="?", help="request JSON path; reads stdin when omitted")

    serve_parser = commands.add_parser("serve", help="serve POST /v1/systemone")
    serve_parser.add_argument(
        "--model", action="append", required=True, help="[adapter:]path-or-repo; repeat to serve several"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)

    convert_parser = commands.add_parser("convert", help="quantize a clef or letters model for MLX")
    convert_parser.add_argument("--model", required=True, help="[adapter:]path-or-repo")
    convert_parser.add_argument("--out", required=True)
    convert_parser.add_argument("--bits", type=int, default=8)
    convert_parser.add_argument("--group-size", type=int, default=64)

    args = parser.parse_args(argv)

    from .adapters import convert, load
    from .api import systemone

    if args.command == "convert":
        adapter, model = parse_model(args.model)
        print(f"wrote {convert(model, args.out, adapter, args.bits, args.group_size)}")
        return

    if args.command == "run":
        adapter, model = parse_model(args.model)
        body = json.load(open(args.request) if args.request else sys.stdin)
        body.setdefault("model", Path(model).name)
        if body.get("images"):
            from PIL import Image  # JSON carries image file paths here

            body["images"] = [Image.open(image).convert("RGB") for image in body["images"]]
        decider = load(model, adapter)
        start = time.perf_counter()
        response = systemone(decider, body)
        response["usage"]["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        print(json.dumps(response, indent=2))
        return

    from .server import make_server

    deciders = {}
    for spec in args.model:
        adapter, model = parse_model(spec)
        deciders[Path(model).name] = load(model, adapter)
    server = make_server(deciders, args.host, args.port)
    print(f"serving {', '.join(deciders)} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
