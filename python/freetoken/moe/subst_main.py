# SPDX-License-Identifier: Apache-2.0
"""``ft moe subst`` / ``ft moe subst-report``: build and inspect the expert substitution table.

A runtime artifact the server loads, so it belongs on the CLI rather than in scripts/ -- the same
distinction ``ft bench bw`` and ``ft checkpoint`` draw. The measurement itself lives in
``freetoken.moe.subst_table``.
"""

from __future__ import annotations

import argparse


def main(argv: list[str], prog: str = "ft moe") -> int:
    sub = argv[0]
    ap = argparse.ArgumentParser(prog=prog)
    if sub == "subst-report":
        ap.add_argument("--table", required=True)
        args = ap.parse_args(argv[1:])
        from freetoken.moe.subst_table import report

        report(args.table)
        return 0

    ap.add_argument("--model", required=True, help="the HF checkpoint the experts come from")
    ap.add_argument("--dumps", required=True,
                    help="route-dump directory captured with FREETOKEN_ROUTE_DUMP_X=1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict to these layers (default: every MoE layer)")
    ap.add_argument("--num-experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--shortlist", type=int, default=32, help="substitutes kept per expert")
    ap.add_argument("--chunk", type=int, default=96, help="tokens per pass over the experts")
    ap.add_argument("--min-tokens", type=int, default=256,
                    help="skip dumps shorter than this (decode-step dumps carry a few tokens)")
    ap.add_argument("--no-resume", action="store_true",
                    help="rebuild layers whose partial already exists")
    args = ap.parse_args(argv[1:])

    from freetoken.moe.subst_table import build_table

    build_table(args.model, args.dumps, args.out, layers=args.layers,
                num_experts=args.num_experts, top_k=args.top_k, m=args.shortlist,
                chunk=args.chunk, min_tokens=args.min_tokens, resume=not args.no_resume)
    return 0
