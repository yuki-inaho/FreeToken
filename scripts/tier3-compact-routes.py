#!/usr/bin/env python3
"""Compact FREETOKEN_ROUTE_DUMP router logits into a small top-k id trace.

The raw dumps carry per-layer router logits ``[T, num_experts]`` plus (under
FREETOKEN_ROUTE_DUMP_R=1) the residual and block input -- ~2.9 GB per forward, 13 GB for a
session. Tier-3 simulation only ever needs the routed expert *ids*, so this does the one
expensive pass and writes ``[num_layers, T, top_k] int16`` per chunk (~4 MB), which
``freetoken.tier3.sim`` can then replay hundreds of times while sweeping cache geometries.

Run it nice'd -- it reads tens of GB off the same volume the model serves from:

    ionice -c3 nice -n19 python scripts/tier3-compact-routes.py \
        --dump-dir "/media/nuko/New Volume/route-dumps-m1" --out routes-m1.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import torch

TOP_K = 10  # num_experts_per_tok (Qwen3.8-Flash-Next)


def compact_chunk(data: dict, top_k: int) -> dict | None:
    """One route dump -> ``{ids: [L, T, top_k] int16, input_ids: [T] int32}``.

    Layers are stacked in ascending layer_id, which is the order the decode loop walks and
    therefore the order the simulator must replay them in.
    """
    layers = data["layers"]
    if not layers:
        return None
    layer_ids = sorted(layers)
    per_layer = []
    for lid in layer_ids:
        logits = layers[lid]["logits"]
        # bf16 -> fp32 before topk: bf16 has 8 mantissa bits and router logits routinely tie
        # within that, which would make the selected SET depend on tie-break order.
        top = torch.topk(logits.float(), top_k, dim=-1).indices
        per_layer.append(top.to(torch.int16))
    return {
        "ids": torch.stack(per_layer),  # [L, T, top_k]
        "input_ids": data["input_ids"].to(torch.int32),
        "layer_ids": torch.tensor(layer_ids, dtype=torch.int32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--min-tokens", type=int, default=256,
                    help="skip warmup/boot forwards smaller than this")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "route_*.pt")))
    if not files:
        raise SystemExit(f"no route_*.pt in {args.dump_dir}")

    chunks, skipped, t0 = [], 0, time.time()
    for i, path in enumerate(files):
        nbytes = os.path.getsize(path)
        data = torch.load(path, map_location="cpu", weights_only=True)
        n_tok = int(data["input_ids"].shape[0])
        if n_tok < args.min_tokens:
            skipped += 1
            del data
            continue
        out = compact_chunk(data, args.top_k)
        del data  # the 'r'/'block_input' tensors dominate; drop them before the next load
        if out is None:
            skipped += 1
            continue
        out["path"] = path
        chunks.append(out)
        L, T, K = out["ids"].shape
        print(f"[{i + 1}/{len(files)}] {os.path.basename(path)}  "
              f"{nbytes / 2**30:.2f} GiB -> L={L} T={T} K={K}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    if not chunks:
        raise SystemExit("no usable chunks")
    torch.save({"chunks": chunks, "top_k": args.top_k}, args.out)
    total = sum(int(c["ids"].shape[1]) for c in chunks)
    print(f"\nwrote {args.out}: {len(chunks)} chunks, {total} tokens, "
          f"{os.path.getsize(args.out) / 2**20:.1f} MiB ({skipped} skipped)")


if __name__ == "__main__":
    main()
