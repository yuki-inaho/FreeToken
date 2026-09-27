"""Locate the Engram embedding tables inside the shipped safetensors shards.

The two tables (``layers.{1,14}.engram.embed.{weight,scale}``, FP8 E4M3 ``[rows, 256]`` and
E8M0 ``[rows, 8]``) are 2 x 101 GB and are never loaded; they are read row by row from NVMe
through the page cache. All that needs is the byte offset of each tensor inside its shard:
safetensors is ``u64 header_len | JSON header | data``, and a tensor's bytes start at
``8 + header_len + data_offsets[0]``. Row ``i`` of a row-major uint8 table is then at
``offset + i * row_bytes`` — no safetensors library, no mmap of the whole file.

Dtypes and shapes are validated against the header because a wrong stride would read
plausible-looking garbage. The shards may live in ``FREETOKEN_ENGRAM_DIR`` (an ext4 copy of just
the two files, for random-read speed) instead of the checkpoint directory.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

ENGRAM_DIR_ENV = "FREETOKEN_ENGRAM_DIR"
WEIGHT_DTYPE = "F8_E4M3"
SCALE_DTYPE = "F8_E8M0"


def read_safetensors_header(path: str) -> tuple[int, dict]:
    """``(header_len, header)`` of a safetensors file; tensor data starts at ``8 + header_len``."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (short header)")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header_len, header


@dataclass(frozen=True)
class EngramTableLoc:
    """Byte layout of one layer's table: row ``i`` of the codes is at ``weight_offset + i*dim`` in
    ``path``; its ``scale_cols`` scale codes at ``scale_offset + i*scale_cols`` in ``scale_path``."""

    path: str
    rows: int
    dim: int
    weight_offset: int
    scale_offset: int
    scale_cols: int
    scale_path: str = ""
    layer_id: int = -1

    def __post_init__(self) -> None:
        if not self.scale_path:
            object.__setattr__(self, "scale_path", self.path)

    @property
    def block(self) -> int:
        """Elements per scale code (32 for the V4.1 tables)."""
        return self.dim // self.scale_cols

    @property
    def weight_bytes(self) -> int:
        return self.rows * self.dim

    @property
    def scale_bytes(self) -> int:
        return self.rows * self.scale_cols


def _tensor_span(path: str, header_len: int, header: dict, name: str, dtype: str) -> tuple[int, int, list[int]]:
    """``(abs_offset, nbytes, shape)`` of ``name`` in the shard, checking its dtype."""
    entry = header.get(name)
    if entry is None:
        raise KeyError(f"{path}: tensor {name!r} not in header")
    if entry["dtype"] != dtype:
        raise ValueError(f"{path}: {name} is {entry['dtype']}, expected {dtype}")
    start, end = entry["data_offsets"]
    shape = [int(s) for s in entry["shape"]]
    n = 1
    for s in shape:
        n *= s
    if end - start != n:  # both dtypes are one byte per element
        raise ValueError(f"{path}: {name} spans {end - start} bytes for shape {shape}")
    return 8 + header_len + int(start), int(end - start), shape


def _find_shard(shard: str, dirs: list[str]) -> str:
    for d in dirs:
        if not d:
            continue
        p = os.path.join(d, shard)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"engram shard {shard} not found in {[d for d in dirs if d]}")


def resolve_engram_tables(
    model_path: str,
    layer_ids,
    engram_dir: str | None = None,
    index_json: str | None = None,
    *,
    expected_rows: dict[int, int] | None = None,
    expected_dim: int | None = None,
) -> dict[int, EngramTableLoc]:
    """Map each engram layer id to its :class:`EngramTableLoc`.

    Shard names come from ``model.safetensors.index.json``; each shard is looked up in
    ``engram_dir`` (argument, else ``$FREETOKEN_ENGRAM_DIR``) and then ``model_path``.
    ``expected_rows`` (``{layer_id: num_embeddings}``) and ``expected_dim`` (head_dim) pin the
    table geometry to the args when given.
    """
    if engram_dir is None:
        engram_dir = os.environ.get(ENGRAM_DIR_ENV) or None
    if index_json is None:
        index_json = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_json) as f:
        weight_map = json.load(f)["weight_map"]
    dirs = [engram_dir or "", model_path]
    headers: dict[str, tuple[int, dict]] = {}

    def header_of(path: str) -> tuple[int, dict]:
        if path not in headers:
            headers[path] = read_safetensors_header(path)
        return headers[path]

    out: dict[int, EngramTableLoc] = {}
    for layer_id in layer_ids:
        layer_id = int(layer_id)
        wname = f"layers.{layer_id}.engram.embed.weight"
        sname = f"layers.{layer_id}.engram.embed.scale"
        for name in (wname, sname):
            if name not in weight_map:
                raise KeyError(f"{index_json}: {name} not in weight_map")
        wpath = _find_shard(weight_map[wname], dirs)
        spath = _find_shard(weight_map[sname], dirs)
        w_off, w_bytes, w_shape = _tensor_span(wpath, *header_of(wpath), wname, WEIGHT_DTYPE)
        s_off, s_bytes, s_shape = _tensor_span(spath, *header_of(spath), sname, SCALE_DTYPE)
        if len(w_shape) != 2 or len(s_shape) != 2:
            raise ValueError(f"layer {layer_id}: engram tables must be 2-D, got {w_shape} / {s_shape}")
        rows, dim = w_shape
        s_rows, scale_cols = s_shape
        if s_rows != rows:
            raise ValueError(f"layer {layer_id}: weight has {rows} rows, scale has {s_rows}")
        if scale_cols <= 0 or dim % scale_cols:
            raise ValueError(f"layer {layer_id}: dim {dim} is not a multiple of {scale_cols} scale blocks")
        if expected_dim is not None and dim != int(expected_dim):
            raise ValueError(f"layer {layer_id}: table dim {dim} != engram_head_dim {expected_dim}")
        if expected_rows is not None and layer_id in expected_rows and rows != int(expected_rows[layer_id]):
            raise ValueError(f"layer {layer_id}: table has {rows} rows, args say {expected_rows[layer_id]}")
        for path, off, nbytes in ((wpath, w_off, w_bytes), (spath, s_off, s_bytes)):
            size = os.path.getsize(path)
            if off + nbytes > size:
                raise ValueError(f"{path}: tensor ends at {off + nbytes} but the file is {size} bytes (truncated?)")
        out[layer_id] = EngramTableLoc(
            path=wpath,
            rows=rows,
            dim=dim,
            weight_offset=w_off,
            scale_offset=s_off,
            scale_cols=scale_cols,
            scale_path=spath,
            layer_id=layer_id,
        )
    return out


__all__ = ["ENGRAM_DIR_ENV", "EngramTableLoc", "read_safetensors_header", "resolve_engram_tables"]
