"""Qwen3.8-Flash-Next (RadixArk NVFP4) checkpoint reader.

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`load_nvfp4_expert_sources` -- the routed NVFP4 experts, into the offload cache's source banks.

Dropped: ``model.visual.*`` (served text-only), and -- unless ``mtp_load_enabled()`` -- the
``mtp.*`` speculative head (including its stacked bf16 ``mtp.layers.0.mlp.experts.*``, which load
as the resident draft-MoE tensors when the head is on).
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import mtp_load_enabled
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import init_logger

logger = init_logger(__name__)
from freetoken.utils import download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


# --dense-quant fp8 with the ``gdn`` part: the GDN layer is built as an fp8 ``in_proj_qkvz``
# (Fp8PerTensorColMerged) plus a bf16 ``in_proj_ba`` (gdn.py), so the four checkpoint parts
# fuse into those two tensors instead of the single bf16 ``in_proj``.
_FUSIONS_FP8_GDN: dict[str, tuple[tuple[str, ...], int]] = {
    k: v for k, v in _FUSIONS.items() if k != ".linear_attn.in_proj.weight"
}
_FUSIONS_FP8_GDN[".linear_attn.in_proj_qkvz.weight"] = ((
    ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
), 0)
_FUSIONS_FP8_GDN[".linear_attn.in_proj_ba.weight"] = ((
    ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
), 0)


def _fusions_for(parts) -> dict[str, tuple[tuple[str, ...], int]]:
    return _FUSIONS_FP8_GDN if "gdn" in parts else _FUSIONS


def _emit_dense(name: str, tensor: torch.Tensor, parts) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield a dense tensor as loaded, or -- when ``name`` is a load-time FP8 target under
    ``parts`` -- its per-output-row fp8 codes plus the ``.weight_scale`` sibling the
    Fp8PerTensor* module / Fp8LMHead expects."""
    from freetoken.models.qwen4_exp.config import dense_fp8_part_of

    if name == "lm_head.weight" and os.getenv("FREETOKEN_LM_HEAD_NVFP4", "0") == "1":
        # NVFP4 target head (model.py): emit the resident triple under lm_head.* and no draft copy
        packed, scales, globals_ = [], [], []
        for r0 in range(0, tensor.shape[0], 16384):
            p, s, g = _quantize_nvfp4_stacked(tensor[r0 : r0 + 16384].unsqueeze(0))
            packed.append(p[0]); scales.append(s[0]); globals_.append(g[0])
        yield "lm_head.weight", torch.cat(packed, dim=0)
        yield "lm_head.weight_scale", torch.cat(scales, dim=0)
        yield "lm_head.weight_global", torch.cat(globals_, dim=0)
        return
    if name == "lm_head.weight":
        from freetoken.models.config import draft_lm_head_enabled

        if draft_lm_head_enabled():
            # NVFP4 copy for the speculative draft passes (model.draft_lm_head), quantized in
            # row chunks to bound the fp32 temporaries; the target head follows below.
            packed, scales, globals_ = [], [], []
            for r0 in range(0, tensor.shape[0], 16384):
                p, s, g = _quantize_nvfp4_stacked(tensor[r0 : r0 + 16384].unsqueeze(0))
                packed.append(p[0]); scales.append(s[0]); globals_.append(g[0])
            yield "draft_lm_head.weight", torch.cat(packed, dim=0)
            yield "draft_lm_head.weight_scale", torch.cat(scales, dim=0)
            yield "draft_lm_head.weight_global", torch.cat(globals_, dim=0)
    if dense_fp8_part_of(name, parts) is None:
        yield name, tensor
        return
    from freetoken.kernel.triton.fp8_pertensor_linear import quantize_fp8_per_row

    assert name.endswith(".weight"), name
    q, scale = quantize_fp8_per_row(tensor)
    yield name, q
    yield name[: -len(".weight")] + ".weight_scale", scale


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        # The draft head is opt-in (see mtp_load_enabled). It is bf16 end to end -- no scales,
        # and its stacked ``mlp.experts.{gate_up_proj,down_proj}`` load as the resident MoE
        # tensors -- and its keys already match the model's ``mtp.`` subtree, so it passes
        # through untouched; the standard _FUSIONS suffix-merges apply to ``mtp.layers.0.*``.
        return raw_name if mtp_load_enabled() else None
    if raw_name.startswith(("model.visual.", "visual.")):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


_MTP_EXPERT_KEYS = ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj")


def mtp_expert_format() -> str:
    """Where the MTP draft experts live. ``offload`` (default): NVFP4-quantized at load and
    appended to the unified expert offload cache as one extra bank layer -- zero resident
    VRAM (falls back to ``fp8_block`` under non-offload MoE backends, where no cache
    exists). ``fp8``: resident block-FP8 (~2.3 GiB). ``bf16``: resident unquantized
    (~4.7 GiB). Only the DRAFT distribution is affected by quantization -- chain
    speculative sampling keeps the emitted distribution exact regardless."""
    value = os.getenv("FREETOKEN_MTP_EXPERTS", "offload").strip().lower()
    formats = {"offload": "offload", "fp8": "fp8_block", "bf16": "bf16"}
    if value not in formats:
        raise ValueError(f"FREETOKEN_MTP_EXPERTS must be one of {sorted(formats)}, got {value!r}")
    return formats[value]


def _quantize_fp8_block(w: torch.Tensor, block: int = 128, chunk: int = 32):
    """Per-128x128-block symmetric FP8-E4M3 quantization of stacked experts [E, R, C]
    (bf16) -> (fp8 weights, bf16 dequant scales [E, R/block, C/block]). Chunked over E to
    bound the fp32 temporaries."""
    from freetoken.kernel.triton.fp8_block_linear import FP8

    e, r, c = w.shape
    assert r % block == 0 and c % block == 0, (r, c)
    q = torch.empty(e, r, c, dtype=FP8, device=w.device)
    scale = torch.empty(e, r // block, c // block, dtype=torch.bfloat16, device=w.device)
    for i in range(0, e, chunk):
        part = w[i : i + chunk].float().view(-1, r // block, block, c // block, block)
        amax = part.abs().amax(dim=(2, 4)).clamp_min(1e-12)
        s = amax / 448.0
        q[i : i + chunk] = (
            (part / s[:, :, None, :, None]).clamp(-448.0, 448.0)
            .view(-1, r, c)
            .to(FP8)
        )
        scale[i : i + chunk] = s.to(torch.bfloat16)
    return q, scale


# Decision bounds between adjacent e2m1 magnitude levels [0, .5, 1, 1.5, 2, 3, 4, 6].
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _quantize_nvfp4_stacked(w: torch.Tensor, chunk: int = 16):
    """Stacked experts [E, R, C] (bf16) -> the NVFP4 bank-row triple the offload decode
    kernels read (kernel/triton/nvfp4_fused_moe.py): packed e2m1 codes ([E, R, C/2] uint8,
    low nibble = even input, sign in bit 3), FP8-E4M3 block scales (one per 16 inputs,
    [E, R, C/16]) and per-output-row FP16 globals, with dequant = lut[code] * block_scale
    * global. Chunked over E to bound the fp32 temporaries."""
    e, r, c = w.shape
    assert c % 16 == 0, (r, c)
    packed = torch.empty(e, r, c // 2, dtype=torch.uint8, device=w.device)
    scale = torch.empty(e, r, c // 16, dtype=torch.float8_e4m3fn, device=w.device)
    glob = torch.empty(e, r, dtype=torch.float16, device=w.device)
    bounds = torch.tensor(_E2M1_MIDPOINTS, device=w.device)
    for i in range(0, e, chunk):
        x = w[i : i + chunk].float()
        g = (x.abs().amax(dim=-1) / (6.0 * 448.0)).clamp_min(2**-24).to(torch.float16)
        blocks = x.view(-1, r, c // 16, 16)
        s = blocks.abs().amax(dim=-1) / (6.0 * g.float().unsqueeze(-1))
        # the min clamp keeps all-zero blocks at a nonzero scale so q stays finite (codes 0)
        s8 = s.clamp(2**-6, 448.0).to(torch.float8_e4m3fn)
        q = blocks / (s8.float() * g.float().unsqueeze(-1)).unsqueeze(-1)
        mag = torch.bucketize(q.abs(), bounds).to(torch.uint8)
        code = torch.where(q < 0, mag + 8, mag).view(-1, r, c)
        packed[i : i + chunk] = code[..., 0::2] | (code[..., 1::2] << 4)
        scale[i : i + chunk] = s8
        glob[i : i + chunk] = g
    return packed, scale, glob


_MTP_EXPERT_ROLES = {
    "mtp.layers.0.mlp.experts.gate_up_proj": "gate_up",
    "mtp.layers.0.mlp.experts.down_proj": "down",
}


def build_mtp_offload_bank_layer(
    model_path: str, config, device: torch.device, *, dummy: bool = False
) -> dict[str, torch.Tensor]:
    """One extra NVFP4 bank layer (pinned host, offload-cache schema) holding the MTP
    draft experts, keyed by bank name. The checkpoint ships them stacked bf16
    ([E, 2I, H] / [E, H, I]); quantizing into the cache's own format lets the draft
    layer share slots with the main tower instead of costing resident VRAM."""
    e = config.num_experts
    h, i = config.hidden_size, config.moe_intermediate_size
    shapes = {
        "gate_up": ((e, 2 * i, h // 2), (e, 2 * i, h // 16), (e, 2 * i)),
        "down": ((e, h, i // 2), (e, h, i // 16), (e, h)),
    }
    out: dict[str, torch.Tensor] = {}
    if dummy:
        for role, (p, s, gl) in shapes.items():
            out[f"{role}_packed"] = torch.randint(0, 256, p, dtype=torch.uint8)
            out[f"{role}_scale"] = torch.ones(s).to(torch.float8_e4m3fn)
            out[f"{role}_global"] = torch.full(gl, 2**-8, dtype=torch.float16)
        return {name: t.pin_memory() for name, t in out.items()}
    found: set[str] = set()
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for raw_name in f.keys():
                role = _MTP_EXPERT_ROLES.get(raw_name)
                if role is None:
                    continue
                packed, scale, glob = _quantize_nvfp4_stacked(f.get_tensor(raw_name).to(device))
                out[f"{role}_packed"] = packed.cpu().pin_memory()
                out[f"{role}_scale"] = scale.cpu().pin_memory()
                out[f"{role}_global"] = glob.cpu().pin_memory()
                found.add(role)
    assert found == set(shapes), f"MTP stacked experts missing from checkpoint: {sorted(set(shapes) - found)}"
    for role, (p, s, gl) in shapes.items():
        assert tuple(out[f"{role}_packed"].shape) == p, (role, out[f"{role}_packed"].shape, p)
    return out


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]],
    fusions: dict[str, tuple[tuple[str, ...], int]] | None = None,
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in (_FUSIONS if fusions is None else fusions).items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: the modelopt
    ``ignore`` list covers everything except those experts, so attention, GDN, HC, PLE, the shared
    expert and lm_head are all plain bf16 (the n-gram hash constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from :func:`load_nvfp4_expert_sources`.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    from freetoken.models.config import dense_fp8_parts

    fp8_parts = dense_fp8_parts()
    fusions = _fusions_for(fp8_parts)
    if fp8_parts:
        logger.info_rank0(
            "dense quant: per-channel FP8 at load for parts "
            + ",".join(sorted(fp8_parts))
            + " (FREETOKEN_DENSE_QUANT=fp8; FREETOKEN_DENSE_FP8_PARTS narrows the set)"
        )
    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                if name in _MTP_EXPERT_KEYS and mtp_expert_format() == "offload":
                    # Served from the unified offload cache as an extra bank layer
                    # (build_mtp_offload_bank_layer); nothing resident to load.
                    continue
                tensor = f.get_tensor(raw_name)
                if name in _MTP_EXPERT_KEYS:
                    if mtp_expert_format() == "fp8_block":
                        # Draft experts quantized at load: the resident MoE layer is built
                        # fp8_block and expects the matching weight + dequant-scale pair.
                        quantized, scale = _quantize_fp8_block(tensor)
                        yield name, quantized
                        yield name.replace("_proj", "_scale_inv"), scale
                        continue
                fused = _try_fuse(name, tensor, fuse_buf, fusions)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield from _emit_dense(fused[0], fused[1], fp8_parts)
                    continue
                yield from _emit_dense(name, tensor, fp8_parts)

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` float8_e4m3fn view of the bank."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    bank = HostBank((expected * rows, cols), torch.float8_e4m3fn)
    shard_bytes = rows * cols
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up fused on the output-row axis, down separate; weight_scale_2 carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
) -> dict:
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "PleTable",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
]
