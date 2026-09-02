"""Triton kernels for Qwen3.8-Flash-Next QSA sparse attention."""

from .attend import qsa_sparse_paged_attention
from .compress import qsa_compress_groups, qsa_index_norm_rope, qsa_store_rows
from .expand import expand_qsa_block_indices
from .fused_decode import qsa_index_write_fused, qsa_plan_index_writes, qsa_qk_norm_rope_fused
from .score import qsa_mqa_paged
from .store_fp8 import qsa_store_kv_fp8
from .store_nvfp4 import qsa_store_kv_nvfp4
from .topk import qsa_block_topk, qsa_block_topk_scratch_width

__all__ = [
    "qsa_index_write_fused",
    "qsa_plan_index_writes",
    "qsa_qk_norm_rope_fused",
    "expand_qsa_block_indices",
    "qsa_block_topk",
    "qsa_block_topk_scratch_width",
    "qsa_compress_groups",
    "qsa_index_norm_rope",
    "qsa_mqa_paged",
    "qsa_sparse_paged_attention",
    "qsa_store_kv_fp8",
    "qsa_store_kv_nvfp4",
    "qsa_store_rows",
]
