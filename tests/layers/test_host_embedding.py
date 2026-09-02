"""HostUVAEmbedding: pinned-host table gathered over UVA == torch embedding lookup."""
from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_host_uva_embedding_matches_lookup():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.embedding import HostUVAEmbedding
    from freetoken.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    V, H = 4096, 2560
    torch.manual_seed(0)
    table = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        emb = HostUVAEmbedding(V, H)
    emb.load_state_dict({"weight": table})
    assert emb.weight.device.type == "cpu" and emb.weight.is_pinned()
    assert set(emb.state_dict().keys()) == {"weight"}
    for shape in [(1,), (3,), (2, 5), (1024,)]:
        ids = torch.randint(0, V, shape, device="cuda", dtype=torch.int64)
        out = emb.forward(ids)
        torch.cuda.synchronize()
        assert out.shape == (*shape, H) and out.dtype == torch.bfloat16
        assert torch.equal(out, torch.nn.functional.embedding(ids, table))
    # int32 ids (the engine's input_ids) and CUDA-graph capture + replay with new ids
    ids = torch.randint(0, V, (3,), device="cuda", dtype=torch.int32)
    emb.forward(ids); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = emb.forward(ids)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        ids.copy_(torch.randint(0, V, (3,), device="cuda", dtype=torch.int32))
        g.replay(); torch.cuda.synchronize()
        assert torch.equal(out, torch.nn.functional.embedding(ids.long(), table))
