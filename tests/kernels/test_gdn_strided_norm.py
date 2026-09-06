"""GDN output norm must read the sliced qkvz gate without changing its arithmetic."""
import pytest
import torch

from freetoken.kernel.fla.layernorm_gated import rms_norm_gated

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _inputs(tokens):
    torch.manual_seed(tokens)
    x = torch.randn(tokens, 48 * 128, device="cuda", dtype=torch.bfloat16)
    qkvz = torch.randn(tokens, 16384, device="cuda", dtype=torch.bfloat16)
    z = qkvz[:, 10240:]
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    return x, z, weight


def _norm(x, z, weight, activation, grouped):
    return rms_norm_gated(
        x=x if grouped else x.reshape(-1, 128),
        z=z if grouped else z.reshape(-1, 128),
        weight=weight,
        bias=None,
        group_size=128 if grouped else None,
        eps=1e-6,
        is_rms_norm=True,
        norm_before_gate=True,
        activation=activation,
    ).reshape_as(x)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4])
@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_strided_gdn_norm_matches_flattened_exactly(tokens, activation):
    x, z, weight = _inputs(tokens)
    assert z.stride() == (16384, 1)
    old = _norm(x, z, weight, activation, False)
    new = _norm(x, z, weight, activation, True)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    # Full-width affine weights keep the original grouped API behavior too.
    full_weight = weight.repeat(48)
    full = _norm(x, z, full_weight, activation, True)
    torch.testing.assert_close(new, full, rtol=0, atol=0)


def test_strided_gdn_norm_graph_replay_matches_flattened():
    x, z, weight = _inputs(4)
    for grouped in (False, True):
        _norm(x, z, weight, "sigmoid", grouped)
    torch.cuda.synchronize()
    graphs, outputs = [], []
    for grouped in (False, True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _norm(x, z, weight, "sigmoid", grouped)
        graphs.append(graph)
        outputs.append(out)
    for _ in range(5):
        x.normal_()
        z.normal_()
        for graph in graphs:
            graph.replay()
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
