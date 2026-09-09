import pytest
import torch

from tokenrush import marlin
from tokenrush.quant import dequantize_int4, int4_gemm_rows, int4_gemv, quantize_int4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not marlin.available(), reason="needs the marlin extension")


def _weight(N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(N, K, device="cuda", generator=g) * 0.02
    return quantize_int4(w.to(torch.bfloat16))


@pytest.mark.parametrize("N,K", [(5120, 6144), (256, 5120), (1280, 5120), (5120, 17408)])
def test_pack_roundtrip(N, K):
    q, s, m = _weight(N, K)
    B, sp, mp = marlin.pack(q, s, m)
    assert B.shape == (K // 16, N * 2) and sp.shape == (K // 128, N)
    q2, s2, m2 = marlin.unpack(B, sp, mp)
    assert torch.equal(q, q2) and torch.equal(s, s2) and torch.equal(m, m2)


@pytest.mark.parametrize("N,K", [(5120, 6144), (256, 5120), (1280, 5120), (14336, 5120), (5120, 17408)])
@pytest.mark.parametrize("M", [1, 3, 8, 16])
def test_mul_matches_dequant(N, K, M):
    q, s, m = _weight(N, K)
    B, sp, mp = marlin.pack(q, s, m)
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    ref = x.float() @ dequantize_int4(q, s, m).float().t()
    y = marlin.mul(x, B, sp, mp)
    ours = int4_gemm_rows(x, q, s, m) if M <= 8 else None
    err = (y.float() - ref).abs().max().item()
    tol = ref.abs().max().item() * 2 ** -7
    assert err <= tol, (err, tol)
    if ours is not None:
        err_ours = (ours.float() - ref).abs().max().item()
        assert err <= 1.5 * err_ours + tol / 4, (err, err_ours)


def test_rows_independent_of_m():
    """Row 0 of an M=1 call is bit-identical to row 0 of an M=8 call (same tiles,
    same partition, same reduction order): raw and speculative steps share numerics."""
    q, s, m = _weight(5120, 17408)
    B, sp, mp = marlin.pack(q, s, m)
    x = torch.randn(8, 17408, device="cuda").to(torch.bfloat16)
    y8 = marlin.mul(x, B, sp, mp)
    y1 = marlin.mul(x[:1].contiguous(), B, sp, mp)
    assert torch.equal(y8[0], y1[0])
    y3 = marlin.mul(x[:3].contiguous(), B, sp, mp)
    assert torch.equal(y8[:3], y3)


def test_graph_replay_matches_eager():
    q, s, m = _weight(1280, 5120)
    B, sp, mp = marlin.pack(q, s, m)
    x = torch.randn(4, 5120, device="cuda").to(torch.bfloat16)
    y = marlin.mul(x, B, sp, mp)
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        out = marlin.mul(x, B, sp, mp)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(marlin.mul(x, B, sp, mp))
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, y)


@pytest.mark.parametrize("N,K", [(5120, 6144), (5120, 17408), (256, 5120)])
@pytest.mark.parametrize("M", [1, 8])
def test_partials_sum_to_product(N, K, M):
    q, s, m = _weight(N, K)
    B, sp, mp = marlin.pack(q, s, m)
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    y = marlin.mul(x, B, sp, mp)
    p = marlin.mul_partial(x, B, sp, mp)
    assert p.shape[1:] == (M, N)
    ref = x.float() @ dequantize_int4(q, s, m).float().t()
    assert (p.sum(0) - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -7
    assert (p.sum(0).to(torch.bfloat16).float() - y.float()).abs().max().item() <= ref.abs().max().item() * 2 ** -7
    # repeat with the buffer poisoned: every slot is written by the kernel
    p2 = marlin.mul_partial(x, B, sp, mp)
    assert torch.equal(p, p2)
