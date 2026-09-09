"""Marlin-layout int4 GEMM for M <= 16 rows (Phase 3c).

The kernel is a port of Marlin (IST-DASLab, Apache-2.0) in `csrc/marlin_bf16.cu`:
bf16 activations, our asymmetric g128 format (w = q * s + m), fp32 cross-block
reduction, no L2 cache hints (illegal on sm_120). It reads the weight once and
dots it with all M rows on the tensor cores, at the same bandwidth for M = 1
and M = 16, where our Triton M-row kernel loses 15-20% by M = 8.

The weight layout is Marlin's: [K/16, N*2] int32 holding 16x16 tiles in the
fragment order the mma instruction wants, scales and minimums [K/128, N] bf16
with columns permuted within groups of 64. `pack` and `unpack` convert to and
from the engine's own packing (`quant.py`).

The extension is built on first import (torch.utils.cpp_extension.load, ~10 s,
cached under csrc/build/).
"""
import os

import torch

_ext = None
_HERE = os.path.dirname(os.path.abspath(__file__))


def ext():
    """The compiled extension (built on first use)."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        build = os.path.join(_HERE, "csrc", "build")
        os.makedirs(build, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _ext = load(name="marlin_bf16", sources=[os.path.join(_HERE, "csrc", "marlin_bf16.cpp"),
                                                 os.path.join(_HERE, "csrc", "marlin_bf16.cu")],
                    extra_cuda_cflags=["-O3"], build_directory=build, verbose=False)
    return _ext


def available() -> bool:
    try:
        ext()
        return True
    except Exception as e:  # no nvcc, wrong arch, ...
        print(f"marlin extension unavailable: {str(e)[:200]}")
        return False


def _perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend(p + 256 * j for p in perm1)
    perm = torch.tensor(perm)
    perm = perm.view(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(-1)
    scale_perm = torch.tensor([i + 8 * j for i in range(8) for j in range(8)])
    return perm, torch.argsort(perm), scale_perm, torch.argsort(scale_perm)


_PERM, _INV_PERM, _SPERM, _INV_SPERM = _perms()
CHUNK = 4096            # output columns packed per pass (bounds the int32 temporaries)


def _to_int32(q: torch.Tensor) -> torch.Tensor:
    """int64 bit patterns in [0, 2^32) -> int32."""
    return torch.where(q >= 2 ** 31, q - 2 ** 32, q).to(torch.int32)


def pack(qweight: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor):
    """Engine packing (uint8 [N, K/2], bf16 [N, K/128] x2) -> Marlin layout
    (int32 [K/16, N*2], bf16 [K/128, N] x2)."""
    from .quant import unpack_int4
    N, K = qweight.shape[0], qweight.shape[1] * 2
    assert N % 64 == 0 and K % 128 == 0, (N, K)
    dev = qweight.device
    perm, sperm = _PERM.to(dev), _SPERM.to(dev)
    codes = unpack_int4(qweight)                                       # [N, K]
    Bs, Ss, Ms = [], [], []
    for c0 in range(0, N, CHUNK):
        c1 = min(N, c0 + CHUNK)
        ch = c1 - c0
        q = codes[c0:c1].t().to(torch.int64)                            # [K, ch]
        q = q.reshape(K // 16, 16, ch // 16, 16).permute(0, 2, 1, 3).reshape(K // 16, ch * 16)
        q = q.reshape(-1, perm.numel())[:, perm].reshape(K // 16, ch * 2, 8)
        B = torch.zeros(K // 16, ch * 2, dtype=torch.int64, device=dev)
        for i in range(8):
            B |= q[..., i] << (4 * i)
        Bs.append(_to_int32(B))
        for src, dst in ((scale, Ss), (mn, Ms)):
            t = src[c0:c1].t()                                          # [G, ch]
            dst.append(t.reshape(-1, 64)[:, sperm].reshape(t.shape[0], ch))
    return (torch.cat(Bs, 1).contiguous(), torch.cat(Ss, 1).contiguous(), torch.cat(Ms, 1).contiguous())


def unpack(B: torch.Tensor, s: torch.Tensor, m: torch.Tensor):
    """Inverse of pack -> (qweight uint8 [N, K/2], scale bf16 [N, G], mn bf16 [N, G])."""
    Kt, N2 = B.shape
    K, N = Kt * 16, N2 // 2
    dev = B.device
    inv, sinv = _INV_PERM.to(dev), _INV_SPERM.to(dev)
    Qs, Ss, Ms = [], [], []
    for c0 in range(0, N, CHUNK):
        c1 = min(N, c0 + CHUNK)
        ch = c1 - c0
        Bc = B[:, 2 * c0:2 * c1]
        q = torch.stack([(Bc >> (4 * i)) & 0xF for i in range(8)], -1).reshape(Kt, ch * 16)
        q = q.reshape(-1, inv.numel())[:, inv].reshape(Kt, ch // 16, 16, 16)
        q = q.permute(0, 2, 1, 3).reshape(K, ch).t()                    # [ch, K] int32 codes
        q = q.reshape(ch, K // 2, 2)
        Qs.append((q[..., 0] | (q[..., 1] << 4)).to(torch.uint8))
        for src, dst in ((s, Ss), (m, Ms)):
            t = src[:, c0:c1]                                           # [G, ch]
            dst.append(t.reshape(-1, 64)[:, sinv].reshape(t.shape[0], ch).t())
    return torch.cat(Qs).contiguous(), torch.cat(Ss).contiguous(), torch.cat(Ms).contiguous()


_ws = {}


def workspaces(N: int, device):
    """Per-device lock array and fp32 reduce buffer, grown to fit N columns.
    Grow before graph capture: QLinear calls this at construction."""
    key = torch.device(device).index
    locks, red = _ws.get(key, (None, None))
    n_locks = max(4096, N // 128)
    if locks is None or locks.numel() < n_locks:
        locks = torch.zeros(n_locks, dtype=torch.int32, device=device)
    if red is None or red.numel() < 16 * N:
        red = torch.empty(16 * N, dtype=torch.float32, device=device)
    _ws[key] = (locks, red)
    return locks, red


def mul(x: torch.Tensor, B: torch.Tensor, s: torch.Tensor, m: torch.Tensor, thread_k=-1, thread_n=-1, sms=-1):
    """x [M, K] bf16 (M <= 16) -> [M, N] bf16."""
    M, K = x.shape
    N = B.shape[1] // 2
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    locks, red = workspaces(N, x.device)
    ext().mul(x, B, y, s, m, locks, red, thread_k, thread_n, sms)
    return y


_slots = {}


def partial_slots(N: int, K: int) -> int:
    key = (N, K)
    if key not in _slots:
        _slots[key] = ext().partial_slots(N, K)
    return _slots[key]


def mul_partial(x: torch.Tensor, B: torch.Tensor, s: torch.Tensor, m: torch.Tensor):
    """x [M, K] bf16 (M <= 16) -> fp32 partials [S, M, N] whose sum over S is the
    product. No cross-block reduction inside the kernel: each block writes its own
    slot, which on the narrow (N = 5120) shapes removes a chain of 4-5 serialized
    block-to-block reductions from the kernel's tail."""
    M, K = x.shape
    N = B.shape[1] // 2
    y = torch.empty(partial_slots(N, K), M, N, dtype=torch.float32, device=x.device)
    locks, red = workspaces(N, x.device)
    ext().mul(x, B, y, s, m, locks, red)
    return y

