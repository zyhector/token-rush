"""The M-row int4 GEMV against mx.quantized_matmul: correctness and cost relative
to MLX's M = 1 path, in-graph over > 1 GB of distinct weights per shape.

    uv run python bench/mac/qmv_rows.py [--rps 4 --nsg 2 --vpt 8]
"""
import argparse
import time

import mlx.core as mx

from tokenrush.mlx.kernels import nax_split, qmm8, qmm_nax, qmv_rows

SHAPES = [("gdn.in_qkvz", 16384, 5120), ("gdn.out", 5120, 6144), ("attn.qkv", 14336, 5120),
          ("mlp.gate_up", 34816, 5120), ("mlp.down", 5120, 17408), ("lm_head", 248320, 5120)]


def per_matrix(fn, mats, iters=6):
    def chain():
        return mx.stack([fn(*m) for m in mats]).sum(0)
    for _ in range(2):
        mx.eval(chain())
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(chain())
        best = min(best, (time.perf_counter() - t0) / iters)
    return best / len(mats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=int, default=4)
    ap.add_argument("--nsg", type=int, default=2)
    ap.add_argument("--vpt", type=int, default=8)
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--kernel", default="rows", choices=["rows", "mma", "nax"])
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--target", type=int, default=2048)
    a = ap.parse_args()
    ms = [int(m) for m in a.ms.split(",")]
    print(f"rps={a.rps} nsg={a.nsg} vpt={a.vpt}; cost relative to mx.quantized_matmul at M=1")
    print(f"{'matrix':<12} " + " ".join(f"{'M=' + str(m):>14}" for m in ms) + "   max rel err")
    for name, N, K in SHAPES:
        w = mx.random.normal((N, K), dtype=mx.bfloat16) * 0.05
        q = mx.quantize(w, group_size=128, bits=4)
        del w
        nb = sum(t.nbytes for t in q)
        n = max(1, int(1.2e9 / nb))
        mats = [tuple(mx.array(t + 0) for t in q) for _ in range(n)]
        mx.eval(mats)
        x1 = mx.random.normal((1, K), dtype=mx.bfloat16)
        base = per_matrix(lambda a, b, c: mx.quantized_matmul(x1, a, b, c, transpose=True, group_size=128, bits=4), mats)
        cells, err = [], 0.0
        for M in ms:
            x = mx.random.normal((M, K), dtype=mx.bfloat16)
            ref = mx.quantized_matmul(x, *q, transpose=True, group_size=128, bits=4).astype(mx.float32)
            ours = (lambda x_, a_, b, c: qmv_rows(x_, a_, b, c, rps=a.rps, nsg=a.nsg, vpt=a.vpt)) if a.kernel == "rows" else (lambda x_, a_, b, c: qmm8(x_, a_, b, c, nsg=a.nsg, nt=a.rps))
            if a.kernel == "nax":
                sp = a.split or nax_split(N, K, a.target)
                ours = lambda x_, a_, b, c: qmm_nax(x_, a_, b, c, nsg=a.nsg, split=sp)
            got = ours(x, *q).astype(mx.float32)
            err = max(err, (mx.abs(got - ref).max() / (mx.abs(ref).max() + 1e-6)).item())
            t_ours = per_matrix(lambda a_, b, c: ours(x, a_, b, c), mats)
            t_mlx = per_matrix(lambda a_, b, c: mx.quantized_matmul(x, a_, b, c, transpose=True, group_size=128, bits=4), mats) if M > 1 else base
            cells.append(f"{t_ours / base:5.2f} (mlx {t_mlx / base:4.2f})")
        print(f"{name:<12} " + " ".join(f"{c:>14}" for c in cells) + f"   {err:.1e}")
        del mats


if __name__ == "__main__":
    main()
