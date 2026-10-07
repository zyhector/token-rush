"""The wall on a Mac: measured read bandwidth of the unified memory, and how close
MLX's own 4-bit matmul gets to it at the engine's real shapes.

The CUDA engine anchors every "% of roofline" on 1701 GB/s measured on the 5090
(docs/environment.md); this is the same measurement for Apple Silicon. Two numbers:

  read     a streaming reduction over a buffer far larger than the SLC, best of N
  q4 GEMV  mx.quantized_matmul (affine, bits 4, group 128: our packing byte for byte)
           at every distinct layer shape, M = 1 (decode) and M = 8 (a K = 7 verify),
           as packed GB/s, i.e. (codes + scales + biases) / time

    uv run python bench/mac/bandwidth.py
"""
import argparse
import json
import platform
import subprocess
import time

import mlx.core as mx

# (name, out, in) for Qwen3.8-27B, the fused matrices the engine actually runs
SHAPES = [
    ("gdn.in_qkvz", 10240 + 6144, 5120),
    ("gdn.out", 5120, 6144),
    ("attn.qkv", 24 * 256 * 2 + 2 * 4 * 256, 5120),
    ("attn.o", 5120, 24 * 256),
    ("mlp.gate_up", 2 * 17408, 5120),
    ("mlp.down", 5120, 17408),
    ("lm_head", 248320, 5120),
]
GROUP = 128


def chip() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()


def timeit(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def read_bandwidth(gb: float, iters: int) -> float:
    n = int(gb * 1e9) // 4
    x = mx.random.uniform(shape=(n,), dtype=mx.float32)
    mx.eval(x)
    # sum is bandwidth-bound; one fp32 scalar out
    t = min(timeit(lambda: mx.sum(x), iters) for _ in range(3))
    del x
    return n * 4 / t / 1e9


def q4_gemv(out: int, inp: int, M: int, iters: int, chain_gb: float = 1.5):
    """Packed bytes and seconds per matrix, run the way a decode step runs them: many
    distinct matrices of this shape in one graph, one eval, so neither the per-eval
    sync nor the SLC (repeated reads of one small matrix would hit it) flatters or
    penalizes the number."""
    w = mx.random.normal(shape=(out, inp), dtype=mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=4)
    del w
    nbytes = wq.nbytes + scales.nbytes + biases.nbytes
    n = max(1, int(chain_gb * 1e9 / nbytes))
    # distinct copies: a tiny add per copy forces separate buffers
    mats = [(wq + i % 2 * 0, scales + 0, biases + 0) if i else (wq, scales, biases) for i in range(n)]
    mats = [(mx.array(a), mx.array(b), mx.array(c)) for a, b, c in mats]
    x = mx.random.normal(shape=(M, inp), dtype=mx.bfloat16)
    mx.eval(mats, x)

    def chain():
        # independent matmuls, all reading x; summed so one output carries the graph
        ys = [mx.quantized_matmul(x, a, b, c, transpose=True, group_size=GROUP, bits=4) for a, b, c in mats]
        return mx.stack(ys).sum(0)
    t = min(timeit(chain, max(2, iters // 4)) for _ in range(3))
    return nbytes, t / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=4.0, help="read-test buffer size")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args()
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    print(f"{chip()}  mlx {mx.__version__}  memory {info.get('memory_size', 0) / 2**30:.0f} GiB")
    res = {"chip": chip(), "mlx": mx.__version__, "read_gbs": {}, "q4": []}
    for gb in (1.0, a.gb):
        bw = read_bandwidth(gb, a.iters)
        res["read_gbs"][str(gb)] = bw
        print(f"read {gb:4.1f} GB buffer: {bw:7.1f} GB/s")
    wall = max(res["read_gbs"].values())
    print(f"\nwall = {wall:.1f} GB/s\n")
    print(f"{'matrix':<12} {'out x in':>14} {'MB':>7} | {'M=1 GB/s':>9} {'%wall':>6} | {'M=8 GB/s':>9} {'x M=1':>6}")
    total_b, total_t = 0, 0.0
    for name, out, inp in SHAPES:
        nb, t1 = q4_gemv(out, inp, 1, a.iters)
        _, t8 = q4_gemv(out, inp, 8, a.iters)
        res["q4"].append({"name": name, "out": out, "in": inp, "bytes": nb, "t_m1": t1, "t_m8": t8})
        print(f"{name:<12} {out:>6} x {inp:<6} {nb / 1e6:7.1f} | {nb / t1 / 1e9:9.1f} {100 * nb / t1 / 1e9 / wall:5.1f}% "
              f"| {nb / t8 / 1e9:9.1f} {t8 / t1:6.2f}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
