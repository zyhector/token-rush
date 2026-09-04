#!/usr/bin/env python3
"""Measure achievable HBM bandwidth and turn it into the decode roofline.

Everything in the project's speed argument is anchored to one number: the read
bandwidth this card actually sustains, not the 1792 GB/s on the spec sheet.
Weight streaming at bs=1 is a pure read, so `read` below is the number to use.
"""
import time
import torch
import triton
import triton.language as tl

SPEC_GBS = 1792.0          # RTX 5090 datasheet
TEXT_PARAMS = 26.896e9     # Qwen3.8-27B text path incl. lm_head, excl. vision+mtp


@triton.jit
def _stream_read(x_ptr, out_ptr, n, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    """Read BLOCK*UNROLL elements per program with UNROLL independent loads in
    flight, reduce them to one number per program. This is what a bandwidth-
    bound GEMV looks like to the memory system; torch's reduction is a
    little more conservative about memory-level parallelism."""
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * BLOCK * UNROLL
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for u in tl.static_range(UNROLL):
        off = base + u * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + off, mask=off < n, other=0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc))


def best_time(fn, reps=5, iters=20, warmup=30):
    """Fastest per-iteration time over `reps` timed blocks.

    Take the best, not the mean. This card idles at 810 MHz memory clock and
    ramps to 14001 MHz under load, so a block that overlaps the ramp reports a
    number that is low by up to 30%. A slow block is always clock contamination;
    none of them can be faster than the hardware. The long warmup exists to get
    the ramp over with before the first measurement.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters)
    return best


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | sm_{p.major}{p.minor} | {p.total_memory / 2**30:.1f} GiB | {p.multi_processor_count} SMs")
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}\n")

    n = 1 << 30  # 1G bf16 elements = 2 GB, >20x the 96 MB L2 so no re-read can hit cache
    a = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    b = torch.empty_like(a)
    nbytes = a.numel() * a.element_size()

    BLOCK, UNROLL = 1024, 8
    grid = (triton.cdiv(n, BLOCK * UNROLL),)
    partial = torch.empty(grid[0], device="cuda", dtype=torch.float32)

    def stream():
        _stream_read[grid](a, partial, n, BLOCK=BLOCK, UNROLL=UNROLL, num_warps=4)

    read_torch = nbytes / best_time(a.sum) / 1e9
    read_triton = nbytes / best_time(stream) / 1e9
    copy = 2 * nbytes / best_time(lambda: b.copy_(a)) / 1e9
    read = max(read_torch, read_triton)

    print(f"{'read-only (torch.sum)':<26} {read_torch:7.0f} GB/s   {read_torch / SPEC_GBS * 100:5.1f}% of spec")
    print(f"{'read-only (triton stream)':<26} {read_triton:7.0f} GB/s   {read_triton / SPEC_GBS * 100:5.1f}% of spec")
    print(f"{'copy (read+write)':<26} {copy:7.0f} GB/s   {copy / SPEC_GBS * 100:5.1f}% of spec")
    print(f"\n{'READ WALL (best of the above)':<26} {read:7.0f} GB/s")

    print(f"\nDecode roofline for {TEXT_PARAMS / 1e9:.2f}B text params, "
          f"weights-only, empty KV:")
    print(f"  {'bpw':>4} {'weights':>9} {'ceiling':>11} {'85-90% of wall':>18}")
    for bpw in (4.0, 4.5, 5.0, 5.5):
        gb = TEXT_PARAMS * bpw / 8 / 1e9
        ceil = read / gb
        print(f"  {bpw:4.1f} {gb:8.2f} GB {ceil:8.1f} t/s {ceil * 0.85:8.0f}-{ceil * 0.90:.0f} t/s")


if __name__ == "__main__":
    main()
