#!/usr/bin/env python3
"""Measure achievable HBM bandwidth and turn it into the decode roofline.

Everything in the project's speed argument is anchored to one number: the read
bandwidth this card actually sustains, not the 1792 GB/s on the spec sheet.
Weight streaming at bs=1 is a pure read, so `read` below is the number to use.
"""
import time
import torch

SPEC_GBS = 1792.0          # RTX 5090 datasheet
TEXT_PARAMS = 26.896e9     # Qwen3.8-27B text path incl. lm_head, excl. vision+mtp


def timed(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | sm_{p.major}{p.minor} | {p.total_memory / 2**30:.1f} GiB | {p.multi_processor_count} SMs")
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}\n")

    n = 1 << 28  # 256M bf16 elements = 512 MB, far past any cache
    a = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    b = torch.empty_like(a)
    nbytes = a.numel() * a.element_size()

    read = nbytes / timed(a.sum) / 1e9
    copy = 2 * nbytes / timed(lambda: b.copy_(a)) / 1e9

    print(f"{'read-only (reduction)':<24} {read:7.0f} GB/s   {read / SPEC_GBS * 100:5.1f}% of spec")
    print(f"{'copy (read+write)':<24} {copy:7.0f} GB/s   {copy / SPEC_GBS * 100:5.1f}% of spec")

    print(f"\nDecode roofline for {TEXT_PARAMS / 1e9:.2f}B text params, "
          f"weights-only, empty KV:")
    print(f"  {'bpw':>4} {'weights':>9} {'ceiling':>11} {'85-90% of wall':>18}")
    for bpw in (4.0, 4.5, 5.0, 5.5):
        gb = TEXT_PARAMS * bpw / 8 / 1e9
        ceil = read / gb
        print(f"  {bpw:4.1f} {gb:8.2f} GB {ceil:8.1f} t/s {ceil * 0.85:8.0f}-{ceil * 0.90:.0f} t/s")


if __name__ == "__main__":
    main()
