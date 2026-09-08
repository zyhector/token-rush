"""python -m tokenrush.quantize --src <hf bf16 dir> --dst <packed dir>"""
import argparse

from .weights import pack_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group", type=int, default=128)
    a = ap.parse_args()
    pack_checkpoint(a.src, a.dst, a.group)


if __name__ == "__main__":
    main()
