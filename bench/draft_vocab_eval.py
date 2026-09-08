#!/usr/bin/env python3
"""Effective tok/s of the speculative decoder for each draft vocabulary on six prompt
families (English essay / code / math, Chinese essay / math, mixed Chinese-English).

    python bench/draft_vocab_eval.py --model <packed> [--new 300]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_accept import PROMPTS  # noqa: E402
from tokenrush.model import Engine  # noqa: E402
from tokenrush.mtp import MTPHead, build_mtp  # noqa: E402
from tokenrush.spec import generate_spec_graph  # noqa: E402
from tokenrush.weights import load_packed  # noqa: E402

ZH = {
    "zh-essay": "请写一篇关于西罗马帝国衰亡原因与后果的短文，涵盖政治、经济和军事因素。",
    "zh-math": "一列火车以每小时60公里从A城出发，一小时后另一列以每小时90公里从450公里外的B城相向出发。它们何时何地相遇？请逐步求解。",
    "mixed": "用中文解释一下 Python 里 threading.Lock 和 asyncio.Lock 的区别，给一个带注释的代码例子，最后用一个简单的公式估算两种方案在 1000 个并发请求下的吞吐量差异。",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--new", type=int, default=300)
    ap.add_argument("--vocab-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenrush", "draft_vocab"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg, w, mtp_t = load_packed(a.model, with_mtp=True)
    prompts = dict(PROMPTS); prompts.update(ZH)
    stop = set(cfg.eos_ids) | {tok.eos_token_id}
    variants = {"full": None, "id_128k": torch.arange(131072)}
    for name in ("en_64k", "mix_64k", "mix_96k"):
        variants[name] = torch.load(os.path.join(a.vocab_dir, name + ".pt")).long()
    ids_by_prompt = {pn: tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                                            add_generation_prompt=True, enable_thinking=False))
                     for pn, t in prompts.items()}
    print("variant   | " + " | ".join(f"{pn:>9}" for pn in prompts) + "   (tok/s, accepted/step)")
    for vname, dv in variants.items():
        eng = Engine(cfg, w, max_len=4096, max_spec=4); eng.capture()
        mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=True), w.embed, w.lm_head, 4096)
        eng.attach_mtp(mtp, draft_vocab=dv)
        for k in (3, 4):
            eng.capture_spec(k)
        cells = []
        for pn, ids in ids_by_prompt.items():
            out, st = generate_spec_graph(eng, mtp, tok, ids, a.new, stop, stream=False)
            cells.append(f"{st['decode_tok_s']:4.0f} {st['accepted_per_step']:.2f}")
        print(f"{vname:9s} | " + " | ".join(f"{c:>9}" for c in cells))
        del eng, mtp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
