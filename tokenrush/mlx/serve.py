"""load_session for tokenrush/serve.py on a Mac: the MLX engine and drafts behind
the same Session interface. serve.py picks this on macOS; the HTTP layer is shared."""
import time

import mlx.core as mx

from ..hub import is_packed, not_packed_message, resolve_model
from .dflash import DFlashDraft
from .model import Engine
from .session import Session
from .spec import MTPHead
from .weights import load_packed


def draft_vocab():
    """The drafts' lm_head rows: the first 128k ids, as the CUDA server uses (torch.arange(131072))."""
    return mx.arange(131072, dtype=mx.int32)


def load_session(args):
    from transformers import AutoTokenizer
    args.model = resolve_model(args.model, download=not args.no_download)
    if not is_packed(args.model):
        raise SystemExit(not_packed_message(args.model))
    want_dflash = args.draft in ("auto", "dflash")
    want_mtp = args.draft in ("auto", "mtp")
    if want_dflash:
        try:
            args.dflash_path = resolve_model(args.dflash_path, download=not args.no_download)
        except SystemExit as e:
            print(f"[warn] {e}; serving with the MTP draft")
            want_dflash, want_mtp = False, True
    cfg, w, mtp_t = load_packed(args.model, with_mtp=want_mtp)
    tok = AutoTokenizer.from_pretrained(args.model)
    engine = Engine(cfg, w, max_len=args.max_len)
    dv = draft_vocab()
    dflash = DFlashDraft(args.dflash_path, engine, draft_vocab=dv) if want_dflash else None
    mtp = MTPHead(engine, mtp_t, draft_vocab=dv) if want_mtp else None
    session = Session(engine, tok, cfg, dflash=dflash, mtp=mtp, chunk=args.chunk)
    t0 = time.perf_counter()
    for mode in [m for m, d in (("dflash", dflash), ("mtp", mtp)) if d is not None] or ["raw"]:
        for _ in session.generate(list(range(1000, 1040)), max_new=8, mode=mode):
            pass
    session.forget()
    print(f"MLX engine ready in {time.perf_counter() - t0:.1f}s warm-up; context {args.max_len}, drafts: "
          f"{'DFlash2 ' if dflash else ''}{'MTP' if mtp else ''}{'none' if not (dflash or mtp) else ''}", flush=True)
    return session, tok, cfg
