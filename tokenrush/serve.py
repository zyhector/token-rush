"""One-stream HTTP server: OpenAI Chat Completions / Completions and the
Anthropic Messages API over the resident engine.

    python -m tokenrush.serve [--port 8000] [--max-len 262144] [--api-key ...]

One request runs at a time (the engine is bs=1); others queue. Conversations
are re-sent whole by every client, so the Session keeps the context between
requests and prefills only the new turn (tokenrush/session.py). Tool calling
goes through the model's own format (tokenrush/chat.py); nothing here is
constrained decoding.

Claude Code:   ANTHROPIC_BASE_URL=http://host:8000 ANTHROPIC_AUTH_TOKEN=x claude
               (set CLAUDE_CODE_MAX_CONTEXT_TOKENS to --max-len; docs/serving.md)
OpenAI clients: base_url http://host:8000/v1, any model name.
"""
import argparse
import asyncio
import json
import os
import queue
import threading
import time
import uuid

import torch

from .chat import (OutputParser, StopFilter, from_anthropic, from_openai, render, tools_from_anthropic,
                   tools_from_openai)
from .session import Session

# ------------------------------------------------------------ the worker


class Job:
    def __init__(self, ids, mode, max_new, temperature, top_p, top_k, seed, stops, schemas, thinking, loop):
        self.ids, self.mode, self.max_new = ids, mode, max_new
        self.temperature, self.top_p, self.top_k, self.seed = temperature, top_p, top_k, seed
        self.stops, self.schemas, self.thinking = stops, schemas, thinking
        self.loop = loop
        self.q = asyncio.Queue()
        self.cancel = threading.Event()

    def push(self, ev):
        self.loop.call_soon_threadsafe(self.q.put_nowait, ev)


class Worker:
    """The one thread that touches the GPU. Events pushed per job:
    ("thinking", s) / ("text", s) / ("tool_call", ToolCall) / ("done", info) / ("error", msg)."""

    def __init__(self, session: Session, tok):
        self.session, self.tok = session, tok
        self.jobs = queue.Queue()
        self.last_stats = {}
        self.busy = False
        threading.Thread(target=self._loop, daemon=True, name="tokenrush-worker").start()

    def submit(self, job: Job):
        self.jobs.put(job)

    def _loop(self):
        while True:
            job = self.jobs.get()
            if job.cancel.is_set():
                continue
            self.busy = True
            try:
                self._run(job)
            except Exception as e:      # noqa: BLE001 — report to the client, keep serving
                import traceback
                traceback.print_exc()
                job.push(("error", f"{type(e).__name__}: {e}"))
                self.session.forget()
            finally:
                self.busy = False

    def _run(self, job: Job):
        sess, tok = self.session, self.tok
        parser = OutputParser(job.schemas, thinking=job.thinking)
        stopf = StopFilter(job.stops)
        out, printed = [], 0
        stop_reason, stop_seq = None, None
        gen = sess.generate(job.ids, max_new=job.max_new, mode=job.mode, temperature=job.temperature,
                            top_p=job.top_p, top_k=job.top_k, seed=job.seed)
        try:
            for toks in gen:
                if job.cancel.is_set():
                    stop_reason = "cancelled"
                    break
                out.extend(toks)
                if len(out) > job.max_new:
                    out = out[:job.max_new]
                ended = bool(out) and out[-1] in sess.stop_ids
                full = len(out) >= job.max_new
                text = tok.decode(out[printed:], skip_special_tokens=True)
                if text.endswith("�") and not (ended or full):
                    continue                                   # half a multibyte character: wait for the rest
                printed = len(out)
                piece, hit = stopf.feed(text)
                for ev in parser.feed(piece):
                    job.push(ev)
                if hit:
                    stop_reason, stop_seq = "stop_sequence", stopf.hit
                    break
                if ended:
                    stop_reason = "end_turn"
                    break
                if full:
                    stop_reason = "max_tokens"
                    break
            else:
                stop_reason = "max_tokens"                     # the context is full
        finally:
            gen.close()
        for ev in parser.feed(stopf.finish()) + parser.finish():
            job.push(ev)
        if stop_reason == "end_turn" and parser.calls:
            stop_reason = "tool_use"
        n_out = len(out) - (1 if out and out[-1] in sess.stop_ids else 0)
        st = sess.stats.as_dict()
        self.last_stats = st
        job.push(("done", {"stop_reason": stop_reason, "stop_sequence": stop_seq,
                           "input_tokens": len(job.ids), "output_tokens": n_out, "stats": st}))


# ------------------------------------------------------------ the app


def build_app(session: Session, tok, cfg, args):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="token-rush")
    worker = Worker(session, tok)
    served = args.served_name
    max_len = session.max_len
    reserve = 8 + 2                                            # a spec step may process K+1 tokens past the prompt

    def auth(req: Request):
        if not args.api_key:
            return
        key = req.headers.get("x-api-key") or req.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if key != args.api_key:
            raise HTTPException(401, "invalid api key")

    def want_thinking(body: dict, proto: str) -> bool:
        if args.think == "on":
            return True
        if args.think == "off":
            return False
        if proto == "anthropic":
            t = body.get("thinking")
            return isinstance(t, dict) and t.get("type") == "enabled"
        kw = body.get("chat_template_kwargs") or {}
        if "enable_thinking" in kw:
            return bool(kw["enable_thinking"])
        return body.get("reasoning_effort") is not None or isinstance(body.get("reasoning"), dict)

    def make_job(ids, body: dict, max_new, stops, schemas, thinking, text: str):
        if len(ids) + reserve >= max_len:
            raise HTTPException(400, f"prompt of {len(ids)} tokens does not fit the {max_len}-token context")
        max_new = max(1, min(int(max_new), max_len - len(ids) - reserve))
        temperature = body.get("temperature", args.temperature)
        top_p = body.get("top_p", args.top_p)
        top_k = body.get("top_k", 64)
        top_k = 64 if top_k is None or top_k <= 0 else min(int(top_k), 64)
        temperature = float(temperature) if temperature is not None else args.temperature
        top_p = float(top_p) if top_p is not None else args.top_p
        mode = session.pick_mode(text, args.draft)
        return Job(ids, mode, max_new, temperature, top_p, top_k, body.get("seed"), stops, schemas, thinking,
                   asyncio.get_running_loop())

    async def events(job: Job, req: Request):
        """Async iterator over the job's events; cancels the job when the client leaves."""
        worker.submit(job)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(job.q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if await req.is_disconnected():
                        job.cancel.set()
                        return
                    continue
                yield ev
                if ev[0] in ("done", "error"):
                    return
        finally:
            job.cancel.set()

    def sse(obj, event=None):
        head = f"event: {event}\n" if event else ""
        return f"{head}data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def encode(text: str):
        return tok.encode(text, add_special_tokens=False)

    def schemas_of(tools):
        return {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}

    # ---------------------------------------------------------------- models

    def model_entries():
        names = [served] + list(args.alias)
        return [{"id": n, "object": "model", "type": "model", "created": int(app.state.t0), "owned_by": "token-rush",
                 "display_name": n, "max_model_len": max_len} for n in names]

    @app.get("/v1/models")
    async def models(req: Request):
        auth(req)
        return {"object": "list", "data": model_entries(), "has_more": False}

    @app.get("/v1/models/{name:path}")
    async def model_one(name: str, req: Request):
        auth(req)
        return {**model_entries()[0], "id": name, "display_name": name}

    @app.get("/health")
    async def health():
        return {"status": "ok", "busy": worker.busy, "queued": worker.jobs.qsize(), "context": max_len}

    @app.get("/stats")
    async def stats():
        return worker.last_stats

    # ---------------------------------------------------------------- OpenAI

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        auth(req)
        body = await req.json()
        msgs = from_openai(body.get("messages") or [])
        tools = tools_from_openai(body.get("tools")) if body.get("tool_choice") != "none" else []
        thinking = want_thinking(body, "openai")
        text = render(tok, msgs, tools, think=thinking, reasoning_effort=body.get("reasoning_effort"))
        ids = encode(text)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        max_new = body.get("max_completion_tokens") or body.get("max_tokens") or args.max_new
        job = make_job(ids, body, max_new, stops, schemas_of(tools), thinking, text)
        rid, created, model = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or served
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(delta, finish=None, usage=None):
            c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                c["usage"] = usage
            return c

        def usage_of(d):
            return {"prompt_tokens": d["input_tokens"], "completion_tokens": d["output_tokens"],
                    "total_tokens": d["input_tokens"] + d["output_tokens"]}

        def finish_of(d):
            return {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls",
                    "cancelled": "stop"}[d["stop_reason"]]

        if body.get("stream"):
            async def gen():
                yield sse(chunk({"role": "assistant", "content": ""}))
                n_calls = 0
                async for kind, v in events(job, req):
                    if kind == "text":
                        yield sse(chunk({"content": v}))
                    elif kind == "thinking":
                        yield sse(chunk({"reasoning_content": v}))
                    elif kind == "tool_call":
                        yield sse(chunk({"tool_calls": [{"index": n_calls, "id": v.id, "type": "function",
                                                         "function": {"name": v.name, "arguments": json.dumps(v.arguments, ensure_ascii=False)}}]}))
                        n_calls += 1
                    elif kind == "error":
                        yield sse({"error": {"message": v, "type": "server_error"}})
                    else:
                        yield sse(chunk({}, finish_of(v), usage_of(v) if include_usage else None))
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

        text_out, think_out, calls, done, err = [], [], [], None, None
        async for kind, v in events(job, req):
            if kind == "text":
                text_out.append(v)
            elif kind == "thinking":
                think_out.append(v)
            elif kind == "tool_call":
                calls.append(v)
            elif kind == "error":
                err = v
            else:
                done = v
        if err:
            raise HTTPException(500, err)
        msg = {"role": "assistant", "content": "".join(text_out) or (None if calls else "")}
        if think_out:
            msg["reasoning_content"] = "".join(think_out)
        if calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                                 for c in calls]
        return {"id": rid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish_of(done)}], "usage": usage_of(done)}

    @app.post("/v1/completions")
    async def completions(req: Request):
        auth(req)
        body = await req.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], int):
                ids, prompt = list(prompt), ""
            else:
                prompt = "".join(prompt)
                ids = encode(prompt)
        else:
            ids = encode(prompt)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        job = make_job(ids, body, body.get("max_tokens") or args.max_new, stops, {}, False, prompt)
        job.thinking = False
        rid, created, model = "cmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or served
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(text, finish=None, usage=None):
            c = {"id": rid, "object": "text_completion", "created": created, "model": model,
                 "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}]}
            if usage is not None:
                c["usage"] = usage
            return c

        def usage_of(d):
            return {"prompt_tokens": d["input_tokens"], "completion_tokens": d["output_tokens"],
                    "total_tokens": d["input_tokens"] + d["output_tokens"]}

        finish_of = lambda d: "length" if d["stop_reason"] == "max_tokens" else "stop"
        if body.get("stream"):
            async def gen():
                async for kind, v in events(job, req):
                    if kind in ("text", "thinking"):
                        yield sse(chunk(v))
                    elif kind == "tool_call":
                        yield sse(chunk("<tool_call>" + json.dumps({"name": v.name, "arguments": v.arguments}) + "</tool_call>"))
                    elif kind == "error":
                        yield sse({"error": {"message": v, "type": "server_error"}})
                    else:
                        yield sse(chunk("", finish_of(v), usage_of(v) if include_usage else None))
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        parts, done, err = [], None, None
        async for kind, v in events(job, req):
            if kind in ("text", "thinking"):
                parts.append(v)
            elif kind == "tool_call":
                parts.append("<tool_call>" + json.dumps({"name": v.name, "arguments": v.arguments}) + "</tool_call>")
            elif kind == "error":
                err = v
            else:
                done = v
        if err:
            raise HTTPException(500, err)
        return {"id": rid, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": "".join(parts), "finish_reason": finish_of(done), "logprobs": None}],
                "usage": usage_of(done)}

    # ---------------------------------------------------------------- Anthropic

    def anthropic_error(status, etype, message):
        return JSONResponse({"type": "error", "error": {"type": etype, "message": message}}, status_code=status)

    def prepare_messages(body: dict):
        msgs = from_anthropic(body.get("system"), body.get("messages") or [])
        tc = body.get("tool_choice") or {}
        tools = [] if tc.get("type") == "none" else tools_from_anthropic(body.get("tools"))
        thinking = want_thinking(body, "anthropic")
        text = render(tok, msgs, tools, think=thinking)
        return text, encode(text), tools, thinking

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(req: Request):
        auth(req)
        body = await req.json()
        _, ids, _, _ = prepare_messages(body)
        return {"input_tokens": len(ids)}

    @app.post("/v1/messages")
    async def messages(req: Request):
        auth(req)
        body = await req.json()
        if "max_tokens" not in body:
            return anthropic_error(400, "invalid_request_error", "max_tokens is required")
        text, ids, tools, thinking = prepare_messages(body)
        try:
            job = make_job(ids, body, body["max_tokens"], body.get("stop_sequences") or [], schemas_of(tools), thinking, text)
        except HTTPException as e:
            return anthropic_error(e.status_code, "invalid_request_error", str(e.detail))
        mid, model = "msg_" + uuid.uuid4().hex[:24], body.get("model") or served

        def usage_of(d):
            return {"input_tokens": d["input_tokens"], "output_tokens": d["output_tokens"],
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": d["stats"].get("reused_tokens", 0)}

        stop_of = lambda d: "end_turn" if d["stop_reason"] == "cancelled" else d["stop_reason"]

        if body.get("stream"):
            async def gen():
                yield sse({"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": model,
                                                                 "content": [], "stop_reason": None, "stop_sequence": None,
                                                                 "usage": {"input_tokens": len(ids), "output_tokens": 0}}}, "message_start")
                idx, open_kind = -1, None          # the content block currently open

                def close():
                    return sse({"type": "content_block_stop", "index": idx}, "content_block_stop")

                async for kind, v in events(job, req):
                    if kind in ("text", "thinking"):
                        if open_kind != kind:
                            if open_kind is not None:
                                yield close()
                            idx += 1
                            open_kind = kind
                            block = {"type": "text", "text": ""} if kind == "text" else {"type": "thinking", "thinking": "", "signature": ""}
                            yield sse({"type": "content_block_start", "index": idx, "content_block": block}, "content_block_start")
                        delta = {"type": "text_delta", "text": v} if kind == "text" else {"type": "thinking_delta", "thinking": v}
                        yield sse({"type": "content_block_delta", "index": idx, "delta": delta}, "content_block_delta")
                    elif kind == "tool_call":
                        if open_kind is not None:
                            yield close()
                        idx += 1
                        open_kind = None
                        tid = "toolu_" + v.id.removeprefix("call_")
                        yield sse({"type": "content_block_start", "index": idx,
                                   "content_block": {"type": "tool_use", "id": tid, "name": v.name, "input": {}}}, "content_block_start")
                        yield sse({"type": "content_block_delta", "index": idx,
                                   "delta": {"type": "input_json_delta", "partial_json": json.dumps(v.arguments, ensure_ascii=False)}}, "content_block_delta")
                        yield close()
                    elif kind == "error":
                        yield sse({"type": "error", "error": {"type": "api_error", "message": v}}, "error")
                    else:
                        if open_kind is not None:
                            yield close()
                        yield sse({"type": "message_delta", "delta": {"stop_reason": stop_of(v), "stop_sequence": v["stop_sequence"]},
                                   "usage": usage_of(v)}, "message_delta")
                        yield sse({"type": "message_stop"}, "message_stop")
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

        content, done, err = [], None, None
        async for kind, v in events(job, req):
            if kind == "text":
                if content and content[-1]["type"] == "text":
                    content[-1]["text"] += v
                else:
                    content.append({"type": "text", "text": v})
            elif kind == "thinking":
                if content and content[-1]["type"] == "thinking":
                    content[-1]["thinking"] += v
                else:
                    content.append({"type": "thinking", "thinking": v, "signature": ""})
            elif kind == "tool_call":
                content.append({"type": "tool_use", "id": "toolu_" + v.id.removeprefix("call_"), "name": v.name, "input": v.arguments})
            elif kind == "error":
                err = v
            else:
                done = v
        if err:
            return anthropic_error(500, "api_error", err)
        return {"id": mid, "type": "message", "role": "assistant", "model": model, "content": content,
                "stop_reason": stop_of(done), "stop_sequence": done["stop_sequence"], "usage": usage_of(done)}

    app.state.t0 = time.time()
    return app


# ------------------------------------------------------------ startup


def load_session(args):
    """The same construction run.py does, kept resident."""
    from transformers import AutoTokenizer
    from .model import Engine
    from .quant import DEFAULT_BACKEND
    from .weights import is_packed, load_packed, not_packed_message, resolve_model
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
    cfg, w, mtp_t = load_packed(args.model, backend=args.backend or DEFAULT_BACKEND, with_mtp=want_mtp)
    tok = AutoTokenizer.from_pretrained(args.model)
    kv = torch.float8_e4m3fn if args.kv == "fp8" else torch.bfloat16
    kmin, kmax = 3, 4
    engine = Engine(cfg, w, max_len=args.max_len, kv_dtype=kv, max_spec=7 if want_dflash else (kmax if want_mtp else 0))
    t0 = time.perf_counter()
    engine.capture()
    dflash = mtp = None
    dv = torch.arange(131072)
    if want_dflash:
        from .dflash import DFlashDraft, load_dflash
        dflash = DFlashDraft(load_dflash(args.dflash_path, int4=True), w.embed, w.lm_head, cfg.hidden, args.max_len)
        engine.attach_dflash(dflash, draft_vocab=dv)
        engine.capture_spec_dflash()
    if want_mtp:
        from .mtp import MTPHead, build_mtp
        mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=True), w.embed, w.lm_head, args.max_len, kv_dtype=engine.state.kv_dtype)
        engine.attach_mtp(mtp, draft_vocab=dv)
        for k in range(kmin, kmax + 1):
            engine.capture_spec(k)
    print(f"captured the graphs in {time.perf_counter() - t0:.1f}s; weights {w.nbytes / 1e9:.2f} GB, state {engine.state.nbytes / 1e9:.2f} GB, "
          f"cuda allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB; context {args.max_len}, drafts: "
          f"{'DFlash2 ' if dflash else ''}{'MTP' if mtp else ''}{'none' if not (dflash or mtp) else ''}", flush=True)
    session = Session(engine, tok, cfg, dflash=dflash, mtp=mtp, chunk=args.chunk, kmin=kmin, kmax=kmax)
    # warm up every prefill shape class (Triton autotunes the first call of each) and the
    # spec loop, so the first request does not pay for it
    t0 = time.perf_counter()
    rows = engine.state.n_slots
    for mode in [m for m, d in (("dflash", dflash), ("mtp", mtp)) if d is not None] or ["raw"]:
        # one eager chunk, then every fused row count 1..rows (each M autotunes separately)
        for n in [args.chunk + 1] + [rows + m for m in range(1, rows + 1)]:
            g = session.generate(torch.randint(1000, 100000, (n,)).tolist(), max_new=4, mode=mode)
            for _ in g:
                pass
    session.forget()
    print(f"warmed up in {time.perf_counter() - t0:.1f}s", flush=True)
    return session, tok, cfg


def main():
    from .weights import DEFAULT_REPO, DFLASH_REPO
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_REPO, help="packed checkpoint: a Hub repo id or a local directory")
    ap.add_argument("--dflash-path", default=DFLASH_REPO)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--draft", default="auto", choices=("auto", "dflash", "mtp", "raw"),
                    help="auto (default): both drafts resident, the MTP chain for CJK prompts, DFlash2 otherwise")
    ap.add_argument("--backend", default=None, choices=("marlin", "triton", "tinygemm", "dequant"))
    ap.add_argument("--kv", default="fp8", choices=("bf16", "fp8"))
    ap.add_argument("--max-len", type=int, default=262144, help="the context window (256k needs ~30 GB with both drafts)")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--think", default="auto", choices=("auto", "on", "off"),
                    help="thinking: auto follows the request (Anthropic `thinking`, OpenAI `reasoning_effort` / chat_template_kwargs)")
    ap.add_argument("--temperature", type=float, default=0.7, help="when the request does not say")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=8192, help="max_tokens when the request does not say")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.environ.get("TOKENRUSH_API_KEY"), help="require it as x-api-key / Bearer")
    ap.add_argument("--served-name", default="token-rush")
    ap.add_argument("--alias", action="append", default=[], help="extra model names to list")
    args = ap.parse_args()
    session, tok, cfg = load_session(args)
    import uvicorn
    uvicorn.run(build_app(session, tok, cfg, args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
