"""The HTTP layer without a GPU: a stub Session that "generates" a canned reply
token by token, the real tokenizer (skipped when it is not on disk), FastAPI's
test client. Checks the shapes of both protocols, streaming and not, tool
calls, stop sequences, usage and the Anthropic event sequence."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, __file__.rsplit("/", 2)[0])


def _tok():
    from tokenrush.weights import DEFAULT_REPO, resolve_model
    for p in ("/workspace/models/Qwen3.8-27B-int4g128-gptq-mse", DEFAULT_REPO):
        try:
            d = resolve_model(p, download=False)
        except SystemExit:
            continue
        if os.path.exists(os.path.join(d, "tokenizer.json")):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(d)
    pytest.skip("checkpoint tokenizer not available")


class StubSession:
    """Replies with `reply` (a string) for every prompt, 3 tokens per step, then
    <|im_end|>. Records the prompt it got."""

    def __init__(self, tok, reply):
        self.tok = tok
        self.reply = reply
        self.stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        self.max_len = 4096
        self.prompts = []
        self.stats = types.SimpleNamespace(as_dict=lambda: {"reused_tokens": 7})

    def pick_mode(self, text, mode):
        return "dflash"

    def forget(self):
        pass

    def generate(self, ids, max_new, mode, temperature, top_p, top_k, seed):
        self.prompts.append((list(ids), max_new, temperature, top_p))
        toks = self.tok.encode(self.reply, add_special_tokens=False) + [self.tok.convert_tokens_to_ids("<|im_end|>")]
        for i in range(0, len(toks), 3):
            yield toks[i:i + 3]


TOOLS_A = [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
TOOLS_O = [{"type": "function", "function": {"name": "Bash", "description": "run", "parameters": TOOLS_A[0]["input_schema"]}}]
REPLY_TOOL = "I'll check.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nwc -l docs/traps.md\n</parameter>\n</function>\n</tool_call>"


def _client(reply):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    sess = StubSession(tok, reply)
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[])
    cfg = types.SimpleNamespace(eos_ids=(tok.eos_token_id,))
    return TestClient(build_app(sess, tok, cfg, args)), sess


def _sse(text):
    return [json.loads(l[5:]) for l in text.splitlines() if l.startswith("data:") and "[DONE]" not in l]


def test_openai_chat_tool_call_and_usage():
    c, sess = _client(REPLY_TOOL)
    r = c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "how many lines?"}], "tools": TOOLS_O}).json()
    msg = r["choices"][0]["message"]
    assert msg["content"] == "I'll check."
    assert msg["tool_calls"][0]["function"] == {"name": "Bash", "arguments": json.dumps({"command": "wc -l docs/traps.md"})}
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    assert r["usage"]["prompt_tokens"] == len(sess.prompts[0][0]) and r["usage"]["completion_tokens"] > 0
    assert sess.prompts[0][2] == 0.7                       # the server default when the request does not say
    # tools reached the template
    assert "<tools>" in sess.tok.decode(sess.prompts[0][0])


def test_openai_chat_stream_shapes():
    c, _ = _client("Hello there.")
    ev = _sse(c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True,
                                                    "stream_options": {"include_usage": True}, "temperature": 0}).text)
    assert ev[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev) == "Hello there."
    assert ev[-1]["choices"][0]["finish_reason"] == "stop" and ev[-1]["usage"]["completion_tokens"] > 0
    assert all(e["object"] == "chat.completion.chunk" for e in ev)


def test_completions_stop_sequence():
    c, sess = _client("Paris.\nThe capital of Germany is Berlin.")
    r = c.post("/v1/completions", json={"model": "m", "prompt": "The capital of France is", "max_tokens": 50, "stop": ["\n"]}).json()
    assert r["choices"][0]["text"] == "Paris."
    assert r["choices"][0]["finish_reason"] == "stop"
    assert sess.prompts[0][0] == sess.tok.encode("The capital of France is", add_special_tokens=False)


def test_anthropic_messages_tool_use_and_round_trip():
    c, sess = _client(REPLY_TOOL)
    body = {"model": "claude-x", "max_tokens": 100, "system": [{"type": "text", "text": "sys"}], "tools": TOOLS_A,
            "messages": [{"role": "user", "content": "how many lines?"}]}
    r = c.post("/v1/messages", json=body).json()
    assert [b["type"] for b in r["content"]] == ["text", "tool_use"]
    assert r["content"][1]["name"] == "Bash" and r["content"][1]["input"] == {"command": "wc -l docs/traps.md"}
    assert r["content"][1]["id"].startswith("toolu_")
    assert r["stop_reason"] == "tool_use" and r["usage"]["cache_read_input_tokens"] == 7
    # the round trip renders tool_use / tool_result through the template
    body["messages"] += [{"role": "assistant", "content": r["content"]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": r["content"][1]["id"], "content": "145 docs/traps.md"}]}]
    c.post("/v1/messages", json=body)
    prompt = sess.tok.decode(sess.prompts[1][0])
    assert "<tool_call>\n<function=Bash>\n<parameter=command>\nwc -l docs/traps.md\n</parameter>\n</function>\n</tool_call>" in prompt
    assert "<tool_response>\n145 docs/traps.md\n</tool_response>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # count_tokens agrees with what the job saw
    assert c.post("/v1/messages/count_tokens", json=body).json()["input_tokens"] == len(sess.prompts[1][0])


def test_anthropic_stream_event_sequence():
    c, _ = _client(REPLY_TOOL)
    text = c.post("/v1/messages", json={"model": "claude-x", "max_tokens": 100, "tools": TOOLS_A, "stream": True,
                                        "messages": [{"role": "user", "content": "how many lines?"}]}).text
    ev = _sse(text)
    types_ = [e["type"] for e in ev]
    assert types_[0] == "message_start" and types_[-2:] == ["message_delta", "message_stop"]
    starts = [e for e in ev if e["type"] == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["text", "tool_use"]
    assert [s["index"] for s in starts] == [0, 1]
    deltas = [e for e in ev if e["type"] == "content_block_delta"]
    assert "".join(d["delta"]["text"] for d in deltas if d["delta"]["type"] == "text_delta") == "I'll check."
    js = "".join(d["delta"]["partial_json"] for d in deltas if d["delta"]["type"] == "input_json_delta")
    assert json.loads(js) == {"command": "wc -l docs/traps.md"}
    assert types_.count("content_block_stop") == 2
    md = next(e for e in ev if e["type"] == "message_delta")
    assert md["delta"]["stop_reason"] == "tool_use" and md["usage"]["output_tokens"] > 0
    assert "event: message_start" in text          # the SSE event: lines Anthropic clients dispatch on


def test_anthropic_rejects_without_max_tokens_and_lists_models():
    c, _ = _client("x")
    r = c.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and r.json()["type"] == "error"
    m = c.get("/v1/models").json()
    assert m["data"][0]["id"] == "token-rush" and m["data"][0]["object"] == "model"
    assert c.get("/health").json()["status"] == "ok"


def test_api_key_required_when_set():
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    args = types.SimpleNamespace(api_key="secret", think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[])
    c = TestClient(build_app(StubSession(tok, "ok"), tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"x-api-key": "secret"}).status_code == 200
    assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
