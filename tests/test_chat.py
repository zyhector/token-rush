"""Rendering (through the checkpoint's template) and output parsing for the
server. Rendering tests need the tokenizer of the published checkpoint and
are skipped when it is not in the Hub cache or on disk."""
import os
import sys

import pytest

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from tokenrush.chat import OutputParser, StopFilter, from_anthropic, from_openai, normalize, parse_tool_call, render, tools_from_anthropic  # noqa: E402

SCHEMAS = {"Bash": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}},
           "Edit": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                     "new": {"type": "string"}, "all": {"type": "boolean"}}},
           "Q": {"type": "object", "properties": {"ids": {"type": "array"}, "opts": {"type": "object"}, "x": {"type": "number"}}}}


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


# ------------------------------------------------------------ parsing

def test_parse_tool_call_types():
    block = ("\n<function=Edit>\n<parameter=path>\n/a/b.py\n</parameter>\n<parameter=old>\nx = 1\ny = 2\n</parameter>\n"
             "<parameter=new>\n\n</parameter>\n<parameter=all>\ntrue\n</parameter>\n</function>\n")
    c = parse_tool_call(block, SCHEMAS)
    assert c["name"] == "Edit"
    assert c["arguments"] == {"path": "/a/b.py", "old": "x = 1\ny = 2", "new": "", "all": True}
    c = parse_tool_call("<function=Q>\n<parameter=ids>\n[1, 2]\n</parameter>\n<parameter=opts>\n{\"a\": 1}\n</parameter>\n"
                        "<parameter=x>\n2.5\n</parameter>\n<parameter=extra>\nplain\n</parameter>\n</function>", SCHEMAS)
    assert c["arguments"] == {"ids": [1, 2], "opts": {"a": 1}, "x": 2.5, "extra": "plain"}
    c = parse_tool_call("<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>", SCHEMAS)
    assert c["arguments"] == {"command": "ls -la", "timeout": 30}


def _drain(p, pieces):
    ev = []
    for s in pieces:
        ev.extend(p.feed(s))
    ev.extend(p.finish())
    return ev


def test_parser_streams_text_then_tool_call_in_pieces():
    text = "I'll list the directory.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
    for n in (1, 3, 7, len(text)):
        p = OutputParser(SCHEMAS)
        ev = _drain(p, [text[i:i + n] for i in range(0, len(text), n)])
        texts = "".join(s for k, s in ev if k == "text")
        calls = [c for k, c in ev if k == "tool_call"]
        assert texts == "I'll list the directory.", (n, texts)
        assert len(calls) == 1 and calls[0].name == "Bash" and calls[0].arguments == {"command": "ls -la"}
        # a "<" that is not a tag is not held forever
    p = OutputParser(SCHEMAS)
    ev = _drain(p, ["a < b and c <t", "ool> d"])
    assert "".join(s for k, s in ev if k == "text") == "a < b and c <tool> d"


def test_parser_thinking_then_text_and_two_calls():
    text = "let me think\n</think>\n\nDone thinking. <tool_call>\n<function=Bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
    p = OutputParser(SCHEMAS, thinking=True)
    ev = _drain(p, [text[i:i + 5] for i in range(0, len(text), 5)])
    assert "".join(s for k, s in ev if k == "thinking") == "let me think\n"
    assert "".join(s for k, s in ev if k == "text") == "Done thinking."
    assert [c.arguments["command"] for k, c in ev if k == "tool_call"] == ["pwd", "ls"]


def test_parser_unclosed_call_at_eos():
    p = OutputParser(SCHEMAS)
    ev = _drain(p, ["<tool_call>\n<function=Bash>\n<parameter=command>\necho hi\n</parameter>\n</function>"])
    assert [c.arguments for k, c in ev if k == "tool_call"] == [{"command": "echo hi"}]


def test_stop_filter_holds_partial_and_cuts():
    f = StopFilter(["\nHuman:", "END"])
    out, hit = f.feed("hello\nHum")
    assert out == "hello" and not hit
    out, hit = f.feed("an: bye")
    assert out == "" and hit and f.hit == "\nHuman:"
    f = StopFilter(["END"])
    assert f.feed("abc EN")[0] == "abc "
    assert f.feed("x")[0] == "ENx"       # not a stop after all


# ------------------------------------------------------------ message shapes

def test_anthropic_blocks_to_template_messages():
    msgs = from_anthropic([{"type": "text", "text": "sys A"}, {"type": "text", "text": " sys B"}], [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "Let me look."},
                                          {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "a.py\n"}]},
                                     {"type": "text", "text": "now?"}]},
    ])
    assert msgs[0] == {"role": "system", "content": "sys A sys B"}
    assert msgs[2]["tool_calls"] == [{"name": "Bash", "arguments": {"command": "ls"}}]
    assert msgs[3] == {"role": "tool", "content": "a.py\n"}
    assert msgs[4] == {"role": "user", "content": "now?"}
    assert tools_from_anthropic([{"name": "Bash", "description": "run", "input_schema": SCHEMAS["Bash"]}])[0]["function"]["parameters"] == SCHEMAS["Bash"]


def test_normalize_moves_late_system_into_user_turn():
    out = normalize([{"role": "system", "content": "first"}, {"role": "user", "content": "q"},
                     {"role": "system", "content": "appended by the client"}])
    assert [m["role"] for m in out] == ["system", "user", "user"]
    assert out[-1]["content"] == "appended by the client"


def test_render_through_template():
    tok = _tok()
    msgs = from_openai([{"role": "system", "content": "You are terse."}, {"role": "user", "content": "List files."},
                        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": "{\"command\": \"ls\"}"}}]},
                        {"role": "tool", "content": "a.py"}, {"role": "system", "content": "late system"}])
    tools = [{"type": "function", "function": {"name": "Bash", "description": "run", "parameters": SCHEMAS["Bash"]}}]
    s = render(tok, msgs, tools, think=False)
    assert "# Tools" in s and '"name": "Bash"' in s
    assert "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>" in s
    assert "<tool_response>\na.py\n</tool_response>" in s
    assert s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<|im_start|>user\nlate system<|im_end|>" in s        # the late system message became a user turn
    assert s.count("<|im_start|>system") == 1 and "You are terse." in s   # tools block and system prompt share one block
    s2 = render(tok, msgs[:2], think=True, reasoning_effort="low")
    assert s2.endswith("<|im_start|>assistant\n<think>\n") and "Reasoning effort is set to low" in s2
