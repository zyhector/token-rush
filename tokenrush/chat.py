"""Conversation rendering and output parsing for the server.

Rendering goes through the checkpoint's own chat template (Qwen's), which
knows the tool-call format the model was trained on: a `# Tools` system
block, `<tool_call><function=name><parameter=k>v</parameter>…</function>
</tool_call>` in assistant turns, `<tool_response>` blocks in user turns.
This module only adapts the two client protocols to the template's message
shape (OpenAI's `tool_calls` / `tool` role and Anthropic's `tool_use` /
`tool_result` blocks), moves a system message the template would refuse
(anything not at position 0 — Claude Code appends one at the end) into a
user turn, and parses the model's output back: thinking, text, tool calls.
"""
import json
import re
import uuid
from dataclasses import dataclass, field

# ------------------------------------------------------------ message shapes


def _text_of(content) -> str:
    """The text of an OpenAI or Anthropic content field (string or block list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") in ("text", "input_text", "output_text") and "text" in b:
            parts.append(b["text"])
        elif isinstance(b, dict) and "text" in b and "type" not in b:
            parts.append(b["text"])
    return "".join(parts)


def _args_dict(args) -> dict:
    if args is None or args == "":
        return {}
    if isinstance(args, dict):
        return args
    try:
        v = json.loads(args)
        return v if isinstance(v, dict) else {"value": v}
    except (TypeError, ValueError):
        return {"value": str(args)}


def from_openai(messages: list) -> list:
    """OpenAI chat messages -> template messages."""
    out = []
    for m in messages:
        role = m.get("role")
        if role in ("user", "system", "developer"):
            out.append({"role": "system" if role == "developer" else role, "content": _text_of(m.get("content"))})
        elif role == "assistant":
            t = {"role": "assistant", "content": _text_of(m.get("content"))}
            rc = m.get("reasoning_content") or m.get("reasoning")
            if rc:
                t["reasoning_content"] = rc
            calls = []
            for c in m.get("tool_calls") or []:
                f = c.get("function", c)
                calls.append({"name": f.get("name", ""), "arguments": _args_dict(f.get("arguments"))})
            if calls:
                t["tool_calls"] = calls
            out.append(t)
        elif role == "tool":
            out.append({"role": "tool", "content": _text_of(m.get("content"))})
    return out


def from_anthropic(system, messages: list) -> list:
    """Anthropic Messages API (system + messages with content blocks) -> template messages."""
    out = []
    sys_text = _text_of(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            texts = []
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    if texts:
                        out.append({"role": "user", "content": "".join(texts)}); texts = []
                    body = _text_of(b.get("content"))
                    if b.get("is_error"):
                        body = "Error: " + body
                    out.append({"role": "tool", "content": body})
                elif b.get("type") == "text":
                    texts.append(b.get("text", ""))
            if texts:
                out.append({"role": "user", "content": "".join(texts)})
        elif role == "assistant":
            t = {"role": "assistant", "content": ""}
            if isinstance(content, str):
                t["content"] = content
            else:
                texts, calls, thinks = [], [], []
                for b in content or []:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text":
                        texts.append(b.get("text", ""))
                    elif bt == "tool_use":
                        calls.append({"name": b.get("name", ""), "arguments": _args_dict(b.get("input"))})
                    elif bt == "thinking":
                        thinks.append(b.get("thinking", ""))
                t["content"] = "".join(texts)
                if calls:
                    t["tool_calls"] = calls
                if thinks:
                    t["reasoning_content"] = "\n".join(thinks)
            out.append(t)
    return out


def tools_from_anthropic(tools: list) -> list:
    """Anthropic tool specs (name / description / input_schema) -> OpenAI function specs."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or "name" not in t:
            continue
        out.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                     "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
    return out


def tools_from_openai(tools: list) -> list:
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        f = t.get("function", t)
        if "name" not in f:
            continue
        out.append({"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                                     "parameters": f.get("parameters") or {"type": "object", "properties": {}}}})
    return out


def normalize(messages: list) -> list:
    """Template constraints: one system message, first, or none; a later system
    message (Claude Code appends one at the end of the array) becomes a user turn;
    empty user turns are dropped; the last message must not be an assistant turn
    that the model is meant to continue (we only support add_generation_prompt)."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "system" and out:
            if m["content"].strip():
                out.append({"role": "user", "content": m["content"]})
            continue
        if m["role"] == "user" and not m["content"].strip():
            continue
        out.append(m)
    if out and out[0]["role"] == "system" and not out[0]["content"].strip():
        out = out[1:]
    if not any(m["role"] == "user" for m in out):
        out.append({"role": "user", "content": "(continue)"})
    return out


def render(tok, messages: list, tools: list = None, think: bool = False, reasoning_effort: str = None) -> str:
    kw = {"enable_thinking": bool(think)}
    if think and reasoning_effort in ("low", "medium", "xhigh"):
        kw["reasoning_effort"] = reasoning_effort
    if tools:
        kw["tools"] = tools
    return tok.apply_chat_template(normalize(messages), tokenize=False, add_generation_prompt=True, **kw)


# ------------------------------------------------------------ output parsing

TOOL_OPEN, TOOL_CLOSE, THINK_CLOSE = "<tool_call>", "</tool_call>", "</think>"
_FUNC = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _convert(value: str, schema: dict):
    t = (schema or {}).get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    s = value
    try:
        if t == "integer":
            return int(s.strip())
        if t == "number":
            f = float(s.strip())
            return int(f) if f.is_integer() and "." not in s else f
        if t == "boolean":
            return s.strip().lower() in ("true", "1", "yes")
        if t in ("array", "object"):
            return json.loads(s)
        if t == "string" or t is None and not (s.lstrip().startswith(("{", "[")) or s.strip() in ("true", "false", "null")):
            return s
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def parse_tool_call(block: str, tool_schemas: dict) -> dict:
    """The inside of one <tool_call>…</tool_call> -> {"name", "arguments"}; unknown
    parameters are kept as strings."""
    m = _FUNC.search(block)
    if not m:
        return {"name": "", "arguments": {}}
    name, body = m.group(1).strip(), m.group(2)
    props = ((tool_schemas.get(name) or {}).get("properties") or {})
    args = {}
    for k, v in _PARAM.findall(body):
        args[k.strip()] = _convert(v, props.get(k.strip()))
    return {"name": name, "arguments": args}


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


class OutputParser:
    """Feed the model's text in pieces; get back ("thinking", s) / ("text", s) /
    ("tool_call", ToolCall) events. Holds back any tail that could be the start of
    a tag, so text is never emitted and then retracted."""

    def __init__(self, tool_schemas: dict = None, thinking: bool = False):
        self.schemas = tool_schemas or {}
        self.state = "think" if thinking else "text"
        self.buf = ""
        self.calls = []
        self._text_started = False

    @staticmethod
    def _held(buf: str, tags) -> int:
        """Length of the longest suffix of buf that is a proper prefix of a tag."""
        best = 0
        for tag in tags:
            for n in range(min(len(tag) - 1, len(buf)), 0, -1):
                if buf.endswith(tag[:n]):
                    best = max(best, n)
                    break
        return best

    def feed(self, piece: str):
        self.buf += piece
        out = []
        while True:
            if self.state == "think":
                i = self.buf.find(THINK_CLOSE)
                if i < 0:
                    h = self._held(self.buf, (THINK_CLOSE,))
                    if len(self.buf) - h:
                        out.append(("thinking", self.buf[:len(self.buf) - h])); self.buf = self.buf[len(self.buf) - h:]
                    return out
                if i:
                    out.append(("thinking", self.buf[:i]))
                self.buf = self.buf[i + len(THINK_CLOSE):].lstrip("\n")
                self.state = "text"
            elif self.state == "text":
                i = self.buf.find(TOOL_OPEN)
                if i < 0:
                    h = self._held(self.buf, (TOOL_OPEN,))
                    emit = self.buf[:len(self.buf) - h]
                    # whitespace right before a tool call belongs to the call, not the text
                    keep = len(emit) - len(emit.rstrip())
                    emit = emit[:len(emit) - keep]
                    if emit:
                        out.append(("text", emit)); self._text_started = True
                    self.buf = self.buf[len(emit):]
                    return out
                pre = self.buf[:i].rstrip()
                if pre:
                    out.append(("text", pre)); self._text_started = True
                self.buf = self.buf[i + len(TOOL_OPEN):]
                self.state = "tool"
            else:
                i = self.buf.find(TOOL_CLOSE)
                if i < 0:
                    return out
                call = parse_tool_call(self.buf[:i], self.schemas)
                tc = ToolCall(call["name"], call["arguments"])
                self.calls.append(tc)
                out.append(("tool_call", tc))
                self.buf = self.buf[i + len(TOOL_CLOSE):].lstrip("\n")
                self.state = "text"

    def finish(self):
        """End of output: flush what is held. An unclosed tool call is parsed anyway
        (the model may stop on <|im_end|> right after </function>)."""
        out = []
        if self.state == "tool":
            if "<function=" in self.buf:
                call = parse_tool_call(self.buf, self.schemas)
                tc = ToolCall(call["name"], call["arguments"])
                self.calls.append(tc)
                out.append(("tool_call", tc))
            elif self.buf.strip():
                out.append(("text", TOOL_OPEN + self.buf))
        elif self.buf:
            kind = "thinking" if self.state == "think" else "text"
            s = self.buf if kind == "thinking" else self.buf.rstrip()
            if s:
                out.append((kind, s))
        self.buf = ""
        return out


class StopFilter:
    """Cut the raw text at the first stop string, holding back a tail that could
    still become one. feed() -> (text to pass on, stopped: bool)."""

    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]
        self.buf = ""
        self.hit = None

    def feed(self, piece: str):
        if not self.stops:
            return piece, False
        self.buf += piece
        first = None
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0 and (first is None or i < first[0]):
                first = (i, s)
        if first is not None:
            self.hit = first[1]
            out, self.buf = self.buf[:first[0]], ""
            return out, True
        h = OutputParser._held(self.buf, self.stops)
        out, self.buf = self.buf[:len(self.buf) - h], self.buf[len(self.buf) - h:]
        return out, False

    def finish(self):
        out, self.buf = self.buf, ""
        return out
