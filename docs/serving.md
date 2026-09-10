# Serving: the engine as the model you open every day

`python -m tokenrush.serve` keeps the engine resident and speaks three
protocols over HTTP: OpenAI Chat Completions, OpenAI Completions, and the
Anthropic Messages API — the last one so **Claude Code** can run on it
directly. One stream, one request at a time (the engine is bs=1; others
queue), speculative decoding with both drafts resident, and the context kept
between requests so a chat turn costs the new tokens, not the conversation.

```bash
python -m tokenrush.serve --port 8000            # downloads the weights on first use, ~30 s to load
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"token-rush","messages":[{"role":"user","content":"Write a haiku about GPUs."}]}'
```

Files: `tokenrush/serve.py` (the app), `tokenrush/session.py` (the resident
engine, prefix reuse), `tokenrush/chat.py` (rendering and output parsing).
Tests: `tests/test_serve.py` (the protocols, no GPU), `tests/test_session.py`
(prefix reuse on random weights), `tests/test_chat.py`.

## Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000     # or the instance's public URL (see below)
export ANTHROPIC_AUTH_TOKEN=anything                 # or the --api-key you started the server with
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144         # = --max-len; Claude Code otherwise assumes 200k
claude
```

Every model name is accepted and served by the one engine, so the default,
`haiku` and `opus` slots all land here. What is implemented of the Messages
API: `system` (string or blocks), `messages` with `text` / `tool_use` /
`tool_result` blocks, `tools` with `input_schema`, `tool_choice` (`none`
honoured, `auto`/`any`/named treated as `auto`), `max_tokens`,
`stop_sequences`, `temperature` / `top_p` / `top_k`, `stream`, `thinking`
(`{"type":"enabled"}` turns Qwen's thinking on; the thinking comes back as a
`thinking` block), `POST /v1/messages/count_tokens`, `GET /v1/models`.
Streaming emits the standard event sequence (`message_start`,
`content_block_start/delta/stop`, `message_delta` with `stop_reason` and
usage, `message_stop`); tool calls stream as one `input_json_delta` per
call. `usage.cache_read_input_tokens` reports how many prompt tokens the
server reused from the previous request.

Verified end to end: Claude Code connected to the server reads files, edits
them and runs shell commands through the model's tool calls, with the tool
results flowing back as `tool_result` blocks (`docs/progress.md` step 33).

Two things the protocol adapter does that a raw template would not: Claude
Code appends a system message at the *end* of the messages array, which
Qwen's template refuses — it becomes a user turn (`chat.normalize`); and tool
results are matched to calls by order, since the template has no ids.

## OpenAI

`POST /v1/chat/completions`: `messages` (string or text-part content;
`system` / `developer` / `user` / `assistant` / `tool` roles, assistant
`tool_calls`, `reasoning_content`), `tools` / `tool_choice`, `max_tokens` /
`max_completion_tokens`, `stop`, `temperature` / `top_p` / `seed`,
`stream` (+ `stream_options.include_usage`), `reasoning_effort` or
`chat_template_kwargs.enable_thinking` for thinking (returned as
`reasoning_content`). Tool calls come back as `tool_calls` with JSON
`arguments`; `finish_reason` is `stop` / `length` / `tool_calls`.

`POST /v1/completions`: a raw `prompt` (string, or a list of token ids — that
is what `bench`-style clients send), `max_tokens`, `stop`, `stream`. No chat
template; what the model emits is returned as text.

`GET /v1/models` lists `--served-name` and any `--alias`; the model name in
a request is echoed back, never checked.

## How a request runs

1. The messages are rendered by the checkpoint's own chat template
   (`chat_template.jinja`), tools included: Qwen's format is a `# Tools`
   system block listing the functions as JSON and a
   `<tool_call><function=name><parameter=k>v</parameter>…</function></tool_call>`
   block in assistant turns. Nothing is constrained-decoded; the model
   writes that block and `chat.OutputParser` parses it back, converting
   parameter values by the tool's JSON-schema types (integers, numbers,
   booleans, arrays and objects; strings verbatim).
2. The prompt's token ids are compared with the token sequence the engine's
   state currently describes. The Session keeps two snapshots per request —
   the state at the end of the prompt and at the end of the generation
   (recurrent state 151 MB, conv ring 16 MB, the last hidden, the drafts'
   cursors; the KV rows and the drafts' caches stay in place) — and resumes
   from the longest one that is a prefix of the new prompt, prefilling only
   what follows. A tool-result turn therefore costs its own tokens: measured
   20 tokens after a 427-token prefix, 0.07 s end to end; 3 tokens after a
   120k prefix, 0.44 s. A prompt that shares no prefix (a new conversation)
   is prefilled from scratch.
3. Short prefills (a delta of up to 1024 tokens) run through the fused
   M-row kernels 8 rows at a time (1.4 ms/token); longer ones through the
   eager 4096-token chunks, whose dequantize-then-GEMM path costs ~1.8 s
   per call whatever the length (that fixed cost is what the 1500 tok/s
   prefill figure is made of, and why short deltas must avoid it).
4. Decoding is the speculative loop of `run.py`: the DFlash2 block draft
   for non-CJK prompts, the MTP chain for CJK ones (`--draft` to force one),
   sampling with the request's `temperature` / `top_p` (server defaults
   `--temperature 0.7 --top-p 0.9` when a request does not say; `0` is
   greedy). Tokens are decoded incrementally and streamed as they commit;
   `stop` strings are applied to the raw text before parsing.
5. One worker thread owns the GPU; requests queue in arrival order. A
   client that disconnects cancels its job at the next step; the state
   stays consistent (a snapshot is taken at whatever point it stopped).

Startup warms every prefill shape (the first call of each M-row count
autotunes for 1–4 s) so the first request does not pay for it.

## Limits, deliberately

- **One request at a time.** Concurrency is what the engine threw away.
- **No constrained decoding**: `response_format` / JSON schema are ignored;
  tool-call arguments are whatever the model wrote (parsed and typed, but
  not validated against the schema). Sampling runs inside the CUDA graph on
  a fixed top-64 candidate set, so a grammar would mean a kernel, not a
  server change.
- **`n > 1`, `logprobs`, `logit_bias`, images**: not supported; ignored.
- **The Responses API** (`/v1/responses`) is not implemented.
- The prompt-end snapshot is dropped when the answer runs past ~2000
  tokens (the DFlash draft's 4096-row ring window would have been
  overwritten); a follow-up then reuses the generation-end snapshot, or
  re-prefills if the client edited the history.
- Memory at `--max-len 262144` with both drafts is ~29 GB allocated, ~31.8
  GB reserved after warm-up; a 120k-token prompt served without incident.
  `--max-len 131072` leaves more headroom if anything else shares the card.

## Exposing it from a vast instance

Run it as a supervisor service on `127.0.0.1:8000` and put it behind the
Caddy auth edge on a free open port (the base-image guide in `CLAUDE.md`,
§7); Claude Code then needs `ANTHROPIC_BASE_URL=http://$PUBLIC_IPADDR:$VAST_TCP_PORT_<port>`
and the instance token as `ANTHROPIC_AUTH_TOKEN` (Caddy accepts it as a
Bearer token). Or forward the port over SSH (`ssh -L 8000:127.0.0.1:8000`)
and point Claude Code at `http://localhost:8000`, which needs no token and
exposes nothing. `--api-key` adds the server's own check on top (`x-api-key`
or `Authorization: Bearer`).
