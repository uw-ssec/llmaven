# data

Utilities for working with raw LiteLLM spend-log data exported via:

```sh
pixi shell -e llmaven
llmaven infra extract --from 2026-01-01 --to 2026-03-31 --out jan-feb-march-2026.zip -e .env
```

- **`reader.py`** — flattens raw request/response JSONL records into a tidy
  per-content-block DataFrame (`load_messages`), plus helpers for deduplicating
  resent history and picking the longest request per session.
- **`group_sessions.py`** — groups raw requests into full per-session
  conversations and writes one JSONL file, one JSON object per session
  (addresses [#46](https://github.com/uw-ssec/llmoxie/issues/46)).
- **`analysis.ipynb`** — exploratory analysis / plots on top of `reader.py`.

## Grouping requests into sessions

```sh
pixi run -e llmaven python data/group_sessions.py path/to/jan-feb-march-2026.zip -o sessions.jsonl
```

`path` accepts a single `.jsonl` file, a directory of daily
`litellm_spend_logs_*.jsonl` files, or a `.zip` of them (the format
`llmaven infra extract` produces).

Each line of the output is one session:

```json
{
  "session_id": "...",
  "device_id": "...",
  "account_uuid": "...",
  "user_api_key_alias": "carlos-api",
  "models": ["claude-sonnet-4-5-20250929"],
  "n_requests": 205,
  "total_spend": 17.07,
  "total_tokens": 19794027,
  "start_time": "2026-03-19T18:23:02.530000Z",
  "end_time": "2026-03-19T22:38:38.010000Z",
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [{"type": "tool_use", "name": "...", "id": "...", "input": {...}}]}
  ]
}
```

Notes:

- Each LiteLLM record re-sends the _entire_ conversation so far, so the
  session's messages are reconstructed from its single longest/most-recent
  request rather than merged across requests.
- `total_spend`/`total_tokens` are summed across every request in the session,
  so they reflect total resources consumed (including the resent-history
  overhead), not the length of the final conversation.
- Requests with no `session_id` in their `end_user` field (about a quarter of
  records in the Jan–Mar 2026 dump) can't be grouped and are skipped — the
  script prints how many.

On the Jan–Mar 2026 dump: 11,992 requests → 279 sessions in ~10s.

## How it works, function by function

### `reader.py` (Carlos's — mostly unchanged)

- **`load_messages(path)`** — reads one raw `.jsonl` file and unpacks it into a
  table (one row per piece of a message: one row of text, one tool call, etc.),
  tagged with which session/request/model it came from.
- **`last_request_per_session(df)`** — since every request re-sends the _whole_
  conversation so far, the request with the most messages in it is the one with
  the fullest picture. This picks that one request per session.
- **`deduplicate_messages(df)`** — because the history keeps getting resent, the
  same message shows up over and over across requests. This keeps only the first
  copy of each one. (Used by the analysis notebook, not by `group_sessions.py` —
  see the note below.)
- **`normalize_model_name(model)`** — model names show up in messy, inconsistent
  formats (`bedrock/us.anthropic.claude-3-5-sonnet-...`). This cleans them into
  one consistent short form.
- **`flatten_value`, `get_value`, `inspect_keys`** — small helpers for poking
  around in the raw JSON while exploring the data in a notebook.
- **`_parse_end_user`, `_base_row`, `_rows_from_block`** — internal plumbing
  `load_messages` uses to pull the session/device/account IDs out of each record
  and turn one message into table rows. Not meant to be called directly.

### `group_sessions.py` (new)

- **`main()`** — the entry point. Reads the command-line arguments, loads the
  data, builds the sessions, writes the output file, and prints a summary (how
  many requests loaded, how many sessions written, how many skipped).
- **`_resolve_input_paths(input)`** — figures out what to actually read: if you
  point it at a single file, a folder, or a `.zip`, it works out the list of
  `.jsonl` files inside.
- **`_load_all(paths)`** — loads every one of those files and stacks them into
  one big table.
- **`build_sessions(df)`** — the main logic. Groups all the rows by
  `session_id`, adds up each session's total spend/tokens/request count, and
  calls `_reconstruct_conversation` to build the actual message list. Also
  counts and skips any requests with no `session_id` at all.
- **`_reconstruct_conversation(rows)`** — takes one session's rows (already
  narrowed down to its fullest request) and puts the messages back in the right
  order: all the input messages first, then the final reply.
- **`_blocks_to_message(rows)`** — a message can be made of several pieces (some
  text, then a tool call). This glues those pieces back into one
  `{role, content}` message.
- **`_block_to_content(row)`** — converts one row back into the original-shaped
  piece it came from (a text block, a tool call, etc.).

**Why `deduplicate_messages` is skipped here:** it keeps only the _first_ copy
of each message, attributing it to whichever request sent it earliest. But we
want the full conversation from the request that has the _most_ messages — and
by the time dedup runs, most of that request's own messages have already been
"claimed" by earlier requests and removed. So `group_sessions.py` reconstructs
each session straight from its raw, un-deduped data instead.
