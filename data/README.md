# data

Utilities for working with raw LiteLLM spend-log data exported via:

```sh
pixi shell -e llmaven
llmaven infra extract --from 2026-01-01 --to 2026-03-31 --out jan-feb-march-2026.zip -e .env
```

- **`reader.py`** — flattens raw request/response JSONL records into a tidy
  per-content-block DataFrame (`load_messages`), plus helpers for
  deduplicating resent history and picking the longest request per session.
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
- Each LiteLLM record re-sends the *entire* conversation so far, so the
  session's messages are reconstructed from its single longest/most-recent
  request rather than merged across requests.
- `total_spend`/`total_tokens` are summed across every request in the
  session, so they reflect total resources consumed (including the
  resent-history overhead), not the length of the final conversation.
- Requests with no `session_id` in their `end_user` field (about a quarter
  of records in the Jan–Mar 2026 dump) can't be grouped and are skipped —
  the script prints how many.

On the Jan–Mar 2026 dump: 11,992 requests → 279 sessions in ~10s.
