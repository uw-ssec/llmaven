"""group_sessions.py — turn raw LiteLLM spend-log requests into per-session conversations.

Each raw record is a single request/response pair, and re-sends the full
conversation history seen so far. This script groups those requests by
``session_id`` and reconstructs, for each session, the single full ordered
conversation (using the last/most-complete request in that session) plus
session-level stats (request count, spend, tokens, time span).

Usage
-----
    python data/group_sessions.py path/to/jan-feb-march-2026.zip -o sessions.jsonl

``path`` may be a single ``.jsonl`` file, a directory of daily
``litellm_spend_logs_*.jsonl`` files, or a ``.zip`` archive of them (the
format produced by ``llmaven infra extract``).

Output is one JSON object per line (JSONL), one line per session:

    {
      "session_id": "...",
      "device_id": "...",
      "account_uuid": "...",
      "user_api_key_alias": "...",
      "models": ["claude-sonnet-4.6"],
      "n_requests": 4,
      "total_spend": 0.0192,
      "total_tokens": 1320,
      "start_time": "2026-03-28T02:46:57.632000Z",
      "end_time": "2026-03-28T02:51:10.221000Z",
      "messages": [
        {"role": "user", "content": [{"type": "text", "text": "..."}]},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "...", "id": "...", "input": {...}}]},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import jsonlines
import pandas as pd
from reader import last_request_per_session, load_messages


def _resolve_input_paths(input_path: Path) -> list[Path]:
    """Return a sorted list of .jsonl paths for a file, directory, or zip input."""
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))
    if input_path.suffix == ".zip":
        tmp_dir = Path(tempfile.mkdtemp(prefix="llmoxie_sessions_"))
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(tmp_dir)
        return sorted(tmp_dir.glob("*.jsonl"))
    return [input_path]


def _load_all(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate every non-empty .jsonl file into one DataFrame."""
    frames = [load_messages(p) for p in paths if p.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _block_to_content(row: pd.Series) -> dict:
    """Convert one flattened block row back into an Anthropic-style content block."""
    if row["type"] == "text":
        return {"type": "text", "text": row["text"]}
    if row["type"] == "thinking":
        return {"type": "thinking", "thinking": row["thinking"]}
    if row["type"] == "tool_use":
        try:
            tool_input = json.loads(row["tool_input"]) if row["tool_input"] else None
        except (json.JSONDecodeError, TypeError):
            tool_input = row["tool_input"]
        return {
            "type": "tool_use",
            "id": row["tool_use_id"],
            "name": row["tool_name"],
            "input": tool_input,
        }
    # Fallback for any other block type (e.g. tool_result), row["text"] holds
    # the JSON-serialised block from reader.py's fallback path.
    if row["text"] is not None:
        try:
            return json.loads(row["text"])
        except (json.JSONDecodeError, TypeError):
            pass
    return {"type": row["type"]}


def _blocks_to_message(block_rows: pd.DataFrame) -> dict:
    """Merge a group of same-message block rows into one {role, content} message."""
    role = block_rows["role"].iloc[0]
    content = [_block_to_content(row) for _, row in block_rows.iterrows()]
    return {"role": role, "content": content}


def _reconstruct_conversation(last_request_rows: pd.DataFrame) -> list[dict]:
    """Reconstruct one session's ordered message list from its last request.

    The last (most complete) request already resent the full input history,
    so its input blocks in msg_idx order are the conversation so far; its
    output blocks are the final assistant turn.
    """
    messages = []
    input_rows = last_request_rows[
        last_request_rows["direction"] == "input"
    ].sort_values(["msg_idx", "block_idx"])
    for _, block_rows in input_rows.groupby("msg_idx", sort=True):
        messages.append(_blocks_to_message(block_rows))

    output_rows = last_request_rows[
        last_request_rows["direction"] == "output"
    ].sort_values("block_idx")
    if not output_rows.empty:
        messages.append(_blocks_to_message(output_rows))

    return messages


def build_sessions(df: pd.DataFrame) -> tuple[list[dict], int]:
    """Group a raw messages DataFrame into per-session conversation records.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`reader.load_messages`, concatenated
        across all input files.

    Returns
    -------
    (sessions, n_requests_skipped)
        ``sessions`` is a list of per-session dicts ready to serialise to
        JSONL. ``n_requests_skipped`` is the count of requests with no
        ``session_id`` that could not be grouped.
    """
    raw_input_df = df[df["direction"] == "input"]
    n_requests_skipped = raw_input_df.loc[
        raw_input_df["session_id"] == "", "request_id"
    ].nunique()

    # Keep the raw (non-deduped) data: last_request_per_session needs each
    # request's own full resent history intact. deduplicate_messages would
    # strip most of that history away, attributing each message to whichever
    # request first introduced it rather than the request we're about to pick.
    df = df[df["session_id"] != ""]
    input_df = df[df["direction"] == "input"]

    per_request = input_df.drop_duplicates(subset=["session_id", "request_id"])
    stats = per_request.groupby("session_id").agg(
        device_id=("device_id", "first"),
        account_uuid=("account_uuid", "first"),
        user_api_key_alias=("user_api_key_alias", "first"),
        n_requests=("request_id", "nunique"),
        total_spend=("spend", "sum"),
        total_tokens=("total_tokens", "sum"),
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
    )
    models = per_request.groupby("session_id")["model"].apply(
        lambda s: sorted(set(s.dropna()))
    )

    last_df = last_request_per_session(df)

    sessions = []
    for session_id, group in last_df.groupby("session_id"):
        row = stats.loc[session_id]
        sessions.append(
            {
                "session_id": session_id,
                "device_id": row["device_id"],
                "account_uuid": row["account_uuid"],
                "user_api_key_alias": row["user_api_key_alias"],
                "models": models.loc[session_id],
                "n_requests": int(row["n_requests"]),
                "total_spend": float(row["total_spend"])
                if pd.notna(row["total_spend"])
                else None,
                "total_tokens": int(row["total_tokens"])
                if pd.notna(row["total_tokens"])
                else None,
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "messages": _reconstruct_conversation(group),
            }
        )

    sessions.sort(key=lambda s: s["start_time"] or "")
    return sessions, int(n_requests_skipped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Raw data: a .jsonl file, a directory of litellm_spend_logs_*.jsonl files, or a .zip of them",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sessions.jsonl"),
        help="Output JSONL path (default: sessions.jsonl)",
    )
    args = parser.parse_args()

    paths = _resolve_input_paths(args.input)
    if not paths:
        raise SystemExit(f"No .jsonl files found at {args.input}")

    df = _load_all(paths)
    n_requests = (
        df.loc[df["direction"] == "input", "request_id"].nunique()
        if not df.empty
        else 0
    )

    sessions, n_skipped = build_sessions(df)

    with jsonlines.open(args.output, mode="w") as writer:
        for record in sessions:
            writer.write(record)

    print(f"Loaded {n_requests} requests from {len(paths)} file(s)")
    print(f"Wrote {len(sessions)} sessions to {args.output}")
    if n_skipped:
        print(f"Skipped {n_skipped} requests with no session_id (could not be grouped)")


if __name__ == "__main__":
    main()
