"""JSONL streaming parser for Claude Code session files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_relay.detect.models import Entry, ParsedSession, UsageData


def _parse_entry(raw: dict[str, Any]) -> Entry:
    """Convert a raw JSON dict into an Entry."""
    entry_type = raw.get("type", "")
    msg = raw.get("message", {})

    # Extract model and usage from assistant messages
    model = ""
    stop_reason = ""
    usage = None
    request_id = raw.get("requestId", "")

    if isinstance(msg, dict):
        model = msg.get("model", "")
        stop_reason = msg.get("stop_reason", "") or ""

        usage_dict = msg.get("usage")
        if isinstance(usage_dict, dict):
            usage = UsageData.from_dict(usage_dict)

    return Entry(
        type=entry_type,
        uuid=raw.get("uuid", ""),
        parent_uuid=raw.get("parentUuid", "") or "",
        timestamp=raw.get("timestamp", ""),
        session_id=raw.get("sessionId", ""),
        subtype=raw.get("subtype", ""),
        version=raw.get("version", ""),
        request_id=request_id,
        is_compact_summary=bool(raw.get("isCompactSummary", False)),
        usage=usage,
        model=model,
        stop_reason=stop_reason,
        raw=raw,
    )


def parse_session(path: Path) -> ParsedSession:
    """Parse a session JSONL file, streaming line by line.

    Handles malformed lines gracefully (skips with error count).
    Detects null bytes in the file content.
    """
    entries: list[Entry] = []
    parse_errors = 0
    null_bytes_found = False

    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue

                if "\x00" in stripped or "\ufffd" in stripped:
                    null_bytes_found = True
                    # Clean null bytes for JSON parsing attempt
                    stripped = stripped.replace("\x00", "")

                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                if not isinstance(raw, dict):
                    parse_errors += 1
                    continue

                entries.append(_parse_entry(raw))
    except OSError:
        pass

    # Derive project path from file location
    project_path = path.parent.name
    session_id = path.stem

    return ParsedSession(
        path=str(path),
        session_id=session_id,
        project_path=project_path,
        entries=entries,
        file_size_bytes=file_size,
        parse_errors=parse_errors,
        null_bytes_found=null_bytes_found,
    )
