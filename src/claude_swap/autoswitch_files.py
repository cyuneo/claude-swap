"""Machine-readable auto-switch records beside settings.json.

Two small files in the backup root that other tools (the VS Code view,
cswap-remote) read instead of parsing the journal:

- ``autoswitch_status.json``: the running engine's last tick (written
  atomically after every tick; see ``AutoSwitchEngine``).
- ``autoswitch_history.jsonl``: one line per switch, automatic AND manual
  (``{"at", "from", "to", "trigger", "auto"}``), trimmed to the newest
  ``HISTORY_MAX_LINES``.

Both carry account numbers only — never an email. Also home to the
``autoswitch.onSwitchCommand`` launcher, which the switcher's switch path
runs after every real switch. Kept free of engine/switcher imports so both
can use it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from claude_swap.fsutil import atomic_write_text
from claude_swap.locking import FileLock

STATUS_FILENAME = "autoswitch_status.json"
STATUS_SCHEMA_VERSION = 1
HISTORY_FILENAME = "autoswitch_history.jsonl"
HISTORY_MAX_LINES = 500
MANUAL_TRIGGER = "manual"

_logger = logging.getLogger("claude-swap")

# Anything shaped like an address. Status details are built from account
# numbers already; this is the backstop for error text from elsewhere.
_EMAIL_RE = re.compile(r"[^\s()<>\[\],;:\"'/]+@[^\s()<>\[\],;:\"'/]+")


def redact_emails(text: str) -> str:
    return _EMAIL_RE.sub("[email]", text)


def status_path(root: Path) -> Path:
    return root / STATUS_FILENAME


def history_path(root: Path) -> Path:
    return root / HISTORY_FILENAME


def write_status(root: Path, payload: dict) -> None:
    """Atomically replace the status file (shared JSON writer, 0600)."""
    # Imported here: settings imports nothing from this module, but keeping
    # the writer shared means one place decides modes and symlink handling.
    from claude_swap.settings import atomic_write_json

    atomic_write_json(status_path(root), payload)


def read_status(root: Path) -> dict | None:
    try:
        raw = json.loads(status_path(root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _num(value) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def append_history(
    root: Path,
    *,
    from_number,
    to_number,
    trigger: str,
    at: float | None = None,
) -> dict:
    """Append one switch to the history file, keeping the newest
    ``HISTORY_MAX_LINES`` lines. Returns the entry written."""
    entry = {
        "at": time.time() if at is None else at,
        "from": _num(from_number),
        "to": _num(to_number),
        "trigger": trigger,
        "auto": trigger != MANUAL_TRIGGER,
    }
    path = history_path(root)
    with FileLock(root / ".autoswitch_history.lock"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        except (OSError, UnicodeDecodeError):
            lines = []
        lines = [ln for ln in lines if ln.strip()]
        lines.append(json.dumps(entry))
        lines = lines[-HISTORY_MAX_LINES:]
        atomic_write_text(path, "\n".join(lines) + "\n")
    return entry


def read_history(root: Path, limit: int = 20) -> list[dict]:
    """Newest-first switch history, at most ``limit`` entries. Bad lines are
    skipped."""
    try:
        lines = history_path(root).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    out: list[dict] = []
    for line in reversed(lines):
        if len(out) >= limit:
            break
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        out.append({
            "at": item.get("at"),
            "from": _num(item.get("from")),
            "to": _num(item.get("to")),
            "trigger": item.get("trigger"),
            "auto": bool(item.get("auto")),
        })
    return out


def run_on_switch_command(
    command: str | None,
    *,
    from_number,
    to_number,
    to_email: str,
    trigger: str,
) -> str | None:
    """Start ``autoswitch.onSwitchCommand`` detached; returns an error
    message when it could not start, else None.

    Fire-and-forget: the caller never waits on it (a cron ``--once`` tick
    returns immediately), and a failure to start it is a warning, never a
    failed switch.
    """
    if not command:
        return None
    env = {
        **os.environ,
        "CSWAP_FROM": "" if from_number is None else str(from_number),
        "CSWAP_TO": "" if to_number is None else str(to_number),
        "CSWAP_TO_EMAIL": to_email or "",
        "CSWAP_TRIGGER": trigger,
    }
    try:
        subprocess.Popen(
            command,
            shell=True,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        return f"autoswitch.onSwitchCommand could not start: {e}"
    return None
