"""Per-account usage policy persisted at ``<backup_root>/account_limits.json``.

A personal layer on top of the auto-switch engine: each account may carry its
own 5h/7d utilization ceiling, a priority, and a reserve flag. Reaching a
personal ceiling is NOT an Anthropic rate limit — it only means the user chose
to stop spending that account for now. The block is always derived from the
current usage reading, never persisted, so it lifts by itself when the window
resets.

Accounts are keyed by the project's stable identity ``(email,
organizationUuid)`` — the same pair the usage store and directory mappings use
— so renaming an alias or renumbering a slot (``cswap swap`` / ``move``) keeps
the policy attached. Accounts without an entry fall back to the global
``autoswitch.threshold``; an absent or empty file leaves the engine's
behavior exactly as it was.

``enabled`` is deliberately not stored here: the project already has a
``disabled`` flag (``cswap disable`` / ``enable``) with exactly the wanted
semantics — held out of automatic selection, still a valid manual target — so
``cswap policy enable/disable`` reuses it rather than keeping a second copy.

An optional ``schedule`` (e.g. ``["09:00-14:00"]``, local time of the machine
running ``cswap auto``) restricts an account to those hours: outside them it
counts as blocked, inside them it is preferred over every other account until
it reaches a limit.

Reading is forgiving (bad file or fields → defaults plus a warning, never a
crash); writing is strict and atomic via ``settings.atomic_write_json``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from claude_swap import oauth
from claude_swap.exceptions import ConfigError
from claude_swap.settings import _read_raw_for_write, atomic_write_json

POLICY_FILENAME = "account_limits.json"
POLICY_SCHEMA_VERSION = 1

# Unconfigured priorities sort after any explicitly configured one (1, 2, 3 …)
# while every account still shares one group when nothing is configured.
DEFAULT_PRIORITY = 100
LIMIT_MIN = 1.0
LIMIT_MAX = 100.0
PRIORITY_MIN = 0
PRIORITY_MAX = 1000

_logger = logging.getLogger("claude-swap")

Identity = tuple[str, str]  # (email, organizationUuid)


def policy_path(backup_root: Path) -> Path:
    return backup_root / POLICY_FILENAME


def policy_key(email: str, organization_uuid: str) -> str:
    """JSON key for an account entry. Informational only: loading reads the
    entry's own ``email``/``organizationUuid`` fields."""
    return f"{email}|{organization_uuid}" if organization_uuid else email


@dataclass(frozen=True)
class AccountPolicy:
    """One account's personal limits. ``None`` means "use the fallback"."""

    email: str
    organization_uuid: str = ""
    max_5h: float | None = None
    max_7d: float | None = None
    priority: int | None = None
    reserve: bool = False
    # "HH:MM-HH:MM" local-time windows; empty = usable at any time.
    schedule: tuple[str, ...] = ()

    @property
    def identity(self) -> Identity:
        return (self.email, self.organization_uuid)


@dataclass(frozen=True)
class PolicyTable:
    """Loaded policies plus any warnings produced while reading them."""

    policies: dict[Identity, AccountPolicy] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.policies)

    def get(self, email: str, organization_uuid: str) -> AccountPolicy | None:
        return self.policies.get((email, organization_uuid or ""))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _lenient_limit(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    if not LIMIT_MIN <= value <= LIMIT_MAX:
        raise ValueError(
            f"{value!r} is outside {LIMIT_MIN:g}-{LIMIT_MAX:g}"
        )
    return float(value)


def _lenient_priority(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"expected an integer, got {value!r}")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"expected an integer, got {value!r}")
    if not PRIORITY_MIN <= value <= PRIORITY_MAX:
        raise ValueError(f"{value!r} is outside {PRIORITY_MIN}-{PRIORITY_MAX}")
    return value


_WINDOW_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


def _window_minutes(text: str) -> tuple[int, int]:
    """``"09:00-14:00"`` → minutes of day ``(540, 840)``. ``24:00`` is allowed
    as an end; a window whose end is before its start crosses midnight."""
    m = _WINDOW_RE.match(text) if isinstance(text, str) else None
    if not m:
        raise ValueError(f"{text!r} is not HH:MM-HH:MM")
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    if h1 > 23 or m1 > 59 or h2 > 24 or m2 > 59 or (h2 == 24 and m2):
        raise ValueError(f"{text!r} is not a valid time range")
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if start == end:
        raise ValueError(f"{text!r} is empty (start equals end)")
    return start, end


def _normalize_window(text: str) -> str:
    start, end = _window_minutes(text)
    return f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"


def _lenient_schedule(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"expected a list of HH:MM-HH:MM, got {value!r}")
    return tuple(_normalize_window(v) for v in value)


def in_schedule(schedule: Sequence[str], when: datetime) -> bool:
    """Whether ``when`` (local time) falls inside any window of ``schedule``.
    An empty schedule means "any time"."""
    if not schedule:
        return True
    minute = when.hour * 60 + when.minute
    for window in schedule:
        start, end = _window_minutes(window)
        if start < end:
            if start <= minute < end:
                return True
        elif minute >= start or minute < end:  # crosses midnight
            return True
    return False


def parse_schedule(raw: str) -> tuple[str, ...]:
    """Strictly parse a CLI schedule: comma-separated ``HH:MM-HH:MM``."""
    try:
        return tuple(
            _normalize_window(part) for part in raw.split(",") if part.strip()
        ) or _lenient_schedule([])
    except ValueError as e:
        raise ConfigError(f"--schedule: {e}") from None


def parse_limit(raw: str, name: str) -> float:
    """Strictly parse a CLI-provided limit percentage (1-100)."""
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} expects a number, got '{raw}'") from None
    if not LIMIT_MIN <= value <= LIMIT_MAX:
        raise ConfigError(
            f"{name} must be between {LIMIT_MIN:g} and {LIMIT_MAX:g}"
        )
    return value


def parse_priority(raw: str) -> int:
    """Strictly parse a CLI-provided priority (lower = preferred)."""
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"--priority expects an integer, got '{raw}'") from None
    if not PRIORITY_MIN <= value <= PRIORITY_MAX:
        raise ConfigError(
            f"--priority must be between {PRIORITY_MIN} and {PRIORITY_MAX}"
        )
    return value


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

# JSON key → (dataclass field, lenient validator)
_FIELDS = {
    "max5h": ("max_5h", _lenient_limit),
    "max7d": ("max_7d", _lenient_limit),
    "priority": ("priority", _lenient_priority),
}


def _parse_entry(key: str, raw: object) -> tuple[AccountPolicy | None, list[str]]:
    """One JSON entry → policy; every bad field degrades to its fallback."""
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return None, [f"entry {key!r} is not an object; ignored"]
    email = raw.get("email")
    org = raw.get("organizationUuid", "")
    if not isinstance(email, str) or not email:
        # Hand-written entries may carry the identity only in the key.
        email, _, org = key.partition("|")
    if not isinstance(org, str):
        org = ""
    if not email:
        return None, [f"entry {key!r} has no email; ignored"]
    kwargs: dict = {}
    for json_key, (attr, validate) in _FIELDS.items():
        if json_key not in raw or raw[json_key] is None:
            continue
        try:
            kwargs[attr] = validate(raw[json_key])
        except ValueError as e:
            warnings.append(f"{email}: {json_key} {e}; using the default")
    if "reserve" in raw:
        if isinstance(raw["reserve"], bool):
            kwargs["reserve"] = raw["reserve"]
        else:
            warnings.append(
                f"{email}: reserve expects true/false, got {raw['reserve']!r}; "
                "treating as false"
            )
    if raw.get("schedule") is not None:
        try:
            kwargs["schedule"] = _lenient_schedule(raw["schedule"])
        except ValueError as e:
            warnings.append(f"{email}: schedule {e}; usable at any time")
    if "enabled" in raw:
        warnings.append(
            f"{email}: 'enabled' is not read from {POLICY_FILENAME}; use "
            "'cswap policy disable/enable' (shared with 'cswap disable')"
        )
    return AccountPolicy(email=email, organization_uuid=org, **kwargs), warnings


def load_policies(backup_root: Path) -> PolicyTable:
    """Load every account policy. Never raises: a missing file is an empty
    table, a corrupt one is an empty table plus a warning."""
    path = policy_path(backup_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return PolicyTable()
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        msg = f"could not read {path} ({e}); per-account policy disabled"
        _logger.warning(msg)
        return PolicyTable(warnings=(msg,))
    if not isinstance(raw, dict) or not isinstance(raw.get("accounts", {}), dict):
        msg = f"{path} is malformed; per-account policy disabled"
        _logger.warning(msg)
        return PolicyTable(warnings=(msg,))
    policies: dict[Identity, AccountPolicy] = {}
    warnings: list[str] = []
    for key, entry in raw.get("accounts", {}).items():
        policy, entry_warnings = _parse_entry(str(key), entry)
        warnings.extend(entry_warnings)
        if policy is not None:
            policies[policy.identity] = policy
    for msg in warnings:
        _logger.warning("%s: %s", POLICY_FILENAME, msg)
    return PolicyTable(policies=policies, warnings=tuple(warnings))


_UNSET = object()


def set_account_policy(
    backup_root: Path,
    email: str,
    organization_uuid: str,
    *,
    max_5h: float | None | object = _UNSET,
    max_7d: float | None | object = _UNSET,
    priority: int | None | object = _UNSET,
    reserve: bool | object = _UNSET,
    schedule: tuple[str, ...] | None | object = _UNSET,
) -> AccountPolicy:
    """Create or update one account's entry; ``None`` clears a field back to
    its fallback, an omitted argument leaves it as is. Values must already be
    validated (``parse_limit`` / ``parse_priority``). A corrupt file raises
    ConfigError instead of being overwritten."""
    path = policy_path(backup_root)
    raw = _read_raw_for_write(path)
    accounts = raw.get("accounts")
    if not isinstance(accounts, dict):
        accounts = {}
    key = policy_key(email, organization_uuid)
    entry = accounts.get(key)
    if not isinstance(entry, dict):
        entry = {}
    entry["email"] = email
    entry["organizationUuid"] = organization_uuid
    for json_key, value in (
        ("max5h", max_5h), ("max7d", max_7d), ("priority", priority),
        ("reserve", reserve), ("schedule", schedule),
    ):
        if value is _UNSET:
            continue
        if value is None:
            entry.pop(json_key, None)
        else:
            entry[json_key] = list(value) if isinstance(value, tuple) else value
    accounts[key] = entry
    raw["version"] = POLICY_SCHEMA_VERSION
    raw["accounts"] = accounts
    atomic_write_json(path, raw)
    policy, _ = _parse_entry(key, entry)
    assert policy is not None
    return policy


def remove_account_policy(
    backup_root: Path, email: str, organization_uuid: str
) -> bool:
    """Delete one account's entry; False (and no write) if it had none."""
    path = policy_path(backup_root)
    raw = _read_raw_for_write(path)
    accounts = raw.get("accounts")
    if not isinstance(accounts, dict):
        return False
    identity = (email, organization_uuid or "")
    doomed = [
        key for key, entry in accounts.items()
        if (p := _parse_entry(str(key), entry)[0]) is not None
        and p.identity == identity
    ]
    if not doomed:
        return False
    for key in doomed:
        del accounts[key]
    raw["version"] = POLICY_SCHEMA_VERSION
    atomic_write_json(path, raw)
    return True


# ---------------------------------------------------------------------------
# Evaluation (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyWindow:
    """One usage window measured against its personal limit."""

    label: str  # "5h", "7d", or a scoped model name
    pct: float
    limit: float

    @property
    def blocked(self) -> bool:
        return self.pct >= self.limit


@dataclass(frozen=True)
class AccountPolicyStatus:
    """An account's policy verdict for one usage reading.

    ``configured`` is False for accounts without an entry: their limits are
    the global threshold, which the engine's own logic already enforces.
    ``windows`` is empty when usage is unknown — unknown is never blocked.
    """

    configured: bool
    reserve: bool
    priority: int
    max_5h: float
    max_7d: float
    windows: tuple[PolicyWindow, ...] = ()
    schedule: tuple[str, ...] = ()
    off_schedule: bool = False  # has a schedule and now is outside it

    @property
    def scheduled_now(self) -> bool:
        """Inside its own time window right now (preferred while usable)."""
        return bool(self.schedule) and not self.off_schedule

    @property
    def known(self) -> bool:
        return bool(self.windows)

    def pct(self, label: str) -> float | None:
        return next((w.pct for w in self.windows if w.label == label), None)

    @property
    def blocked_5h(self) -> bool:
        return any(w.blocked for w in self.windows if w.label == "5h")

    @property
    def blocked_7d(self) -> bool:
        return any(w.blocked for w in self.windows if w.label == "7d")

    @property
    def blocked(self) -> bool:
        return self.off_schedule or any(w.blocked for w in self.windows)

    def blocking_window(self) -> PolicyWindow | None:
        """The window furthest past its limit, or None when not blocked."""
        over = [w for w in self.windows if w.blocked]
        return max(over, key=lambda w: w.pct - w.limit) if over else None

    def reason(self) -> str:
        """e.g. ``personal 5h limit: 30.4% / 30%`` (empty when not blocked)."""
        w = self.blocking_window()
        if w is not None:
            return f"personal {w.label} limit: {w.pct:.10g}% / {w.limit:.10g}%"
        if self.off_schedule:
            return f"outside its schedule ({', '.join(self.schedule)})"
        return ""

    def summary(self) -> str:
        """e.g. ``5h 72% / 90%, 7d 61% / 90%``."""
        return ", ".join(
            f"{w.label} {w.pct:.10g}% / {w.limit:.10g}%" for w in self.windows
        )


def evaluate_account_policy(
    policy: AccountPolicy | None,
    usage: dict | str | None,
    threshold: float,
    models: Sequence[str] = (),
    now: float | None = None,
) -> AccountPolicyStatus:
    """Measure one usage reading against an account's personal limits.

    5h and 7d are checked independently against ``max_5h`` / ``max_7d`` (never
    folded into one binding number, so either window alone can block). Scoped
    per-model windows (``autoswitch.model``) keep the global threshold. Any
    limit the policy leaves unset falls back to ``threshold``. ``now`` (epoch
    seconds, read as local time) is checked against the policy's schedule;
    ``None`` skips the schedule check.
    """
    schedule = policy.schedule if policy else ()
    off_schedule = bool(schedule) and now is not None and not in_schedule(
        schedule, datetime.fromtimestamp(now)
    )
    max_5h = policy.max_5h if policy and policy.max_5h is not None else threshold
    max_7d = policy.max_7d if policy and policy.max_7d is not None else threshold
    limits = {"5h": max_5h, "7d": max_7d}
    windows = tuple(
        PolicyWindow(label, pct, limits.get(label, threshold))
        for label, pct, _ in oauth.relevant_windows(
            usage if isinstance(usage, dict) else None, models
        )
    )
    return AccountPolicyStatus(
        configured=policy is not None,
        reserve=bool(policy and policy.reserve),
        priority=(
            policy.priority
            if policy and policy.priority is not None
            else DEFAULT_PRIORITY
        ),
        max_5h=max_5h,
        max_7d=max_7d,
        windows=windows,
        schedule=schedule,
        off_schedule=off_schedule,
    )


def candidate_tiers(
    candidates: Sequence[str],
    statuses: Mapping[str, AccountPolicyStatus],
    *,
    include_reserve: bool = True,
    max_priority: int | None = None,
    only_scheduled: bool = False,
) -> list[list[str]]:
    """Split candidates into the ordered groups the engine ranks one by one.

    Policy-blocked (including off-schedule) accounts are dropped. Normal
    accounts inside their own time window come first, then the other normal
    accounts, one group per priority (ascending); reserve accounts follow the
    same way, so a
    reserve account is only reached when no normal group yields a target.
    Accounts with unknown usage stay in (the engine's ranking already skips
    unreadable rows). Input (sequence) order is kept inside a group, so the
    existing strategy decides between equals exactly as before.
    """
    groups: dict[tuple[bool, bool, int], list[str]] = {}
    for num in candidates:
        status = statuses.get(num)
        if status is None:
            continue
        if status.blocked:
            continue
        if status.reserve and not include_reserve:
            continue
        if only_scheduled and not (status.scheduled_now and not status.reserve):
            continue
        if (
            max_priority is not None
            and status.priority > max_priority
            and not status.scheduled_now
        ):
            continue
        key = (status.reserve, not status.scheduled_now, status.priority)
        groups.setdefault(key, []).append(num)
    return [groups[k] for k in sorted(groups)]


def all_policy_blocked(
    candidates: Sequence[str], statuses: Mapping[str, AccountPolicyStatus]
) -> bool:
    """True when there are candidates and every one is blocked (a usage
    limit needs a measurement; being off-schedule does not)."""
    return bool(candidates) and all(
        (s := statuses.get(n)) is not None and s.blocked for n in candidates
    )
