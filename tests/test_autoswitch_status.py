"""Machine-readable auto-switch state for other tools.

Covers autoswitch_status.json (written after every engine tick),
autoswitch_history.jsonl (every switch, auto and manual, trimmed),
onSwitchCommand on manual switches (exactly once per switch), pause/resume
(`cswap auto --pause/--resume` and the engine honouring it), and the
`autoswitch` block of `cswap policy list --json`.

Synthetic fixtures only (tmp home, fake example.com emails).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap.account_policy import set_account_policy
from claude_swap.autoswitch import (
    NoSwitchEvent,
    PollEvent,
    SwitchEvent,
    TickOutcome,
    escape_candidate,
)
from claude_swap.autoswitch_files import (
    HISTORY_MAX_LINES,
    append_history,
    history_path,
    read_history,
    read_status,
    status_path,
)
from claude_swap.exceptions import ConfigError
from claude_swap.settings import (
    load_settings,
    pause_active,
    set_setting,
    settings_path,
)
from tests.test_account_policy import _of, _u
from tests.test_autoswitch import EngineHarness

STATUS_KEYS = {
    "schemaVersion", "pid", "startedAt", "lastTickAt", "nextTickAt",
    "outcome", "trigger", "reason", "detail", "active", "paused", "settings",
}


@pytest.fixture
def h(temp_home: Path) -> EngineHarness:
    """1=a, 2=b (live), 3=c."""
    harness = EngineHarness(temp_home)
    for num, email in ((1, "a@example.com"), (2, "b@example.com"), (3, "c@example.com")):
        harness.seed(num, email)
    harness.make_live("b@example.com", 2)
    return harness


def _wait_lines(path: Path, n: int, timeout: float = 10.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            if len(lines) >= n:
                return lines
        time.sleep(0.05)
    return path.read_text().splitlines() if path.exists() else []


# ---------------------------------------------------------------------------
# autoswitch_status.json
# ---------------------------------------------------------------------------


class TestStatusFile:
    def test_shape_after_a_no_action_tick(self, h):
        assert h.tick_with_usage({"1": _u(10), "2": _u(50), "3": _u(0)}) \
            is TickOutcome.NO_ACTION
        status = read_status(h.switcher.backup_dir)
        assert set(status) == STATUS_KEYS
        assert status["schemaVersion"] == 1
        assert status["outcome"] == "no-action"
        assert status["reason"] == "below-threshold"
        assert status["detail"] == "5h 50% < 90%"
        assert status["trigger"] is None
        assert status["active"] == "2"
        assert status["paused"] is False
        assert status["lastTickAt"] == h.clock.now
        assert status["nextTickAt"] is None  # --once style tick
        assert status["settings"]["threshold"] == 90.0
        assert status["settings"]["onSwitchCommand"] is False
        assert "@" not in status_path(h.switcher.backup_dir).read_text()

    def test_switch_tick_records_trigger(self, h):
        assert h.tick_with_usage({"1": _u(10), "2": _u(95), "3": _u(50)}) \
            is TickOutcome.SWITCHED
        status = read_status(h.switcher.backup_dir)
        assert status["outcome"] == "switched"
        assert status["trigger"] == "proactive"
        assert status["reason"] == "switched"
        assert status["detail"] == "#2 -> #1"
        assert status["active"] == "1"

    def test_dry_run_writes_nothing(self, h):
        h.engine = h._make_engine(dry_run=True)
        h.tick_with_usage({"1": _u(10), "2": _u(95), "3": _u(50)})
        assert not status_path(h.switcher.backup_dir).exists()

    def test_run_loop_records_next_tick(self, h):
        def on_event(event):
            h.events.append(event)
            if isinstance(event, NoSwitchEvent):
                h.engine.stop()

        h.engine = h._make_engine()
        h.engine.on_event = on_event
        entries = {}
        from tests.test_autoswitch import _entry_for

        for num, value in {"1": _u(10), "2": _u(50), "3": _u(0)}.items():
            entries[num] = _entry_for(value, h.clock.now)
        with patch.object(h.switcher, "usage_entries_by_account", return_value=entries):
            assert h.engine.run_loop() == 0
        status = read_status(h.switcher.backup_dir)
        assert status["nextTickAt"] is not None
        assert status["nextTickAt"] > status["lastTickAt"]


# ---------------------------------------------------------------------------
# autoswitch_history.jsonl and the hook on every switch
# ---------------------------------------------------------------------------


class TestHistory:
    def test_auto_switch_appends_numbers_only(self, h):
        h.tick_with_usage({"1": _u(10), "2": _u(95), "3": _u(50)})
        [entry] = read_history(h.switcher.backup_dir)
        assert entry["from"] == "2" and entry["to"] == "1"
        assert entry["trigger"] == "proactive" and entry["auto"] is True
        assert "@" not in history_path(h.switcher.backup_dir).read_text()

    def test_manual_switch_appends_manual(self, h):
        h.switcher.switch_to("3", json_output=True)
        [entry] = read_history(h.switcher.backup_dir)
        assert (entry["from"], entry["to"]) == ("2", "3")
        assert entry["trigger"] == "manual" and entry["auto"] is False

    def test_no_entry_when_nothing_changed(self, h):
        h.switcher.switch_to("2", json_output=True)  # already active
        h.engine = h._make_engine(dry_run=True)
        h.tick_with_usage({"1": _u(10), "2": _u(95), "3": _u(50)})
        assert read_history(h.switcher.backup_dir) == []

    def test_trimmed_to_cap_newest_kept(self, tmp_path):
        lines = [
            json.dumps({"at": i, "from": "1", "to": "2", "trigger": "manual",
                        "auto": False})
            for i in range(HISTORY_MAX_LINES + 100)
        ]
        history_path(tmp_path).write_text("\n".join(lines) + "\n")
        append_history(tmp_path, from_number=2, to_number=3,
                       trigger="policy-limit", at=99999.0)
        kept = history_path(tmp_path).read_text().splitlines()
        assert len(kept) == HISTORY_MAX_LINES
        assert json.loads(kept[0])["at"] == 101  # oldest dropped
        newest = read_history(tmp_path, limit=2)
        assert newest[0] == {"at": 99999.0, "from": "2", "to": "3",
                             "trigger": "policy-limit", "auto": True}
        assert newest[1]["at"] == HISTORY_MAX_LINES + 99

    def test_read_skips_bad_lines(self, tmp_path):
        history_path(tmp_path).write_text('{"at": 1, "from": 1, "to": 2, '
                                          '"trigger": "manual", "auto": false}\n'
                                          "not json\n[1]\n")
        assert read_history(tmp_path) == [
            {"at": 1, "from": "1", "to": "2", "trigger": "manual", "auto": False}
        ]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
class TestHookOnEverySwitch:
    def test_manual_switch_runs_hook_with_manual_trigger(self, h, tmp_path):
        out = tmp_path / "hook.txt"
        set_setting(
            h.switcher.backup_dir, "autoswitch.onSwitchCommand",
            f'echo "$CSWAP_FROM>$CSWAP_TO $CSWAP_TRIGGER" >> {out}',
        )
        h.switcher.switch_to("1", json_output=True)
        assert _wait_lines(out, 1) == ["2>1 manual"]
        time.sleep(0.3)
        assert out.read_text().splitlines() == ["2>1 manual"]  # once

    def test_auto_switch_runs_hook_exactly_once(self, h, tmp_path):
        out = tmp_path / "hook.txt"
        h.settings = h.settings.__class__(
            on_switch_command=f'echo "$CSWAP_FROM>$CSWAP_TO $CSWAP_TRIGGER" >> {out}'
        )
        h.engine = h._make_engine()
        h.tick_with_usage({"1": _u(10), "2": _u(95), "3": _u(50)})
        assert _wait_lines(out, 1) == ["2>1 proactive"]
        time.sleep(0.5)
        assert out.read_text().splitlines() == ["2>1 proactive"]

    def test_engine_command_wins_over_settings_file(self, h, tmp_path):
        """The engine hands its own settings snapshot's command down, so
        the switcher does not also read (and run) the file's."""
        file_out, engine_out = tmp_path / "file.txt", tmp_path / "engine.txt"
        set_setting(h.switcher.backup_dir, "autoswitch.onSwitchCommand",
                    f"echo file >> {file_out}")
        h.settings = h.settings.__class__(
            on_switch_command=f"echo engine >> {engine_out}"
        )
        h.engine = h._make_engine()
        h.tick_with_usage({"1": _u(10), "2": _u(95), "3": _u(50)})
        assert _wait_lines(engine_out, 1) == ["engine"]
        time.sleep(0.3)
        assert not file_out.exists()

    def test_hook_start_failure_is_a_warning(self, h):
        with patch("claude_swap.autoswitch_files.subprocess.Popen",
                   side_effect=OSError("no shell")):
            set_setting(h.switcher.backup_dir, "autoswitch.onSwitchCommand", "x")
            result = h.switcher.switch_to("1", json_output=True)
        assert result["switched"] is True
        assert any("could not start" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Pause
# ---------------------------------------------------------------------------


class TestPauseSetting:
    def test_default_is_none(self, tmp_path):
        assert load_settings(tmp_path).paused_until is None

    def test_set_validates(self, tmp_path):
        assert set_setting(tmp_path, "autoswitch.pausedUntil", "forever") == "forever"
        assert load_settings(tmp_path).paused_until == "forever"
        assert set_setting(tmp_path, "autoswitch.pausedUntil", "1761100000") \
            == 1761100000.0
        for bad in ("soon", "-5", "nan", "inf"):
            with pytest.raises(ConfigError, match="epoch seconds or 'forever'"):
                set_setting(tmp_path, "autoswitch.pausedUntil", bad)

    def test_garbage_in_file_loads_as_not_paused(self, tmp_path):
        for raw in ("tomorrow", True, [1], -1):
            settings_path(tmp_path).write_text(
                json.dumps({"autoswitch": {"pausedUntil": raw}})
            )
            assert load_settings(tmp_path).paused_until is None

    def test_pause_active(self):
        assert pause_active("forever", 0)
        assert pause_active(200.0, 100.0)
        assert not pause_active(100.0, 200.0)  # expired
        assert not pause_active(None, 0)


class TestPauseCli:
    def _auto(self, argv, capsys):
        from claude_swap import cli

        with patch("os.geteuid", return_value=1000, create=True):
            cli._auto_command(argv)
        return capsys.readouterr().out

    def _root(self, temp_home) -> Path:
        from claude_swap.switcher import ClaudeAccountSwitcher

        return ClaudeAccountSwitcher().backup_dir

    def test_pause_duration_then_resume(self, temp_home, capsys):
        before = time.time()
        out = self._auto(["--pause", "2h"], capsys)
        assert out.startswith("Auto-switch paused until ") and out.count("\n") == 1
        until = load_settings(self._root(temp_home)).paused_until
        assert before + 7200 - 5 <= until <= time.time() + 7200 + 5
        assert self._auto(["--resume"], capsys).strip() == "Auto-switch resumed"
        assert load_settings(self._root(temp_home)).paused_until is None
        assert "was not paused" in self._auto(["--resume"], capsys)

    def test_pause_defaults_to_forever(self, temp_home, capsys):
        self._auto(["--pause"], capsys)
        assert load_settings(self._root(temp_home)).paused_until == "forever"

    def test_pause_does_not_start_an_engine(self, temp_home, capsys, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("engine started")

        monkeypatch.setattr("claude_swap.autoswitch.AutoSwitchEngine", boom)
        self._auto(["--pause", "30m"], capsys)

    @pytest.mark.parametrize("bad", ["soon", "0", "-2h", "1e14d"])
    def test_bad_duration_errors(self, temp_home, capsys, bad):
        with pytest.raises(SystemExit) as exc:
            self._auto(["--pause", bad], capsys)
        assert exc.value.code == 2

    def test_pause_and_resume_are_exclusive(self, temp_home, capsys):
        with pytest.raises(SystemExit):
            self._auto(["--pause", "--resume"], capsys)

    def test_duration_units(self):
        from claude_swap.cli import _parse_pause_duration

        assert _parse_pause_duration("30m") == 1800
        assert _parse_pause_duration("1d") == 86400
        assert _parse_pause_duration("45") == 2700  # bare number = minutes
        assert _parse_pause_duration("FOREVER") is None


class TestEnginePause:
    def test_paused_polls_but_never_switches(self, h):
        set_setting(h.switcher.backup_dir, "autoswitch.pausedUntil", "forever")
        assert h.tick_with_usage({"1": _u(10), "2": _u(100), "3": _u(0)}) \
            is TickOutcome.NO_ACTION
        assert _of(h, PollEvent)  # still polled
        assert not _of(h, SwitchEvent)
        [no] = _of(h, NoSwitchEvent)
        assert no.reason == "paused" and "--resume" in no.detail
        status = read_status(h.switcher.backup_dir)
        assert status["paused"] is True and status["reason"] == "paused"
        assert status["settings"]["pausedUntil"] == "forever"
        assert h.switcher.current_account_number() == "2"

    def test_resume_reaches_the_next_tick(self, h):
        root = h.switcher.backup_dir
        set_setting(root, "autoswitch.pausedUntil", str(h.clock.now + 600))
        h.tick_with_usage({"1": _u(10), "2": _u(95), "3": _u(0)})
        assert _of(h, NoSwitchEvent)[-1].reason == "paused"
        h.clock.advance(601)  # expired == not paused
        assert h.tick_with_usage({"1": _u(10), "2": _u(95), "3": _u(0)}) \
            is TickOutcome.SWITCHED
        assert read_status(root)["paused"] is False


# ---------------------------------------------------------------------------
# `cswap policy list --json`: new row fields and the autoswitch block
# ---------------------------------------------------------------------------


class TestEscapeCandidate:
    def test_reserve_only_when_no_normal_account(self):
        from claude_swap.account_policy import AccountPolicy, evaluate_account_policy

        statuses = {
            "1": evaluate_account_policy(AccountPolicy("a", reserve=True), _u(10), 90),
            "3": evaluate_account_policy(None, _u(95), 90),
        }
        target, tier, tiers = escape_candidate(
            ["1", "3"], statuses, {"1": 90.0, "3": 5.0}
        )
        assert target == "1" and tier == 0 and tiers == [["1"]]
        assert escape_candidate([], {}, {}) == (None, None, [])


class TestPolicyListAutoswitch:
    @pytest.fixture
    def fleet(self, temp_home):
        from claude_swap.usage_store import FetchRecord

        h = EngineHarness(temp_home)
        emails = {"1": "a@co.com", "2": "b@co.com", "3": "c@co.com", "4": "d@co.com"}
        for num, email in emails.items():
            h.seed(int(num), email)
        h.make_live("c@co.com", 3)
        switcher = h.switcher
        switcher._usage_store.clock = time.time
        root = switcher.backup_dir
        settings_path(root).write_text(
            json.dumps({"autoswitch": {"threshold": 98, "cooldownSeconds": 300}})
        )
        set_account_policy(root, "a@co.com", "", reserve=True)
        set_account_policy(root, "c@co.com", "", priority=1)
        switcher.set_account_disabled("2", True)
        switcher._usage_store.record(
            {
                "1": FetchRecord(usage={
                    "five_hour": {"pct": 26.0, "resets_at": "2030-01-01T00:00:00Z"},
                    "seven_day": {"pct": 10.0, "resets_at": "2030-01-05T00:00:00Z"},
                }),
                "3": FetchRecord(usage={"five_hour": {"pct": 47.0},
                                        "seven_day": {"pct": 22.0}}),
                "4": FetchRecord(error="http-400"),
            },
            {n: (e, "") for n, e in emails.items()},
        )
        with patch("os.geteuid", return_value=1000, create=True):
            yield h

    def _doc(self, capsys) -> dict:
        from claude_swap import cli

        capsys.readouterr()
        cli._policy_command(["list", "--json"])
        return json.loads(capsys.readouterr().out)

    def test_row_fields(self, fleet, capsys):
        rows = {r["number"]: r for r in self._doc(capsys)["accounts"]}
        one = rows["1"]
        assert one["resetsAt5h"] == 1893456000.0
        assert one["resetsAt7d"] == 1893801600.0
        assert 0 <= one["usageAgeSeconds"] < 60
        assert one["lastError"] is None and one["quarantined"] is None
        assert one["switchable"] is True
        assert rows["2"]["switchable"] is False  # disabled
        assert rows["4"]["lastError"] == "http-400"
        assert rows["4"]["usageAgeSeconds"] is None

    def test_block_shape_and_next_candidate(self, fleet, capsys):
        block = self._doc(capsys)["autoswitch"]
        assert set(block) == {
            "threshold", "weeklyThreshold", "intervalSeconds", "cooldownSeconds",
            "hysteresisPct", "strategy", "pausedUntil", "settingsMtime",
            "lastSwitch", "cooldownUntil", "nextCandidate", "excluded",
            "service", "history", "manualHoldUntil", "manualHoldSeconds",
        }
        assert block["threshold"] == 98.0 and block["pausedUntil"] is None
        assert isinstance(block["settingsMtime"], float)
        assert block["nextCandidate"] == {
            "number": "1", "reserve": True, "priority": 100,
            "scheduledNow": False, "headroomPct": 74.0,
            "why": "only eligible account; reserve: used only while no "
                   "normal account qualifies",
        }
        assert block["excluded"] == {
            "2": "disabled", "4": "usage unknown (http-400)",
        }
        assert block["service"] is None and block["history"] == []
        assert block["lastSwitch"] is None and block["cooldownUntil"] is None

    def test_quarantine_excludes_and_marks_row(self, fleet, capsys):
        fleet.engine._mutate_state(lambda s: s.setdefault("quarantine", {}).update(
            {"1": {"email": "a@co.com", "reason": "invalid_grant", "at": "x"}}
        ))
        doc = self._doc(capsys)
        row = next(r for r in doc["accounts"] if r["number"] == "1")
        assert row["quarantined"] == "invalid_grant"
        assert doc["autoswitch"]["excluded"]["1"] == "quarantined: invalid_grant"
        assert doc["autoswitch"]["nextCandidate"] is None
        assert "@" not in json.dumps(doc["autoswitch"])

    def test_service_stale_flag(self, fleet, capsys):
        root = fleet.switcher.backup_dir
        status = {"schemaVersion": 1, "lastTickAt": time.time(),
                  "settings": {"intervalSeconds": 60}}
        status_path(root).write_text(json.dumps(status))
        assert self._doc(capsys)["autoswitch"]["service"]["stale"] is False
        status["lastTickAt"] = time.time() - 181  # 3 x 60 < 180 floor
        status_path(root).write_text(json.dumps(status))
        assert self._doc(capsys)["autoswitch"]["service"]["stale"] is True
        status["settings"]["intervalSeconds"] = 120  # 3 x 120 = 360
        status_path(root).write_text(json.dumps(status))
        assert self._doc(capsys)["autoswitch"]["service"]["stale"] is False

    def test_history_last_switch_cooldown_pause(self, fleet, capsys):
        root = fleet.switcher.backup_dir
        now = time.time()
        append_history(root, from_number=2, to_number=3, trigger="policy-limit",
                       at=now - 100)
        append_history(root, from_number=3, to_number=1, trigger="manual", at=now - 10)
        fleet.engine._mutate_state(lambda s: s.update(
            lastSwitchAt=now - 100, lastSwitchFrom=2, lastSwitchTo="3",
            leftTrigger="policy-limit",
        ))
        set_setting(root, "autoswitch.pausedUntil", "forever")
        block = self._doc(capsys)["autoswitch"]
        assert [e["trigger"] for e in block["history"]] == ["manual", "policy-limit"]
        assert block["lastSwitch"]["trigger"] == "manual"
        assert block["lastSwitch"]["auto"] is False
        assert block["cooldownUntil"] == pytest.approx(now + 200)
        assert block["pausedUntil"] == "forever"
        assert "@" not in json.dumps(block)

    def test_manual_hold_only_for_a_manual_switch_onto_the_active(self, fleet, capsys):
        root = fleet.switcher.backup_dir
        now = time.time()
        append_history(root, from_number=1, to_number=3, trigger="manual", at=now - 10)
        block = self._doc(capsys)["autoswitch"]
        assert block["manualHoldSeconds"] == 7200.0
        assert block["manualHoldUntil"] == pytest.approx(now - 10 + 7200)
        append_history(root, from_number=3, to_number=1, trigger="manual", at=now - 5)
        assert self._doc(capsys)["autoswitch"]["manualHoldUntil"] is None  # #1 is not active
        append_history(root, from_number=1, to_number=3, trigger="reserve-return", at=now)
        assert self._doc(capsys)["autoswitch"]["manualHoldUntil"] is None  # automatic
        append_history(root, from_number=1, to_number=3, trigger="manual", at=now, hold=False)
        assert self._doc(capsys)["autoswitch"]["manualHoldUntil"] is None  # a tool switching back
        append_history(root, from_number=1, to_number=3, trigger="manual", at=now + 86400)
        assert self._doc(capsys)["autoswitch"]["manualHoldUntil"] is None  # a clock a day ahead
        set_setting(root, "autoswitch.manualHoldSeconds", 0)
        append_history(root, from_number=1, to_number=3, trigger="manual", at=now)
        assert self._doc(capsys)["autoswitch"]["manualHoldUntil"] is None  # off

    def test_state_only_last_switch_from_older_engine(self, fleet, capsys):
        now = time.time()
        fleet.engine._mutate_state(lambda s: s.update(
            lastSwitchAt=now - 1000, lastSwitchFrom=2, lastSwitchTo="3",
            leftTrigger="policy-limit",
        ))
        block = self._doc(capsys)["autoswitch"]
        assert block["lastSwitch"] == {
            "at": now - 1000, "from": "2", "to": "3",
            "trigger": "policy-limit", "auto": True,
        }
        assert block["cooldownUntil"] is None


# ---------------------------------------------------------------------------
# CLI surface: threshold note, top-level help
# ---------------------------------------------------------------------------


class TestCliSurface:
    def test_threshold_prints_policy_note_on_stderr(self, temp_home, capsys):
        from claude_swap import cli
        from claude_swap.switcher import ClaudeAccountSwitcher

        sw = ClaudeAccountSwitcher()
        sw._setup_directories()
        sw._init_sequence_file()
        with patch("os.geteuid", return_value=1000, create=True):
            cli._threshold_command([])
        captured = capsys.readouterr()
        assert "cswap policy set N --max-5h" in captured.err
        assert "Switch-away thresholds" in captured.out

    def test_top_level_help_lists_policy_and_pause(self, capsys):
        from claude_swap import cli

        with patch.object(sys, "argv", ["cswap", "--help"]), \
                pytest.raises(SystemExit):
            cli.main()
        out = capsys.readouterr().out
        assert "policy list" in out
        assert "auto --pause" in out and "auto --resume" in out
        assert "threshold <num|email>" not in out


class TestQuietRepeats:
    """The human `cswap auto` log leaves out a routine line that repeats."""

    def test_repeats_are_dropped_until_the_heartbeat(self):
        from claude_swap.cli import QuietRepeats

        now = [0.0]
        q = QuietRepeats(every=1800.0, clock=lambda: now[0])
        assert q.should_print("no-switch", "no switch: paused")
        now[0] = 60
        assert not q.should_print("no-switch", "no switch: paused")
        assert q.should_print("poll", "#1: 5h 76% used")  # another kind
        now[0] = 120
        assert q.should_print("no-switch", "no switch: below threshold")  # changed
        now[0] = 180
        assert q.should_print("no-switch", "no switch: paused")  # changed back
        now[0] = 180 + 1799
        assert not q.should_print("no-switch", "no switch: paused")
        now[0] = 180 + 1800
        assert q.should_print("no-switch", "no switch: paused")  # heartbeat

    def test_switches_and_errors_always_print(self):
        from claude_swap.cli import QuietRepeats

        q = QuietRepeats(clock=lambda: 0.0)
        # after an error the routine lines print again: they show it is over
        for kind, text in (("poll", "p"), ("no-switch", "n")):
            assert q.should_print(kind, text)
        assert q.should_print("error", "boom")
        assert q.should_print("poll", "p") and q.should_print("no-switch", "n")
        assert not q.should_print("poll", "p")
        for kind in ("switch", "error", "policy-limit"):
            assert q.should_print(kind, "same")
            assert q.should_print(kind, "same")
