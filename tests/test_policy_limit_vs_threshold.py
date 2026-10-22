"""A personal limit is only an entry's own max5h/max7d.

An account_limits.json entry that sets just priority/reserve/schedule keeps
its tiering, schedule and reserve behaviour but leaves through the normal
threshold path (proactive/at-limit, with cooldown and hysteresis). Also
covers the status vocabulary of `cswap policy list`, the email-free human
lines of `cswap auto`, and the running engine picking up settings.json.

Synthetic fixtures only (tmp home, fake emails).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap.account_policy import (
    AccountPolicy,
    evaluate_account_policy,
    set_account_policy,
)
from claude_swap.autoswitch import (
    NoSwitchEvent,
    PolicyExhaustedEvent,
    PolicyLimitEvent,
    PollEvent,
    ReserveInUseEvent,
    SettingsReloadedEvent,
    SwitchEvent,
    TickOutcome,
)
from claude_swap.settings import load_settings, settings_path
from tests.test_account_policy import _of, _u, _window
from tests.test_autoswitch import EngineHarness

# ---------------------------------------------------------------------------
# Pure evaluation: where each limit comes from
# ---------------------------------------------------------------------------


class TestLimitSources:
    def test_sources_and_has_own_limit(self):
        st = evaluate_account_policy(AccountPolicy("a@x", priority=1), _u(10), 90.0)
        assert (st.max_5h_source, st.max_7d_source) == ("threshold", "threshold")
        assert st.configured and not st.has_own_limit
        st = evaluate_account_policy(
            AccountPolicy("a@x", max_5h=30), _u(10), 90.0, weekly_threshold=97.0
        )
        assert (st.max_5h_source, st.max_7d_source) == ("own", "weekly")
        assert st.has_own_limit
        st = evaluate_account_policy(None, _u(10), 90.0)
        assert not st.has_own_limit and st.max_7d_source == "threshold"

    def test_reason_names_threshold_or_personal(self):
        over_line = evaluate_account_policy(None, _u(99), 98.0)
        assert over_line.reason() == "threshold 5h: 99% / 98%"
        own = evaluate_account_policy(AccountPolicy("a@x", max_5h=30), _u(31), 98.0)
        assert own.reason() == "personal 5h limit: 31% / 30%"
        # Own 5h limit, but the 7d window is over the plain switch line.
        mixed = evaluate_account_policy(
            AccountPolicy("a@x", max_5h=50), _u(10, 99), 98.0
        )
        assert mixed.reason() == "threshold 7d: 99% / 98%"


# ---------------------------------------------------------------------------
# Engine: priority/reserve/schedule-only entries use the threshold path
# ---------------------------------------------------------------------------


@pytest.fixture
def th(temp_home: Path) -> EngineHarness:
    """1=A (priority 1), 2=B live (priority 2), 3=C reserve; no own limits."""
    h = EngineHarness(temp_home)
    for num, email in ((1, "a@example.com"), (2, "b@example.com"), (3, "c@example.com")):
        h.seed(num, email)
    h.make_live("b@example.com", 2)
    root = h.switcher.backup_dir
    set_account_policy(root, "a@example.com", "", priority=1)
    set_account_policy(root, "b@example.com", "", priority=2)
    set_account_policy(root, "c@example.com", "", reserve=True)
    return h


class TestEntryWithoutOwnLimit:
    def test_below_threshold_is_a_plain_hold(self, th):
        assert th.tick_with_usage({"1": _u(10), "2": _u(50), "3": _u(0)}) \
            is TickOutcome.NO_ACTION
        [no] = _of(th, NoSwitchEvent)
        assert no.reason == "below-threshold" and no.detail == "5h 50% < 90%"
        [poll] = _of(th, PollEvent)
        assert poll.personal_limits == {}
        assert "switch at 90%" in poll.human()

    def test_over_threshold_departs_proactive_not_policy_limit(self, th):
        assert th.tick_with_usage({"1": _u(10), "2": _u(91), "3": _u(0)}) \
            is TickOutcome.SWITCHED
        [sw] = _of(th, SwitchEvent)
        assert sw.trigger == "proactive" and sw.to_ref["number"] == 1
        assert not _of(th, PolicyLimitEvent)

    def test_threshold_departure_honors_cooldown(self, th):
        th.engine._mutate_state(lambda s: s.update(lastSwitchAt=th.clock.now))
        assert th.tick_with_usage({"1": _u(10), "2": _u(91), "3": _u(0)}) \
            is not TickOutcome.SWITCHED
        assert _of(th, NoSwitchEvent)[-1].reason == "cooldown"

    def test_threshold_departure_honors_hysteresis(self, th):
        # A sits 5 points under its line, B is 1 over: a 6-point gain is
        # below the 10-point hysteresis, so a proactive move is refused.
        # (An own limit would leave anyway: test_personal_limit_beats_hysteresis.)
        th.tick_with_usage({"1": _u(85), "2": _u(91), "3": _u(85)})
        assert not _of(th, SwitchEvent) and not _of(th, PolicyLimitEvent)
        assert th.switcher.current_account_number() == "2"

    def test_tiering_still_applies(self, th):
        # C (reserve) has the most room, A (priority 1) still wins.
        th.tick_with_usage({"1": _u(40), "2": _u(91), "3": _u(0)})
        [sw] = _of(th, SwitchEvent)
        assert sw.to_ref["number"] == 1

    def test_reserve_only_entry_still_returns_to_a_normal_account(self, th):
        th.make_live("c@example.com", 3)
        assert th.tick_with_usage({"1": _u(10), "2": _u(10), "3": _u(5)}) \
            is TickOutcome.SWITCHED
        [sw] = _of(th, SwitchEvent)
        assert sw.trigger == "reserve-return"

    def test_reserve_hold_reason(self, th):
        th.make_live("c@example.com", 3)
        assert th.tick_with_usage({"1": _u(95), "2": _u(95), "3": _u(5)}) \
            is TickOutcome.NO_ACTION
        assert _of(th, NoSwitchEvent)[-1].reason == "reserve-hold"

    def test_schedule_only_entry_inside_window_uses_threshold(self, th):
        set_account_policy(
            th.switcher.backup_dir, "b@example.com", "",
            schedule=(_window(th.clock.now, -60, 60),),
        )
        th.tick_with_usage({"1": _u(10), "2": _u(91), "3": _u(0)})
        [sw] = _of(th, SwitchEvent)
        assert sw.trigger == "proactive"

    def test_schedule_only_entry_off_schedule_still_leaves(self, th):
        # Outside its own window the schedule itself says "stop using it":
        # that departure stays a personal-policy escape.
        set_account_policy(
            th.switcher.backup_dir, "b@example.com", "",
            schedule=(_window(th.clock.now, 60, 120),),
        )
        th.tick_with_usage({"1": _u(10), "2": _u(10), "3": _u(0)})
        [limit] = _of(th, PolicyLimitEvent)
        assert "outside its schedule" in limit.reason
        assert _of(th, SwitchEvent)[0].trigger == "policy-limit"


class TestEntryWithOwnLimit:
    def test_own_limit_equal_to_threshold_is_policy_limit(self, th):
        set_account_policy(th.switcher.backup_dir, "b@example.com", "", max_5h=90.0)
        th.engine._mutate_state(lambda s: s.update(lastSwitchAt=th.clock.now))
        # Own limit: no cooldown, no hysteresis.
        assert th.tick_with_usage({"1": _u(85), "2": _u(91), "3": _u(85)}) \
            is TickOutcome.SWITCHED
        [sw] = _of(th, SwitchEvent)
        assert sw.trigger == "policy-limit"
        [limit] = _of(th, PolicyLimitEvent)
        assert limit.reason == "personal 5h limit: 91% / 90%"

    def test_below_own_limit_reason(self, th):
        set_account_policy(
            th.switcher.backup_dir, "b@example.com", "", max_5h=60.0, max_7d=70.0
        )
        th.tick_with_usage({"1": _u(10), "2": _u(50, 20), "3": _u(0)})
        [no] = _of(th, NoSwitchEvent)
        assert no.reason == "below-personal-limit"
        assert no.detail == "personal policy: 5h 50% / 60%, 7d 20% / 70%"
        [poll] = _of(th, PollEvent)
        assert poll.personal_limits == {"5h": 60.0, "7d": 70.0}


# ---------------------------------------------------------------------------
# Human lines carry no emails; policy-exhausted JSON keyed by number
# ---------------------------------------------------------------------------


class TestNoEmailsInHumanLines:
    def test_poll_switch_and_policy_lines(self, th):
        set_account_policy(th.switcher.backup_dir, "b@example.com", "", max_5h=30.0)
        th.switcher.set_alias("2", "bee")
        th.tick_with_usage({"1": _u(10), "2": _u(31), "3": _u(0)})
        lines = [e.human() for e in th.events]
        assert lines and all("@" not in line for line in lines)
        [poll] = _of(th, PollEvent)
        assert poll.human().startswith("#2 (bee): ")
        [sw] = _of(th, SwitchEvent)
        assert sw.human() == "Switched #2 -> #1 (policy-limit)"
        # JSON keeps its contract (refs still carry the email).
        assert sw.to_json()["to"]["email"] == "a@example.com"
        [limit] = _of(th, PolicyLimitEvent)
        assert limit.name == "#2 (bee)" and "@" not in json.dumps(limit.to_json())

    def test_email_shaped_alias_is_not_shown(self, th):
        # `cswap alias` refuses "@", but a hand-edited sequence.json may not.
        data = th.switcher._get_sequence_data()
        data["accounts"]["1"]["alias"] = "someone@example.com"
        th.switcher._write_json(th.switcher.sequence_file, data)
        th.tick_with_usage({"1": _u(10), "2": _u(91), "3": _u(0)})
        [sw] = _of(th, SwitchEvent)
        assert sw.human() == "Switched #2 -> #1 (proactive)"

    def test_reserve_in_use_line(self, th):
        th.tick_with_usage({"1": _u(95), "2": _u(91), "3": _u(0)})
        [notice] = _of(th, ReserveInUseEvent)
        assert notice.human().startswith("Using reserve account #3 ")
        assert "@" not in json.dumps(notice.to_json())

    def test_policy_exhausted_keyed_by_number(self, th):
        root = th.switcher.backup_dir
        set_account_policy(root, "a@example.com", "", max_5h=20.0)
        set_account_policy(root, "b@example.com", "", max_5h=20.0)
        set_account_policy(root, "c@example.com", "", max_5h=20.0, reserve=True)
        th.switcher.set_alias("3", "spare")
        assert th.tick_with_usage({"1": _u(25), "2": _u(25), "3": _u(25)}) \
            is TickOutcome.BLOCKED
        [report] = _of(th, PolicyExhaustedEvent)
        payload = report.to_json()
        assert set(payload["accounts"]) == {"1", "2", "3"}
        assert payload["accounts"]["3"].startswith("reserve, policy limit reached")
        assert payload["names"]["3"] == "#3 (spare)"
        assert "@" not in json.dumps(payload) and "@" not in report.human()
        assert "    #3 (spare): reserve, policy limit reached" in report.human()

    def test_policy_exhausted_names_threshold_accounts_honestly(self, th):
        root = th.switcher.backup_dir
        set_account_policy(root, "b@example.com", "", max_5h=20.0)
        th.tick_with_usage({"1": _u(95), "2": _u(25), "3": _u(95)})
        [report] = _of(th, PolicyExhaustedEvent)
        assert report.accounts["1"] == "over threshold (threshold 5h: 95% / 90%)"
        assert report.accounts["2"].startswith("policy limit reached (personal 5h")


# ---------------------------------------------------------------------------
# The running engine reloads settings.json when it changes
# ---------------------------------------------------------------------------


def _write_settings(h: EngineHarness, section: dict, mtime: float) -> None:
    path = settings_path(h.switcher.backup_dir)
    path.write_text(json.dumps({"autoswitch": section}))
    os.utime(path, (mtime, mtime))


class TestSettingsReload:
    def _engine(self, h: EngineHarness):
        root = h.switcher.backup_dir
        h.engine = h._make_engine(settings_loader=lambda: load_settings(root))

    def test_threshold_change_applies_at_next_tick(self, th):
        _write_settings(th, {"threshold": 90}, 1_000_000)
        self._engine(th)
        usage = {"1": _u(10), "2": _u(85), "3": _u(0)}
        assert th.tick_with_usage(usage) is TickOutcome.NO_ACTION
        assert not _of(th, SettingsReloadedEvent)

        _write_settings(th, {"threshold": 80}, 1_000_100)
        th.events.clear()
        assert th.tick_with_usage(usage) is TickOutcome.SWITCHED
        [reloaded] = _of(th, SettingsReloadedEvent)
        assert reloaded.human() == "settings reloaded (threshold 80%, every 60s)"
        assert th.engine.settings.threshold == 80.0
        # Reported before the tick's own poll: the tick ran on the new values.
        assert th.events.index(reloaded) == 0

        th.make_live("b@example.com", 2)
        th.events.clear()
        th.clock.advance(3600)
        th.tick_with_usage(usage)
        assert not _of(th, SettingsReloadedEvent)  # mtime unchanged

    def test_touch_without_change_is_silent(self, th):
        _write_settings(th, {"threshold": 90}, 1_000_000)
        self._engine(th)
        _write_settings(th, {"threshold": 90}, 1_000_100)
        th.tick_with_usage({"1": _u(10), "2": _u(50), "3": _u(0)})
        assert not _of(th, SettingsReloadedEvent)

    def test_without_loader_settings_stay_fixed(self, th):
        _write_settings(th, {"threshold": 50}, 1_000_100)
        th.tick_with_usage({"1": _u(10), "2": _u(60), "3": _u(0)})
        assert th.engine.settings.threshold == 90.0
        assert not _of(th, SettingsReloadedEvent)

    def test_cli_loader_keeps_command_line_overrides(self, temp_home, monkeypatch):
        from claude_swap import cli
        from claude_swap.autoswitch import TickOutcome as Outcome

        captured: dict = {}

        class FakeEngine:
            def __init__(self, switcher, settings, on_event, *, dry_run=False,
                         state_path=None, clock=None, settings_loader=None):
                captured["loader"] = settings_loader
                captured["root"] = switcher.backup_dir

            def tick(self):
                return Outcome.NO_ACTION

        monkeypatch.setattr("claude_swap.autoswitch.AutoSwitchEngine", FakeEngine)
        with patch("os.geteuid", return_value=1000, create=True), \
                pytest.raises(SystemExit):
            cli._auto_command(["--once", "--interval", "30"])
        captured["root"].mkdir(parents=True, exist_ok=True)
        settings_path(captured["root"]).write_text(
            json.dumps({"autoswitch": {"threshold": 70, "intervalSeconds": 120}})
        )
        reloaded = captured["loader"]()
        assert reloaded.threshold == 70.0  # from the file
        assert reloaded.interval_seconds == 30.0  # the CLI flag still wins


# ---------------------------------------------------------------------------
# `cswap policy list`: state, blockedWindow, limit sources, wording
# ---------------------------------------------------------------------------


class TestPolicyListVocabulary:
    @pytest.fixture
    def fleet(self, temp_home):
        from claude_swap.usage_store import FetchRecord

        h = EngineHarness(temp_home)
        emails = {"1": "a@co.com", "2": "b@co.com", "3": "c@co.com", "4": "d@co.com"}
        for num, email in emails.items():
            h.seed(int(num), email)
        h.make_live("c@co.com", 3)
        switcher = h.switcher
        # `policy list` reads the store on wall time; record on it too.
        switcher._usage_store.clock = time.time
        (switcher.backup_dir / "settings.json").write_text(
            json.dumps({"autoswitch": {"threshold": 98}})
        )
        root = switcher.backup_dir
        set_account_policy(root, "a@co.com", "", priority=1)        # 1: no own
        set_account_policy(root, "b@co.com", "", max_5h=30.0)       # 2: own 5h
        set_account_policy(root, "c@co.com", "", reserve=True)      # 3: active reserve
        set_account_policy(root, "gone@co.com", "", max_7d=40.0)    # orphan
        switcher._usage_store.record(
            {
                "1": FetchRecord(usage={"five_hour": {"pct": 99.0},
                                        "seven_day": {"pct": 10.0}}),
                "2": FetchRecord(usage={"five_hour": {"pct": 31.0},
                                        "seven_day": {"pct": 10.0}}),
                "3": FetchRecord(usage={"five_hour": {"pct": 5.0},
                                        "seven_day": {"pct": 5.0}}),
            },
            {n: (e, "") for n, e in emails.items()},
        )
        with patch("os.geteuid", return_value=1000, create=True):
            yield switcher

    def _rows(self, capsys) -> dict:
        from claude_swap import cli

        capsys.readouterr()
        cli._policy_command(["list", "--json"])
        doc = json.loads(capsys.readouterr().out)
        return {r["number"]: r for r in doc["accounts"]}

    def test_json_rows(self, fleet, capsys):
        rows = self._rows(capsys)
        one, two, three, four = rows["1"], rows["2"], rows["3"], rows["4"]
        assert (one["state"], one["blockedWindow"]) == ("limit", "5h")
        assert one["status"] == "Over threshold (5h)"
        assert one["reason"] == "threshold 5h: 99% / 98%"
        assert (one["max5hSource"], one["max7dSource"]) == ("threshold", "threshold")
        assert (two["state"], two["blockedWindow"]) == ("limit", "5h")
        assert two["status"] == "Policy-limit (5h)"
        assert two["reason"] == "personal 5h limit: 31% / 30%"
        assert (two["max5hSource"], two["max7dSource"]) == ("own", "threshold")
        assert (three["state"], three["status"]) == ("reserve", "Reserve")
        assert three["blockedWindow"] is None
        assert four["state"] == "unknown"
        orphan = rows[None]
        assert orphan["state"] == "unmanaged" and orphan["max7dSource"] == "own"
        assert orphan["max5hSource"] == "threshold"
        # Existing fields are untouched.
        assert one["configured"] and not four["configured"]

    def test_disabled_and_ready_states(self, fleet, capsys):
        fleet.set_account_disabled("1", True)
        rows = self._rows(capsys)
        assert (rows["1"]["state"], rows["1"]["blockedWindow"]) == ("disabled", None)
        fleet.set_account_disabled("1", False)
        set_account_policy(fleet.backup_dir, "c@co.com", "", reserve=False)
        rows = self._rows(capsys)
        assert rows["3"]["state"] == "active"

    def test_human_table_marks_limits_not_status(self, fleet, capsys):
        from claude_swap import cli

        capsys.readouterr()
        cli._policy_command(["list"])
        out = capsys.readouterr().out
        row1 = next(line for line in out.splitlines() if "1: a@co.com" in line)
        row2 = next(line for line in out.splitlines() if "2: b@co.com" in line)
        assert "98% (default)" in row1 and row1.rstrip().endswith("Over threshold (5h)")
        assert "30%  " in row2 and row2.count("(default)") == 1  # 7d only
        assert row2.rstrip().endswith("Policy-limit (5h)")
        assert "personal" not in row1


# ---------------------------------------------------------------------------
# TUI auto view styles the policy events like the CLI does
# ---------------------------------------------------------------------------


class TestAutoViewRoles:
    @pytest.mark.parametrize(
        "event, role",
        [
            (PolicyLimitEvent(number="2", name="#2", reason="r"), "sev_warn"),
            (ReserveInUseEvent(number="3", name="#3"), "sev_warn"),
            (PolicyExhaustedEvent(accounts={"2": "r"}), "sev_crit"),
        ],
    )
    def test_policy_events_are_emphasized(self, event, role):
        from claude_swap.tui.autoview import event_text
        from claude_swap.tui.theme import Palette

        text = event_text(event, palette=Palette.DARK)
        styles = {str(span.style) for span in text.spans}
        assert str(getattr(Palette.DARK, role)) in styles
