"""Tests for the per-account usage policy (account_policy.py) and its
integration into the auto-switch engine."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from claude_swap.account_policy import (
    DEFAULT_PRIORITY,
    POLICY_FILENAME,
    AccountPolicy,
    all_policy_blocked,
    candidate_tiers,
    evaluate_account_policy,
    load_policies,
    parse_limit,
    parse_priority,
    remove_account_policy,
    set_account_policy,
)
from claude_swap.autoswitch import (
    ConfigWarningEvent,
    NoSwitchEvent,
    PolicyExhaustedEvent,
    PolicyLimitEvent,
    PollEvent,
    ReserveInUseEvent,
    SwitchEvent,
    TickOutcome,
)
from claude_swap.exceptions import ConfigError
from tests.test_autoswitch import EngineHarness


def _u(h5: float, d7: float = 0.0) -> dict:
    return {"five_hour": {"pct": h5}, "seven_day": {"pct": d7}}


# ---------------------------------------------------------------------------
# Pure evaluation
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_under_limit_is_eligible(self):  # Test 1
        st = evaluate_account_policy(
            AccountPolicy("a@x", max_5h=90, max_7d=90), _u(50), 90.0
        )
        assert st.configured and st.known and not st.blocked

    def test_at_5h_limit_is_blocked(self):  # Test 2
        st = evaluate_account_policy(AccountPolicy("b@x", max_5h=30), _u(30), 90.0)
        assert st.blocked and st.blocked_5h and not st.blocked_7d
        assert st.reason() == "personal 5h limit: 30% / 30%"

    def test_over_7d_limit_is_blocked(self):  # Test 3
        st = evaluate_account_policy(
            AccountPolicy("b@x", max_5h=90, max_7d=30), _u(5, 31), 90.0
        )
        assert st.blocked and st.blocked_7d and not st.blocked_5h

    @pytest.mark.parametrize(
        "h5, d7", [(31, 10), (20, 51)], ids=["5h-over", "7d-over"]
    )
    def test_windows_are_checked_independently(self, h5, d7):
        # Never folded into one binding number: either window alone blocks.
        st = evaluate_account_policy(
            AccountPolicy("b@x", max_5h=30, max_7d=50), _u(h5, d7), 90.0
        )
        assert st.blocked

    def test_reset_makes_it_eligible_again(self):  # Test 6
        policy = AccountPolicy("b@x", max_5h=30)
        assert evaluate_account_policy(policy, _u(30), 90.0).blocked
        assert not evaluate_account_policy(policy, _u(0), 90.0).blocked

    def test_no_policy_falls_back_to_global_threshold(self):  # Test 7
        st = evaluate_account_policy(None, _u(89), 90.0)
        assert not st.configured and not st.blocked and not st.reserve
        assert (st.max_5h, st.max_7d, st.priority) == (90.0, 90.0, DEFAULT_PRIORITY)
        assert evaluate_account_policy(None, _u(90), 90.0).blocked

    def test_unset_field_falls_back_per_window(self):
        st = evaluate_account_policy(AccountPolicy("b@x", max_5h=30), _u(5, 85), 90.0)
        assert (st.max_5h, st.max_7d) == (30.0, 90.0)
        assert not st.blocked

    def test_unknown_usage_is_never_blocked(self):
        for usage in (None, "token expired", {}):
            st = evaluate_account_policy(AccountPolicy("b@x", max_5h=1), usage, 90.0)
            assert not st.known and not st.blocked

    def test_scoped_model_window_keeps_global_threshold(self):
        usage = _u(5, 5) | {"scoped": [{"name": "Fable", "pct": 95.0}]}
        st = evaluate_account_policy(
            AccountPolicy("b@x", max_5h=99, max_7d=99), usage, 90.0, ("Fable",)
        )
        assert st.blocked and not st.blocked_5h and not st.blocked_7d


class TestTiers:
    def _st(self, pct, *, priority=None, reserve=False, max_5h=50.0, usage=True):
        return evaluate_account_policy(
            AccountPolicy("x", max_5h=max_5h, priority=priority, reserve=reserve),
            _u(pct) if usage else None,
            90.0,
        )

    def test_reserve_comes_after_every_normal_group(self):
        statuses = {
            "1": self._st(10, priority=1, reserve=True),
            "2": self._st(10, priority=3),
            "3": self._st(10, priority=2),
        }
        assert candidate_tiers(["1", "2", "3"], statuses) == [["3"], ["2"], ["1"]]

    def test_blocked_dropped_unknown_kept_order_preserved(self):
        statuses = {
            "1": self._st(60),                 # blocked (max 50)
            "2": self._st(0, usage=False),     # unknown usage
            "3": self._st(10),
        }
        assert candidate_tiers(["1", "2", "3"], statuses) == [["2", "3"]]

    def test_voluntary_filters(self):
        statuses = {
            "1": self._st(10, priority=1),
            "2": self._st(10, priority=5),
            "3": self._st(10, priority=1, reserve=True),
        }
        assert candidate_tiers(
            ["1", "2", "3"], statuses, include_reserve=False, max_priority=1
        ) == [["1"]]

    def test_all_policy_blocked(self):
        blocked = {"1": self._st(60), "2": self._st(70)}
        assert all_policy_blocked(["1", "2"], blocked)
        assert not all_policy_blocked([], blocked)
        mixed = blocked | {"3": self._st(0, usage=False)}
        assert not all_policy_blocked(["1", "2", "3"], mixed)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_missing_file_is_empty(self, tmp_path: Path):
        table = load_policies(tmp_path)
        assert not table and table.warnings == ()

    @pytest.mark.parametrize("content", ["{not json", "[]", '{"accounts": []}'])
    def test_corrupt_file_never_raises(self, tmp_path: Path, content):  # Test 8
        (tmp_path / POLICY_FILENAME).write_text(content)
        table = load_policies(tmp_path)
        assert not table and table.warnings

    def test_bad_fields_degrade_to_defaults(self, tmp_path: Path):
        (tmp_path / POLICY_FILENAME).write_text(json.dumps({
            "version": 1,
            "accounts": {
                "a@x": {"email": "a@x", "max5h": 150, "max7d": -3,
                        "priority": "high", "reserve": "yes", "enabled": False},
                "b@x|org": {"max5h": "30", "max7d": 40, "priority": 2.0},
                "junk": 7,
            },
        }))
        table = load_policies(tmp_path)
        a = table.get("a@x", "")
        assert a == AccountPolicy("a@x")
        b = table.get("b@x", "org")  # identity recovered from the key
        assert (b.max_5h, b.max_7d, b.priority) == (None, 40.0, 2)
        assert len(table.warnings) == 7

    def test_set_update_remove_round_trip(self, tmp_path: Path):
        set_account_policy(tmp_path, "b@x", "org-1", max_5h=30.0, max_7d=30.0)
        set_account_policy(tmp_path, "b@x", "org-1", priority=2, reserve=True)
        set_account_policy(tmp_path, "c@x", "", max_5h=20.0)
        table = load_policies(tmp_path)
        assert table.get("b@x", "org-1") == AccountPolicy(
            "b@x", "org-1", max_5h=30.0, max_7d=30.0, priority=2, reserve=True
        )
        # None clears a field back to its fallback.
        set_account_policy(tmp_path, "b@x", "org-1", max_7d=None)
        assert load_policies(tmp_path).get("b@x", "org-1").max_7d is None
        assert remove_account_policy(tmp_path, "b@x", "org-1")
        assert not remove_account_policy(tmp_path, "b@x", "org-1")
        assert load_policies(tmp_path).get("c@x", "").max_5h == 20.0

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
    def test_written_with_private_mode(self, tmp_path: Path):
        set_account_policy(tmp_path, "b@x", "", max_5h=30.0)
        mode = stat.S_IMODE(os.stat(tmp_path / POLICY_FILENAME).st_mode)
        assert mode == 0o600

    def test_write_refuses_to_clobber_corrupt_file(self, tmp_path: Path):
        (tmp_path / POLICY_FILENAME).write_text("{oops")
        with pytest.raises(ConfigError):
            set_account_policy(tmp_path, "b@x", "", max_5h=30.0)
        assert (tmp_path / POLICY_FILENAME).read_text() == "{oops"

    def test_strict_cli_parsing(self):
        assert parse_limit("30", "--max-5h") == 30.0
        assert parse_priority("2") == 2
        for bad in ("0", "101", "abc"):
            with pytest.raises(ConfigError):
                parse_limit(bad, "--max-5h")
        for bad in ("1.5", "x", "-1"):
            with pytest.raises(ConfigError):
                parse_priority(bad)


# ---------------------------------------------------------------------------
# Engine integration. Accounts: 1=A (a@), 2=B (b@), 3=C (c@); B is live.
# ---------------------------------------------------------------------------


@pytest.fixture
def ph(temp_home: Path) -> EngineHarness:
    h = EngineHarness(temp_home)
    for num, email in ((1, "a@example.com"), (2, "b@example.com"), (3, "c@example.com")):
        h.seed(num, email)
    h.make_live("b@example.com", 2)
    root = h.switcher.backup_dir
    set_account_policy(root, "a@example.com", "", max_5h=90.0, max_7d=90.0, priority=1)
    set_account_policy(root, "b@example.com", "", max_5h=30.0, max_7d=30.0, priority=2)
    set_account_policy(
        root, "c@example.com", "", max_5h=20.0, max_7d=20.0, priority=3, reserve=True
    )
    return h


def _of(h: EngineHarness, cls):
    return [e for e in h.events if isinstance(e, cls)]


class TestEnginePolicy:
    def test_under_personal_limit_stays(self, ph):
        assert ph.tick_with_usage({"1": _u(10), "2": _u(29), "3": _u(0)}) \
            is TickOutcome.NO_ACTION
        [no] = _of(ph, NoSwitchEvent)
        assert no.reason == "below-threshold" and "personal policy" in no.detail
        [poll] = _of(ph, PollEvent)
        assert poll.personal_limits == {"5h": 30.0, "7d": 30.0}
        assert "personal limit 5h 30%" in poll.human()

    def test_personal_limit_switches_to_normal_not_reserve(self, ph):  # Test 4
        # C has more room than A, but A is a normal account and must win.
        outcome = ph.tick_with_usage({"1": _u(65), "2": _u(30), "3": _u(10)})
        assert outcome is TickOutcome.SWITCHED
        [limit] = _of(ph, PolicyLimitEvent)
        assert "personal 5h limit: 30% / 30%" in limit.human()
        assert "not an Anthropic rate limit" in limit.human()
        [sw] = _of(ph, SwitchEvent)
        assert sw.trigger == "policy-limit" and sw.to_ref["number"] == 1
        assert ph.active_number() == 1
        assert not _of(ph, ReserveInUseEvent)

    def test_personal_limit_beats_hysteresis(self, ph):
        # A (35 headroom) is far worse than B (69): proactive hysteresis would
        # refuse, the personal hard limit must not.
        ph.tick_with_usage({"1": _u(65), "2": _u(5, 31), "3": _u(0)})
        assert ph.active_number() == 1

    def test_personal_limit_bypasses_cooldown(self, ph):
        ph.engine._mutate_state(lambda s: s.update(lastSwitchAt=ph.clock.now))
        assert ph.tick_with_usage({"1": _u(10), "2": _u(30), "3": _u(0)}) \
            is TickOutcome.SWITCHED

    def test_reserve_only_when_every_normal_is_blocked(self, ph):  # Test 5
        outcome = ph.tick_with_usage({"1": _u(90), "2": _u(30), "3": _u(5)})
        assert outcome is TickOutcome.SWITCHED
        assert ph.active_number() == 3
        [notice] = _of(ph, ReserveInUseEvent)
        assert "Using reserve account Account-3" in notice.human()

    def test_everything_blocked_holds_without_switching(self, ph):  # spec §12
        usage = {"1": _u(90), "2": _u(30), "3": _u(20)}
        for _ in range(3):
            assert ph.tick_with_usage(usage) is TickOutcome.BLOCKED
            ph.clock.advance(60)
        assert ph.switcher.current_account_number() == "2"
        assert not _of(ph, SwitchEvent)
        report = _of(ph, PolicyExhaustedEvent)[0]
        text = report.human()
        assert "No account is available under your personal usage policy." in text
        assert "reserve, policy limit reached" in text
        assert ph.engine._next_delay(TickOutcome.BLOCKED) \
            <= ph.settings.interval_seconds * 1.1

    def test_reset_restores_eligibility(self, ph):  # Test 6 (engine)
        assert ph.tick_with_usage({"1": _u(90), "2": _u(30), "3": _u(20)}) \
            is TickOutcome.BLOCKED
        ph.clock.advance(5 * 3600)
        assert ph.tick_with_usage({"1": _u(0), "2": _u(30), "3": _u(20)}) \
            is TickOutcome.SWITCHED
        assert ph.active_number() == 1

    def test_priority_beats_headroom_strategy_breaks_ties(self, ph):
        ph.seed(4, "d@example.com")
        root = ph.switcher.backup_dir
        set_account_policy(root, "d@example.com", "", max_5h=90.0, priority=3)
        ph.tick_with_usage({"1": _u(80), "2": _u(30), "3": _u(0), "4": _u(5)})
        assert ph.active_number() == 1  # priority 1 despite less headroom
        # Same priority: the existing `best` strategy picks the most headroom.
        set_account_policy(root, "d@example.com", "", priority=2)
        ph.make_live("b@example.com", 2)
        set_account_policy(root, "a@example.com", "", priority=2)
        ph.events.clear()
        ph.tick_with_usage({"1": _u(80), "2": _u(30), "3": _u(0), "4": _u(5)})
        assert ph.active_number() == 4

    def test_disabled_account_is_not_an_auto_candidate(self, ph):  # Test 9
        ph.switcher.set_account_disabled("1", True)
        ph.tick_with_usage({"1": _u(0), "2": _u(30), "3": _u(5)})
        assert ph.active_number() == 3  # A has room but is disabled

    def test_manual_switch_to_disabled_account_still_works(self, ph):  # Test 10
        ph.switcher.set_account_disabled("1", True)
        result = ph.switcher.switch_to("1", json_output=True)
        assert result and result.get("switched")

    def test_reserve_returns_to_normal_when_one_qualifies(self, ph):
        ph.make_live("c@example.com", 3)
        outcome = ph.tick_with_usage({"1": _u(50), "2": _u(30), "3": _u(5)})
        assert outcome is TickOutcome.SWITCHED
        [sw] = _of(ph, SwitchEvent)
        assert sw.trigger == "reserve-return" and ph.active_number() == 1

    def test_reserve_holds_while_normals_blocked_or_in_cooldown(self, ph):
        ph.make_live("c@example.com", 3)
        assert ph.tick_with_usage({"1": _u(90), "2": _u(30), "3": _u(5)}) \
            is TickOutcome.NO_ACTION
        assert _of(ph, NoSwitchEvent)[-1].reason == "reserve-in-use"
        ph.engine._mutate_state(lambda s: s.update(lastSwitchAt=ph.clock.now))
        assert ph.tick_with_usage({"1": _u(10), "2": _u(30), "3": _u(5)}) \
            is TickOutcome.NO_ACTION
        assert _of(ph, NoSwitchEvent)[-1].reason == "cooldown"

    def test_personal_limit_above_global_threshold_is_honored(self, ph):
        set_account_policy(
            ph.switcher.backup_dir, "b@example.com", "", max_5h=95.0, max_7d=95.0
        )
        assert ph.tick_with_usage({"1": _u(0), "2": _u(92), "3": _u(0)}) \
            is TickOutcome.NO_ACTION

    def test_alias_rename_keeps_policy(self, ph):
        ph.switcher.set_alias("2", "bee")
        ph.switcher.set_alias("bee", "bravo")
        ph.tick_with_usage({"1": _u(10), "2": _u(30), "3": _u(0)})
        [limit] = _of(ph, PolicyLimitEvent)
        assert limit.name == "Account-2 (bravo)"

    def test_consume_first_never_nudges_down_priority_or_to_reserve(self, ph):
        ph.settings = ph.settings.__class__(strategy="consume-first")
        ph.engine = ph._make_engine()
        ph.make_live("a@example.com", 1)
        soon = "2099-01-01T00:00:00Z"
        later = "2099-02-01T00:00:00Z"

        def u(pct, reset):
            return {"five_hour": {"pct": pct},
                    "seven_day": {"pct": pct, "resets_at": reset}}

        # B (priority 2) and C (reserve) reset sooner than A, but A is top
        # priority and healthy: a voluntary nudge must not leave it.
        ph.tick_with_usage({"1": u(10, later), "2": u(5, soon), "3": u(5, soon)})
        assert ph.active_number() == 1 and not _of(ph, SwitchEvent)


class TestEngineWithoutPolicy:
    def test_unconfigured_account_uses_global_threshold(self, ph):  # Test 7
        os.remove(ph.switcher.backup_dir / POLICY_FILENAME)
        set_account_policy(ph.switcher.backup_dir, "c@example.com", "", reserve=True)
        ph.tick_with_usage({"1": _u(0), "2": _u(50), "3": _u(0)})
        [no] = _of(ph, NoSwitchEvent)
        assert no.detail == "5h 50% < 90%"
        ph.events.clear()
        ph.tick_with_usage({"1": _u(0), "2": _u(91), "3": _u(0)})
        [sw] = _of(ph, SwitchEvent)
        assert sw.trigger == "proactive" and sw.to_ref["number"] == 1

    def test_corrupt_policy_file_degrades_to_plain_engine(self, ph):  # Test 8
        (ph.switcher.backup_dir / POLICY_FILENAME).write_text("{broken")
        for _ in range(2):
            assert ph.tick_with_usage({"1": _u(0), "2": _u(30), "3": _u(0)}) \
                is TickOutcome.NO_ACTION
        # Plain threshold semantics (30% < 90%), and the warning is not spammed.
        assert _of(ph, NoSwitchEvent)[-1].detail == "5h 30% < 90%"
        assert len(_of(ph, ConfigWarningEvent)) == 1


# ---------------------------------------------------------------------------
# CLI: `cswap policy ...`
# ---------------------------------------------------------------------------


class TestPolicyCommand:
    @pytest.fixture
    def cli_env(self, temp_home):
        from unittest.mock import patch

        from claude_swap.switcher import ClaudeAccountSwitcher

        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._init_sequence_file()
        data = switcher._get_sequence_data()
        for num, email in (("1", "a@co.com"), ("2", "b@co.com")):
            data["accounts"][num] = {
                "email": email, "uuid": f"u{num}", "organizationUuid": "",
                "organizationName": "", "added": "2024-01-01T00:00:00Z",
            }
        data["accounts"]["2"]["alias"] = "bee"
        data["sequence"] = [1, 2]
        switcher._write_json(switcher.sequence_file, data)
        with patch("os.geteuid", return_value=1000, create=True):
            yield switcher

    def _run(self, *argv):
        from claude_swap import cli

        cli._policy_command(list(argv))

    def test_set_by_alias_then_list(self, cli_env, capsys):
        self._run("set", "bee", "--max-5h", "30", "--max-7d", "30",
                  "--priority", "2", "--reserve")
        table = load_policies(cli_env.backup_dir)
        assert table.get("b@co.com", "") == AccountPolicy(
            "b@co.com", max_5h=30.0, max_7d=30.0, priority=2, reserve=True
        )
        capsys.readouterr()
        self._run("list")
        out = capsys.readouterr().out
        assert "ACCOUNT" in out and "2: bee" in out and "30%" in out
        assert "Reserve" in out
        assert "1: a@co.com" in out and "(default)" in out

    def test_list_json(self, cli_env, capsys):
        self._run("set", "1", "--max-5h", "90")
        capsys.readouterr()
        self._run("list", "--json")
        rows = json.loads(capsys.readouterr().out)["accounts"]
        assert rows[0]["max5h"] == 90.0 and rows[0]["configured"]
        assert rows[1]["max5h"] == 90.0 and not rows[1]["configured"]  # fallback

    def test_set_rejects_bad_values_without_writing(self, cli_env, capsys):
        with pytest.raises(SystemExit):
            self._run("set", "1", "--max-5h", "150")
        assert "between 1 and 100" in capsys.readouterr().err
        assert not (cli_env.backup_dir / POLICY_FILENAME).exists()

    def test_default_clears_a_field(self, cli_env):
        self._run("set", "1", "--max-5h", "30", "--priority", "1")
        self._run("set", "1", "--max-5h", "default")
        policy = load_policies(cli_env.backup_dir).get("a@co.com", "")
        assert policy.max_5h is None and policy.priority == 1

    def test_disable_enable_reuse_rotation_flag(self, cli_env):
        self._run("disable", "bee")
        assert cli_env.is_account_disabled("2")
        self._run("enable", "2")
        assert not cli_env.is_account_disabled("2")

    def test_remove_and_show(self, cli_env, capsys):
        self._run("set", "1", "--max-7d", "40")
        capsys.readouterr()
        self._run("show", "1")
        out = capsys.readouterr().out
        assert "limit 40%" in out and "account_limits.json" in out
        self._run("remove", "1")
        assert not load_policies(cli_env.backup_dir)

    def test_remove_orphaned_entry_by_email(self, cli_env):
        set_account_policy(cli_env.backup_dir, "gone@co.com", "", max_5h=10.0)
        self._run("remove", "gone@co.com")
        assert not load_policies(cli_env.backup_dir)


# ---------------------------------------------------------------------------
# Time-window schedules
# ---------------------------------------------------------------------------


def _window(epoch: float, start_min: int, end_min: int) -> str:
    """A local-time HH:MM-HH:MM window offset (minutes) from ``epoch``."""
    from datetime import datetime

    dt = datetime.fromtimestamp(epoch)
    m = dt.hour * 60 + dt.minute
    a, b = (m + start_min) % 1440, (m + end_min) % 1440
    return f"{a // 60:02d}:{a % 60:02d}-{b // 60:02d}:{b % 60:02d}"


class TestSchedule:
    def test_in_schedule_plain_and_across_midnight(self):
        from datetime import datetime

        from claude_swap.account_policy import in_schedule

        at = lambda h, m=0: datetime(2026, 1, 1, h, m)  # noqa: E731
        assert in_schedule(("09:00-14:00",), at(9))
        assert not in_schedule(("09:00-14:00",), at(14))
        assert in_schedule(("22:00-02:00",), at(23)) and in_schedule(("22:00-02:00",), at(1))
        assert not in_schedule(("22:00-02:00",), at(12))
        assert in_schedule(("01:00-02:00", "12:00-24:00"), at(23, 59))
        assert in_schedule((), at(3))  # no schedule = any time

    def test_strict_parse(self):
        from claude_swap.account_policy import parse_schedule

        assert parse_schedule("9:00-14:00, 22:00-2:00") == ("09:00-14:00", "22:00-02:00")
        for bad in ("25:00-26:00", "9-14", "10:00-10:00", ""):
            with pytest.raises(ConfigError):
                parse_schedule(bad)

    def test_off_schedule_blocks_even_without_usage(self):
        now = 1_000_000.0
        policy = AccountPolicy("d@x", schedule=(_window(now, 60, 120),))
        st = evaluate_account_policy(policy, None, 90.0, now=now)
        assert st.off_schedule and st.blocked and not st.scheduled_now
        assert st.reason().startswith("outside its schedule")
        inside = AccountPolicy("d@x", schedule=(_window(now, -60, 60),))
        st = evaluate_account_policy(inside, _u(10), 90.0, now=now)
        assert st.scheduled_now and not st.blocked

    def test_load_accepts_string_and_warns_on_bad(self, tmp_path: Path):
        (tmp_path / POLICY_FILENAME).write_text(json.dumps({"accounts": {
            "a@x": {"schedule": "9:00-14:00"},
            "b@x": {"schedule": ["nope"]},
        }}))
        table = load_policies(tmp_path)
        assert table.get("a@x", "").schedule == ("09:00-14:00",)
        assert table.get("b@x", "").schedule == () and len(table.warnings) == 1

    def test_scheduled_tier_first_regardless_of_priority(self):
        now = 1_000_000.0
        open_ = (_window(now, -60, 60),)
        statuses = {
            "1": evaluate_account_policy(AccountPolicy("a", priority=1), _u(10), 90.0, now=now),
            "4": evaluate_account_policy(
                AccountPolicy("d", priority=9, schedule=open_), _u(10), 90.0, now=now),
        }
        assert candidate_tiers(["1", "4"], statuses) == [["4"], ["1"]]
        assert candidate_tiers(["1", "4"], statuses, only_scheduled=True) == [["4"]]


@pytest.fixture
def sh(ph: EngineHarness) -> EngineHarness:
    """ph plus account 4 = D, limited to a time window set per test."""
    ph.seed(4, "d@example.com")
    return ph


def _schedule_d(h: EngineHarness, window: str, **kw) -> None:
    set_account_policy(
        h.switcher.backup_dir, "d@example.com", "",
        max_5h=kw.get("max_5h", 50.0), priority=kw.get("priority", 5),
        schedule=(window,),
    )


class TestEngineSchedule:
    def test_window_open_moves_onto_scheduled_account(self, sh):
        _schedule_d(sh, _window(sh.clock.now, -60, 60))
        outcome = sh.tick_with_usage(
            {"1": _u(10), "2": _u(10), "3": _u(0), "4": _u(5)}
        )
        assert outcome is TickOutcome.SWITCHED
        [sw] = _of(sh, SwitchEvent)
        assert sw.trigger == "schedule" and sw.to_ref["number"] == 4

    def test_window_open_waits_for_cooldown(self, sh):
        _schedule_d(sh, _window(sh.clock.now, -60, 60))
        sh.engine._mutate_state(lambda s: s.update(lastSwitchAt=sh.clock.now))
        assert sh.tick_with_usage(
            {"1": _u(10), "2": _u(10), "3": _u(0), "4": _u(5)}
        ) is TickOutcome.NO_ACTION
        assert _of(sh, NoSwitchEvent)[-1].reason == "cooldown"

    def test_used_up_in_window_is_not_a_target(self, sh):
        _schedule_d(sh, _window(sh.clock.now, -60, 60))
        assert sh.tick_with_usage(
            {"1": _u(10), "2": _u(10), "3": _u(0), "4": _u(50)}
        ) is TickOutcome.NO_ACTION
        assert not _of(sh, SwitchEvent)

    def test_hitting_5h_limit_inside_window_moves_by_priority(self, sh):
        _schedule_d(sh, _window(sh.clock.now, -60, 60))
        sh.make_live("d@example.com", 4)
        sh.tick_with_usage({"1": _u(10), "2": _u(0), "3": _u(0), "4": _u(50)})
        [sw] = _of(sh, SwitchEvent)
        assert sw.trigger == "policy-limit" and sw.to_ref["number"] == 1

    def test_window_closing_moves_off_and_never_back(self, sh):
        _schedule_d(sh, _window(sh.clock.now, -60, 1))
        sh.make_live("d@example.com", 4)
        usage = {"1": _u(10), "2": _u(0), "3": _u(0), "4": _u(5)}
        assert sh.tick_with_usage(usage) is TickOutcome.NO_ACTION  # still inside
        sh.clock.advance(3 * 60)
        assert sh.tick_with_usage(usage) is TickOutcome.SWITCHED
        [limit] = _of(sh, PolicyLimitEvent)
        assert "outside its schedule" in limit.human()
        # Off-schedule D is never picked, even as the emptiest account.
        sh.make_live("b@example.com", 2)
        sh.events.clear()
        sh.tick_with_usage({"1": _u(90), "2": _u(30), "3": _u(5), "4": _u(0)})
        assert _of(sh, SwitchEvent)[0].to_ref["number"] == 3  # reserve, not D

    def test_cli_set_and_list_schedule(self, temp_home, capsys):
        from unittest.mock import patch

        from claude_swap import cli
        from claude_swap.switcher import ClaudeAccountSwitcher

        sw = ClaudeAccountSwitcher()
        sw._setup_directories()
        sw._init_sequence_file()
        data = sw._get_sequence_data()
        data["accounts"]["1"] = {
            "email": "d@co.com", "uuid": "u1", "organizationUuid": "",
            "organizationName": "", "added": "2024-01-01T00:00:00Z",
        }
        data["sequence"] = [1]
        sw._write_json(sw.sequence_file, data)
        with patch("os.geteuid", return_value=1000, create=True):
            cli._policy_command(["set", "1", "--schedule", "9:00-14:00"])
            assert load_policies(sw.backup_dir).get("d@co.com", "").schedule \
                == ("09:00-14:00",)
            capsys.readouterr()
            cli._policy_command(["list"])
            assert "09:00-14:00" in capsys.readouterr().out
            cli._policy_command(["set", "1", "--schedule", "default"])
            assert load_policies(sw.backup_dir).get("d@co.com", "").schedule == ()
            with pytest.raises(SystemExit):
                cli._policy_command(["set", "1", "--schedule", "9-14"])


# ---------------------------------------------------------------------------
# autoswitch.onSwitchCommand
# ---------------------------------------------------------------------------


class TestOnSwitchHook:
    def _wait_for(self, path: Path, timeout: float = 10.0) -> str:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists() and path.read_text().strip():
                return path.read_text().strip()
            time.sleep(0.05)
        return ""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
    def test_runs_after_switch_with_switch_details(self, ph, tmp_path):
        out = tmp_path / "hook.txt"
        ph.settings = ph.settings.__class__(
            on_switch_command=(
                f'echo "$CSWAP_FROM>$CSWAP_TO $CSWAP_TO_EMAIL $CSWAP_TRIGGER" > {out}'
            )
        )
        ph.engine = ph._make_engine()
        ph.tick_with_usage({"1": _u(10), "2": _u(30), "3": _u(0)})
        assert self._wait_for(out) == "2>1 a@example.com policy-limit"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
    def test_not_run_on_dry_run_or_without_switch(self, ph, tmp_path):
        import time

        out = tmp_path / "hook.txt"
        ph.settings = ph.settings.__class__(on_switch_command=f"echo ran > {out}")
        ph.engine = ph._make_engine(dry_run=True)
        ph.tick_with_usage({"1": _u(10), "2": _u(30), "3": _u(0)})
        ph.engine = ph._make_engine()
        ph.tick_with_usage({"1": _u(10), "2": _u(5), "3": _u(0)})  # below limit
        time.sleep(0.5)
        assert not out.exists()


# ---------------------------------------------------------------------------
# Bridge to upstream's per-account / weekly switch lines (#318, #474)
# ---------------------------------------------------------------------------


class TestUpstreamLineBridge:
    def test_unset_7d_falls_back_to_weekly_threshold(self):
        st = evaluate_account_policy(
            AccountPolicy("b@x", max_5h=30), _u(5, 95), 90.0, weekly_threshold=97.0
        )
        assert (st.max_5h, st.max_7d) == (30.0, 97.0) and not st.blocked
        st = evaluate_account_policy(None, _u(5, 95), 90.0, weekly_threshold=97.0)
        assert (st.max_5h, st.max_7d) == (90.0, 97.0) and not st.blocked

    def test_scoped_window_follows_weekly_line_not_max_7d(self):
        usage = _u(5, 5) | {"scoped": [{"name": "Fable", "pct": 95.0}]}
        st = evaluate_account_policy(
            AccountPolicy("b@x", max_7d=50), usage, 90.0, ("Fable",),
            weekly_threshold=97.0,
        )
        assert not st.blocked

    def test_unconfigured_candidate_uses_its_own_threshold_override(self, ph):
        # A has no policy but `cswap threshold 1 95`: at 92% it is a valid
        # landing spot under its own line, so B's personal limit moves to A
        # rather than falling through to the reserve account C.
        remove_account_policy(ph.switcher.backup_dir, "a@example.com", "")
        ph.switcher.set_account_threshold("1", 95)
        outcome = ph.tick_with_usage({"1": _u(92), "2": _u(31), "3": _u(0)})
        assert outcome is TickOutcome.SWITCHED
        [sw] = _of(ph, SwitchEvent)
        assert sw.trigger == "policy-limit" and sw.to_ref["number"] == 1

    def test_poll_cadence_pins_to_lowest_personal_limit(self, ph):
        ph.tick_with_usage({"1": _u(10), "2": _u(5), "3": _u(0)})
        threshold, weekly, _models = ph.switcher._poll_inputs_override
        assert (threshold, weekly) == (20.0, None)

    def test_poll_cadence_unchanged_without_policy(self, ph):
        os.remove(ph.switcher.backup_dir / POLICY_FILENAME)
        ph.tick_with_usage({"1": _u(10), "2": _u(5), "3": _u(0)})
        threshold, weekly, _models = ph.switcher._poll_inputs_override
        assert (threshold, weekly) == (90.0, None)


class TestPolicyListFollowsUpstreamLines:
    def test_list_json_uses_threshold_override_and_weekly_line(
        self, temp_home, capsys
    ):
        from unittest.mock import patch

        from claude_swap import cli
        from claude_swap.switcher import ClaudeAccountSwitcher

        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._init_sequence_file()
        data = switcher._get_sequence_data()
        for num, email in (("1", "a@co.com"), ("2", "b@co.com")):
            data["accounts"][num] = {
                "email": email, "uuid": f"u{num}", "organizationUuid": "",
                "organizationName": "", "added": "2024-01-01T00:00:00Z",
            }
        data["sequence"] = [1, 2]
        switcher._write_json(switcher.sequence_file, data)
        (switcher.backup_dir / "settings.json").write_text(
            json.dumps({"autoswitch": {"weeklyThreshold": 97}})
        )
        with patch("os.geteuid", return_value=1000, create=True):
            switcher.set_account_threshold("1", 85)
            cli._policy_command(["set", "2", "--max-5h", "30"])
            capsys.readouterr()
            cli._policy_command(["list", "--json"])
        doc = json.loads(capsys.readouterr().out)
        assert doc["schemaVersion"] == 1
        rows = {r["number"]: r for r in doc["accounts"]}
        assert (rows["1"]["max5h"], rows["1"]["max7d"]) == (85.0, 97.0)
        assert (rows["2"]["max5h"], rows["2"]["max7d"]) == (30.0, 97.0)
        assert not rows["1"]["configured"] and rows["2"]["configured"]
