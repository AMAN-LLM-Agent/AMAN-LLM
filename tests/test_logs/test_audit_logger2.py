import importlib
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import src.audit_logging.audit_logger as al

TEST_LOG = Path("./test_logs/audit_test.jsonl")
os.environ["AUDIT_LOG_PATH"] = str(TEST_LOG)


@pytest.fixture(name="al")
def _al_fixture():
    """
    Fresh, isolated instance of the module per test.

    TEST_LOG is a fixed path (rather than tmp_path) to match this
    project's test layout, so isolation instead comes from wiping the
    log file and reloading the module -- which resets _last_hash,
    _log_path, and _current_trace_id -- before and after every test.
    """
    shutil.rmtree(TEST_LOG.parent, ignore_errors=True)
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)

    importlib.reload(al)

    yield al

    shutil.rmtree(TEST_LOG.parent, ignore_errors=True)


def _read_lines(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# =========================================================
# REDACTION
# =========================================================

def test_redacts_email_in_string_value(al):
    entry = al.log_event("stage", "BLOCK", {"attempt": "contact me at a@b.com"})
    assert "a@b.com" not in entry["details"]["attempt"]
    assert "[REDACTED:email]" in entry["details"]["attempt"]
    assert entry["redacted"] is True


def test_redacts_credit_card_pattern(al):
    entry = al.log_event("stage", "BLOCK", {"msg": "card 4111111111111111 used"})
    assert "4111111111111111" not in entry["details"]["msg"]
    assert entry["redacted"] is True


def test_redacts_by_key_name_regardless_of_value(al):
    entry = al.log_event("stage", "BLOCK", {"password": "hunter2"})
    assert entry["details"]["password"] == "[REDACTED]"
    assert entry["redacted"] is True


def test_redacts_nested_structures(al):
    entry = al.log_event(
        "stage", "BLOCK",
        {"user": {"email": "x@y.com", "meta": {"token": "abc123"}}},
    )
    assert "x@y.com" not in json.dumps(entry["details"])
    assert entry["details"]["user"]["meta"]["token"] == "[REDACTED]"


def test_no_redacted_flag_when_nothing_sensitive(al):
    entry = al.log_event("stage", "PASS", {"note": "all clean here"})
    assert "redacted" not in entry


def test_redact_pii_false_preserves_raw_value(al):
    entry = al.log_event(
        "stage", "PASS", {"attempt": "a@b.com"}, redact_pii=False,
    )
    assert entry["details"]["attempt"] == "a@b.com"
    assert "redacted" not in entry


# =========================================================
# RESERVED KEYS (including the new chain/redaction fields)
# =========================================================

@pytest.mark.parametrize("key", ["prev_hash", "entry_hash", "redacted", "event_id"])
def test_reserved_keys_still_rejected(al, key):
    with pytest.raises(ValueError, match="reserved keys"):
        al.log_event("stage", "PASS", {key: "x"})


# =========================================================
# HASH CHAIN
# =========================================================

def test_chain_is_valid_after_clean_writes(al):
    for i in range(5):
        al.log_event("stage", "PASS", {"i": i})
    result = al.verify_chain()
    assert result["ok"] is True
    assert result["checked"] == 5
    assert result["first_break_line"] is None


def test_genesis_prev_hash_on_first_entry(al):
    entry = al.log_event("stage", "PASS", {})
    assert entry["prev_hash"] == al._GENESIS_HASH


def test_entries_are_linked(al):
    e1 = al.log_event("stage", "PASS", {"i": 1})
    e2 = al.log_event("stage", "PASS", {"i": 2})
    assert e2["prev_hash"] == e1["entry_hash"]


@pytest.mark.parametrize("tamper_line", [0, 2, 4])
def test_tamper_detected_at_any_position(al, tamper_line):
    for i in range(5):
        al.log_event("stage", "PASS", {"i": i})

    path = al._get_log_path()
    lines = open(path).readlines()
    rec = json.loads(lines[tamper_line])
    rec["details"] = {"i": "TAMPERED"}
    lines[tamper_line] = json.dumps(rec) + "\n"
    with open(path, "w") as f:
        f.writelines(lines)

    result = al.verify_chain()
    assert result["ok"] is False
    assert result["first_break_line"] == tamper_line + 1  # 1-indexed


def test_deleting_a_line_breaks_the_chain(al):
    for i in range(4):
        al.log_event("stage", "PASS", {"i": i})

    path = al._get_log_path()
    lines = open(path).readlines()
    del lines[1]  # remove the second entry
    with open(path, "w") as f:
        f.writelines(lines)

    result = al.verify_chain()
    assert result["ok"] is False


def test_chain_continuity_across_simulated_restart(al, monkeypatch):
    al.log_event("stage", "PASS", {"i": 1})
    al.log_event("stage", "PASS", {"i": 2})

    log_path = al._get_log_path()

    # Simulate a fresh process: reset in-memory chain state but keep the file.
    monkeypatch.setattr(al, "_last_hash", None)

    al.log_event("stage", "PASS", {"i": 3})  # must pick up chain from disk

    result = al.verify_chain()
    assert result["ok"] is True
    assert result["checked"] == 3


def test_concurrent_writers_keep_chain_valid(al):
    def writer(n):
        for i in range(20):
            al.log_event("stage", "PASS", {"thread": n, "i": i})

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    result = al.verify_chain()
    assert result["ok"] is True
    assert result["checked"] == 160


# =========================================================
# TRACE ID
# =========================================================

def test_trace_id_is_not_truncated(al):
    trace_id = al.start_trace()
    assert len(trace_id) >= 32  # full uuid4, not the old [:8] slice


def test_caller_supplied_trace_id_is_respected(al):
    trace_id = al.start_trace("my-custom-trace")
    assert trace_id == "my-custom-trace"
    entry = al.log_event("stage", "PASS", {})
    assert entry["trace_id"] == "my-custom-trace"


# =========================================================
# STAGE TIMER (regression check -- behavior should be unchanged)
# =========================================================

def test_stage_timer_logs_error_status_on_exception(al):
    with pytest.raises(RuntimeError):
        with al.StageTimer("risky_stage") as t:
            t.details["before"] = True
            raise RuntimeError("boom")

    entries = _read_lines(al._get_log_path())
    assert entries[-1]["status"] == "ERROR"
    assert entries[-1]["details"]["error"] == "boom"
    assert entries[-1]["details"]["error_type"] == "RuntimeError"


def test_stage_timer_records_latency(al):
    with al.StageTimer("timed_stage"):
        pass

    entries = _read_lines(al._get_log_path())
    assert entries[-1]["latency_ms"] >= 0


# =========================================================
# VALIDATION (regression check)
# =========================================================

def test_invalid_status_rejected(al):
    with pytest.raises(ValueError):
        al.log_event("stage", "NOT_A_STATUS", {})


def test_invalid_risk_level_rejected(al):
    with pytest.raises(ValueError):
        al.log_event("stage", "PASS", {}, risk_level="extreme")
