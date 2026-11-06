import os
import json
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import src.audit_logging.audit_logger as al

TEST_LOG = Path("./test_logs/audit_test.jsonl")
os.environ["AUDIT_LOG_PATH"] = str(TEST_LOG)


def reset_log():
    if TEST_LOG.exists():
        TEST_LOG.unlink()
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)

#Basic manual test
def test_log_event():
    reset_log()

    al.start_trace("trace123")

    entry = al.log_event(
        stage="input_filter",
        status="PASS",
        details={"message": "hello world"},
        risk_level="low",
        user_id="user_1"
    )

    assert TEST_LOG.exists()

    with open(TEST_LOG, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())

    assert data["stage"] == "input_filter"
    assert data["status"] == "PASS"
    assert data["trace_id"] == "trace123"
    assert data["details"]["message"] == "hello world"

    print("✔ log_event works")
def test_invalid_status():
    reset_log()

    try:
        al.log_event(
            stage="layer1",
            status="INVALID",
            details={}
        )
        assert False, "Expected ValueError for invalid status"
    except ValueError as e:
        print("✔ invalid status blocked:", e)
def test_reserved_key_error():
    reset_log()

    try:
        al.log_event(
            stage="layer1",
            status="PASS",
            details={"trace_id": "hack_attempt"}
        )
        assert False, "Expected ValueError for reserved key"
    except ValueError as e:
        print("✔ reserved key blocked:", e)
import time

def test_stage_timer():
    reset_log()

    al.start_trace("trace_timer")

    with al.StageTimer("llm_call", "PASS", user_id="u1") as t:
        time.sleep(0.1)
        t.details["tokens"] = 42

    assert TEST_LOG.exists()

    with open(TEST_LOG, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())

    assert data["stage"] == "llm_call"
    assert data["status"] == "PASS"
    assert "latency_ms" in data
    assert data["user_id"] == "u1"
    assert data["details"]["tokens"] == 42

    print("✔ StageTimer works")
                    

if __name__ == "__main__":
    test_log_event()
    test_invalid_status()
    test_reserved_key_error()
    test_stage_timer()