"""
tests/test_sandbox.py
─────────────────────────────
Phase 5 — Docker Sandbox test suite

Covers three acceptance criteria from the roadmap:
  ✅ Step 1 – Build and test Docker sandbox image
  ✅ Step 2 – Integration with docker_executor / tool registry (dispatch)
  ✅ Step 3 – Isolation verification (no network, read-only FS, non-root)

Run:
    pytest tests/test_sandbox.py -v

Prerequisites:
    docker build -t aman-sandbox:latest -f docker/Dockerfile.sandbox .
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import pytest

# ─── Skip guard ───────────────────────────────────────────────────────────────
# All tests require Docker + the built image. Skip gracefully if unavailable.

def _docker_available() -> bool:
    try:
        import docker
        client = docker.from_env()
        client.images.get("aman-sandbox:latest")
        return True
    except Exception:
        return False

SKIP_IF_NO_DOCKER = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not running or 'aman-sandbox:latest' image not built."
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_container(payload: dict, extra_flags: list[str] | None = None) -> dict:
    """
    Run the sandbox container with the given payload and return parsed stdout.
    Uses subprocess so tests don't depend on the docker-py client being importable.
    """
    env_str = json.dumps(payload)
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:size=32m,mode=1777",
        "--memory", "256m",
        "--security-opt", "no-new-privileges",
        "-e", f"TOOL_PAYLOAD={env_str}",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append("aman-sandbox:latest")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build & smoke-test the image
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageBuild:

    @SKIP_IF_NO_DOCKER
    def test_image_exists(self):
        """The sandbox image must be present in local Docker."""
        import docker
        client = docker.from_env()
        image = client.images.get("aman-sandbox:latest")
        assert image is not None

    @SKIP_IF_NO_DOCKER
    def test_image_runs_without_payload(self):
        """Container should exit with a non-zero code and a JSON error if no payload."""
        res = _run_container({})   # empty dict → TOOL_PAYLOAD={}
        parsed = json.loads(res["stdout"])
        assert parsed["success"] is False
        assert "tool" in parsed["error"].lower() or "unknown" in parsed["error"].lower()

    @SKIP_IF_NO_DOCKER
    def test_runs_as_non_root(self):
        """The container process must NOT run as root (uid 0)."""
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:size=32m,mode=1777",
            "aman-sandbox:latest",
            "id", "-u",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        uid = proc.stdout.strip()
        assert uid != "0", f"Container is running as root (uid=0)! Got: {uid}"
        assert uid == "10001", f"Expected uid=10001, got {uid}"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Tool dispatch (docker_executor integration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolDispatch:

    @SKIP_IF_NO_DOCKER
    def test_web_search_stub(self):
        """web_search tool returns a structured stub response."""
        res = _run_container({"tool": "web_search", "args": {"query": "اختبار"}})
        assert res["exit_code"] == 0
        parsed = json.loads(res["stdout"])
        assert parsed["success"] is True
        assert parsed["result"]["tool"] == "web_search"
        assert "results" in parsed["result"]

    @SKIP_IF_NO_DOCKER
    def test_database_query_stub(self):
        """database_query returns a stub row-set for read queries."""
        res = _run_container({"tool": "database_query", "args": {"query": "SELECT 1"}})
        assert res["exit_code"] == 0
        parsed = json.loads(res["stdout"])
        assert parsed["success"] is True
        assert parsed["result"]["tool"] == "database_query"
        assert isinstance(parsed["result"]["rows"], list)

    @SKIP_IF_NO_DOCKER
    def test_database_blocks_write(self):
        """database_query must reject destructive SQL (defense-in-depth)."""
        res = _run_container(
            {"tool": "database_query", "args": {"query": "DROP TABLE users"}}
        )
        parsed = json.loads(res["stdout"])
        assert parsed["success"] is False
        assert "DROP" in parsed["error"] or "not allowed" in parsed["error"].lower()

    @SKIP_IF_NO_DOCKER
    def test_code_interpreter_safe_code(self):
        """code_interpreter executes safe Python and captures stdout."""
        res = _run_container(
            {"tool": "code_interpreter", "args": {"script": "print(2 + 2)"}}
        )
        assert res["exit_code"] == 0
        parsed = json.loads(res["stdout"])
        assert parsed["success"] is True
        assert "4" in parsed["result"]["stdout"]

    @SKIP_IF_NO_DOCKER
    def test_code_interpreter_blocks_open(self):
        """code_interpreter must not expose the built-in open() function."""
        res = _run_container(
            {"tool": "code_interpreter",
             "args": {"script": "open('/etc/passwd').read()"}}
        )
        parsed = json.loads(res["stdout"])
        # open() is stripped from safe builtins → NameError → tool error
        assert parsed["success"] is False

    @SKIP_IF_NO_DOCKER
    def test_translation_stub(self):
        """translation tool echoes input as stub."""
        res = _run_container(
            {"tool": "translation", "args": {"text": "مرحبا"}}
        )
        parsed = json.loads(res["stdout"])
        assert parsed["success"] is True
        assert parsed["result"]["tool"] == "translation"

    @SKIP_IF_NO_DOCKER
    def test_unknown_tool_rejected(self):
        """Unknown tool names must be rejected with a clear error message."""
        res = _run_container({"tool": "evil_tool", "args": {}})
        parsed = json.loads(res["stdout"])
        assert parsed["success"] is False
        assert "evil_tool" in parsed["error"]

    @SKIP_IF_NO_DOCKER
    def test_missing_args_rejected(self):
        """Empty args dict must produce a validation error, not a crash."""
        res = _run_container({"tool": "web_search", "args": {}})
        parsed = json.loads(res["stdout"])
        assert parsed["success"] is False
        assert "query" in parsed["error"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Isolation verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsolation:

    @SKIP_IF_NO_DOCKER
    def test_no_network_access(self):
        """
        Verify the container cannot reach the internet.
        We try to ping 8.8.8.8; the command should fail (or not exist).
        """
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:size=32m,mode=1777",
            "aman-sandbox:latest",
            "python", "-c",
            "import socket; socket.setdefaulttimeout(3); "
            "s=socket.socket(); s.connect(('8.8.8.8',53)); print('CONNECTED')"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        assert "CONNECTED" not in proc.stdout, (
            "Container was able to reach the internet — network isolation FAILED!"
        )

    @SKIP_IF_NO_DOCKER
    def test_read_only_filesystem(self):
        """
        Verify the root filesystem is read-only.
        Attempting to write outside /tmp must fail.
        """
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:size=32m,mode=1777",
            "aman-sandbox:latest",
            "python", "-c",
            "open('/sandbox/evil.txt','w').write('pwned')"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert proc.returncode != 0, (
            "Container was able to write to the root filesystem — FS isolation FAILED!"
        )

    @SKIP_IF_NO_DOCKER
    def test_tmp_is_writable(self):
        """/tmp should be a writable tmpfs (needed by some tools)."""
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:size=32m,mode=1777",
            "aman-sandbox:latest",
            "python", "-c",
            "open('/tmp/ok.txt','w').write('ok'); print('writable')"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert "writable" in proc.stdout, "/tmp must be writable via tmpfs mount."

    @SKIP_IF_NO_DOCKER
    def test_no_new_privileges_flag(self):
        """
        Verify the container was started with --security-opt no-new-privileges.
        We inspect the container config directly.
        """
        import docker
        client = docker.from_env()
        # Run a container that sleeps briefly so we can inspect it
        container = client.containers.run(
            "aman-sandbox:latest",
            command="sleep 2",
            network_mode="none",
            read_only=True,
            tmpfs={"/tmp": "size=32m,mode=1777"},
            security_opt=["no-new-privileges"],
            detach=True,
            remove=False,
        )
        try:
            info = client.api.inspect_container(container.id)
            sec_opts = info["HostConfig"].get("SecurityOpt") or []
            assert "no-new-privileges" in sec_opts, (
                f"no-new-privileges not found in SecurityOpt: {sec_opts}"
            )
        finally:
            container.stop(timeout=3)
            container.remove()