"""

AMAN-LLM
Cross-cutting component — Audit Logger

PURPOSE
    Provides structured, append-only audit logging for every stage of
    the AMAN-LLM runtime pipeline. Each pipeline decision (PASS, BLOCK,
    ERROR, SOFT, ALLOW) is recorded as one JSON object per line
    (JSONL), so that a request can be reconstructed and inspected
    after the fact.

ARCHITECTURAL ROLE
    The Audit Logger is not one of the enforcement layers. It is used
    by the following components to record their outcomes:

        Input Sanitizer
        Layer 1  — Input-Level Ethical Filtering
        Policy Engine Orchestrator (Layer 2)
        Output Validator Orchestrator (Layer 3)
        Constitutional Self-Critique (Layer 4)
        LLM Agent Core

    Forensic and repudiation-related claims made elsewhere in the
    pipeline depend on the integrity of this log. For that reason the
    logger has its own scope statement (see THREAT MODEL SCOPE below)
    and provides three mechanisms: hash-chained entries, default
    redaction of selected PII patterns, and collision-resistant
    trace identifiers.

INPUTS
    log_event(stage, status, details, ...)
        stage       : name of the pipeline stage emitting the event.
        status      : one of PASS, BLOCK, ERROR, SOFT, ALLOW.
        details     : JSON-serialisable dict with stage-specific data.
        risk_level  : one of low, medium, high, critical.
        optional    : model_name, latency_ms, user_id, trace_id,
                      session_id, redact_pii.
    StageTimer(stage, ...)
        Context manager that measures stage latency and calls
        log_event() on exit.

OUTPUTS
    - One JSON line appended to the file given by the AUDIT_LOG_PATH
      environment variable (default: ./logs/audit.jsonl).
    - log_event() returns the entry dict that was written.
    - verify_chain() returns a dict summarising hash-chain integrity.

IMPLEMENTATION NOTES
    - Each entry contains `prev_hash` and `entry_hash` (SHA-256).
      `entry_hash` covers the canonical JSON of the entry (sorted
      keys, excluding `entry_hash` itself) prefixed by `prev_hash`.
    - Redaction is applied to `details` only (not to user_id,
      session_id, stage or model_name) and is enabled by default.
    - Thread safety is provided by process-local locks. Where
      `fcntl` is available (POSIX), an exclusive file lock also
      serialises concurrent writers in separate processes.
    - The current trace identifier is held in a module-level
      variable (see "Potential implementation issue" notes
      accompanying this file for implications).

REPRODUCIBILITY
    - Set AUDIT_LOG_PATH to select the log location.
    - Call verify_chain() on a log file to check the hash chain
      independently of the process that wrote it.
    - Timestamps are UTC, ISO 8601. Event identifiers are UUID4, so
      two runs of the same experiment will not produce identical
      event_id, timestamp or hash values; compare the other fields.

"""

import json
import uuid
import time
import re
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
import os

# `fcntl` is available on POSIX systems (e.g. Linux, Google Colab) and is
# used for cross-process file locking. When it is unavailable (e.g.
# Windows), the logger relies on process-local locks only.
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    fcntl = None
    _HAS_FCNTL = False



# THREAT MODEL SCOPE NOTE
# (see the accompanying paper, Sec. 2.2 and the STRIDE table in Sec. 7)

#
# This module is the logging substrate for the forensic and
# repudiation-related analysis of the pipeline, so its own security
# properties are stated explicitly.
#
#   IN SCOPE (addressed by this module):
#       * Log injection through `details` values. Entries are
#         serialised with json.dumps(), which escapes newlines and
#         control characters, so a value cannot introduce an
#         additional JSONL record.
#       * Trace-identifier collision. start_trace() generates a full
#         UUID4, which makes accidental collisions negligible at
#         realistic trace volumes. A caller may supply its own
#         trace_id; such a value is trusted to the same degree as the
#         calling session.
#       * Retroactive modification or deletion of past entries. A
#         hash chain (see below) makes such changes detectable by
#         verify_chain(). The mechanism is tamper-EVIDENT, not
#         tamper-PROOF.
#       * Raw PII in the audit trail. Selected patterns are redacted
#         by default (see REDACTION POLICY). This matters for the
#         data-residency discussion in Sec. 7.3 / 8.1 of the paper,
#         because an unredacted log would itself hold sensitive data
#         independently of the Layer 4 critic API.
#
#   OUT OF SCOPE (assumed, not mitigated):
#       * An attacker with filesystem or process access on the host
#         that runs this logger. The log is assumed to be written on
#         infrastructure outside the blast radius of the evaluated
#         pipeline (e.g. a separate log-shipping host).
#       * Tamper prevention (WORM storage, external or RFC 3161
#         timestamping, append-only object-lock storage). An attacker
#         who controls the file can recompute the chain from any
#         point forward, and verify_chain() cannot detect this. Where
#         tamper prevention is required, log lines should be shipped
#         to such storage and this file treated as a local buffer.



# CONSTANTS


# Allowed values for the `status` field of an audit event.
_VALID_STATUSES = frozenset({
    "PASS",
    "BLOCK",
    "ERROR",
    "SOFT",
    "ALLOW",
})

# Allowed values for the `risk_level` field of an audit event.
_VALID_RISK_LEVELS = frozenset({
    "low",
    "medium",
    "high",
    "critical",
})



# REDACTION POLICY

#
# `details` payloads can contain user- or attacker-controlled content.
# For example, the audit record of a blocked PII-exfiltration attempt
# may itself contain the PII that was blocked. Redaction is therefore
# enabled by default. A caller that needs raw values for a specific
# forensic purpose may pass redact_pii=False for that call.
#
# Redaction is pattern-based and heuristic. It reduces the likelihood
# of PII appearing in the log but does not guarantee that all
# sensitive data is removed.

# Key names (case-insensitive substring match) whose values are
# replaced with "[REDACTED]" regardless of their content.
_REDACT_KEY_SUBSTRINGS = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "token",
    "ssn",
    "social_security",
    "credit_card",
    "card_number",
    "cvv",
    "cvc",
    "private_key",
)

# Regular expressions applied to string values irrespective of key
# name. Matches are replaced by "[REDACTED:<label>]".
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,2}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")

# Content patterns in the order in which they are applied.
_CONTENT_PATTERNS = (
    ("email", _EMAIL_RE),
    ("ssn", _SSN_RE),
    ("credit_card", _CC_RE),
    ("phone", _PHONE_RE),
)


def _key_is_sensitive(key: str) -> bool:
    """Return True if `key` contains any entry of _REDACT_KEY_SUBSTRINGS (case-insensitive)."""
    lowered = key.lower()
    return any(s in lowered for s in _REDACT_KEY_SUBSTRINGS)


def _redact_string(value: str) -> tuple[str, bool]:
    """
    Apply every pattern in _CONTENT_PATTERNS to a string.

    Returns (redacted_string, was_anything_redacted).
    """
    redacted = False
    out = value

    for label, pattern in _CONTENT_PATTERNS:
        if pattern.search(out):
            out = pattern.sub(f"[REDACTED:{label}]", out)
            redacted = True

    return out, redacted


def _redact_value(value: Any) -> tuple[Any, bool]:
    """
    Recursively redact a JSON-like value.

    Strings are scanned with the content patterns. Dicts are processed
    key by key: a sensitive key name causes its whole value to be
    replaced, otherwise the value is processed recursively. Lists are
    processed element-wise. Other types are returned unchanged.

    Returns (redacted_value, was_anything_redacted). The input is not
    modified; new containers are built.
    """
    if isinstance(value, str):
        return _redact_string(value)

    if isinstance(value, dict):
        new_dict = {}
        any_redacted = False

        for k, v in value.items():
            if _key_is_sensitive(k):
                new_dict[k] = "[REDACTED]"
                any_redacted = True
                continue

            new_v, was_redacted = _redact_value(v)
            new_dict[k] = new_v
            any_redacted = any_redacted or was_redacted

        return new_dict, any_redacted

    if isinstance(value, list):
        new_list = []
        any_redacted = False

        for item in value:
            new_item, was_redacted = _redact_value(item)
            new_list.append(new_item)
            any_redacted = any_redacted or was_redacted

        return new_list, any_redacted

    return value, False


def redact_details(details: dict) -> tuple[dict, bool]:
    """
    Apply the redaction policy to a `details` payload.

    Returns (redacted_details, was_anything_redacted).
    """

    return _redact_value(details)



# HASH CHAIN / TAMPER-EVIDENCE

#
# Each entry stores the hash of the preceding entry (`prev_hash`) and
# a hash computed over its own canonical content together with that
# prev_hash (`entry_hash`). Retroactive modification or deletion of a
# line breaks the chain from that point onward, which verify_chain()
# can detect. The first entry uses _GENESIS_HASH as its prev_hash.
# This is tamper-EVIDENT, not tamper-PROOF (see the scope note above).

_GENESIS_HASH = "0" * 64
_last_hash: Optional[str] = None  # process-local cache of the latest entry_hash
_hash_lock = threading.Lock()


def _hash_entry(entry: dict, prev_hash: str) -> str:
    """
    Compute the SHA-256 digest of `prev_hash` concatenated with the
    canonical JSON of `entry` (sorted keys, non-ASCII preserved).
    """
    canonical = json.dumps(
        entry,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        (prev_hash + canonical).encode("utf-8")
    ).hexdigest()


def _read_last_entry_hash_tail(path: Path, chunk_size: int = 8192) -> str:
    """
    Return the `entry_hash` of the last record in the log file by
    reading only the end of the file.

    This is the fast-path counterpart of _load_last_hash(), which scans
    the whole file. It is used on the write path so that the cost of a
    write does not grow linearly with log size. The read window starts
    at `chunk_size` bytes and is enlarged (by a factor of four) until
    it contains at least two lines or covers the whole file.

    Returns _GENESIS_HASH if the file is missing or empty. If the tail
    cannot be read or parsed, it falls back to _load_last_hash(), and to
    _GENESIS_HASH if that also fails.
    """
    if not path.exists():
        return _GENESIS_HASH

    try:
        size = path.stat().st_size
        if size == 0:
            return _GENESIS_HASH

        read_size = min(chunk_size, size)
        with open(path, "rb") as f:
            while True:
                f.seek(-read_size, os.SEEK_END)
                data = f.read()
                lines = [l for l in data.split(b"\n") if l.strip()]
                if len(lines) >= 2 or read_size >= size:
                    break
                read_size = min(read_size * 4, size)

        if not lines:
            return _GENESIS_HASH

        last_line = lines[-1].decode("utf-8", errors="strict")
        record = json.loads(last_line)
        return record.get("entry_hash", _GENESIS_HASH)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # The tail could not be read or parsed (e.g. a partially written
        # last line). Use the full-file scan instead of risking a
        # wrong prev_hash.
        try:
            return _load_last_hash(path)
        except Exception:
            return _GENESIS_HASH


def _load_last_hash(path: Path) -> str:
    """
    Return the `entry_hash` of the last valid record by scanning the
    whole log file.

    Used to restore chain continuity after a process restart and as the
    fallback of _read_last_entry_hash_tail(). Blank lines and lines
    that are not valid JSON are skipped. Returns _GENESIS_HASH if the
    file does not exist or cannot be read.
    """

    if not path.exists():
        return _GENESIS_HASH

    last_hash = _GENESIS_HASH

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                    last_hash = record.get("entry_hash", last_hash)
                except json.JSONDecodeError:
                    continue

    except OSError:
        return _GENESIS_HASH

    return last_hash


def verify_chain(path: Optional[Path] = None) -> dict:
    """
    Verify the hash chain of an audit log file.

    Each record is re-hashed from its content (excluding `entry_hash`)
    and the preceding hash, and compared with the stored `prev_hash`
    and `entry_hash`. Verification continues after a mismatch so that
    all records are counted, but only the first failing line is
    reported.

    Args:
        path: Log file to verify. Defaults to the configured log path
            (AUDIT_LOG_PATH or ./logs/audit.jsonl).

    Returns:
        {"ok": bool, "checked": int, "first_break_line": Optional[int]}
        where `checked` is the number of non-blank records examined and
        `first_break_line` is the 1-based line number of the first
        inconsistent record (None if the chain is intact). A missing
        file is reported as ok with zero records checked.

    A passing result shows internal consistency of the file only. It
    does not detect an attacker who rewrote the chain consistently, nor
    truncation of the most recent entries (see the scope note above).
    """

    target_path = path or _get_log_path()
    prev_hash = _GENESIS_HASH
    checked = 0
    first_break_line = None

    if not target_path.exists():
        return {"ok": True, "checked": 0, "first_break_line": None}

    with open(target_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            claimed_prev = record.get("prev_hash")
            claimed_self = record.get("entry_hash")

            # Hash input at write time: all fields except `entry_hash`.
            verifiable = {
                k: v for k, v in record.items()
                if k not in ("entry_hash",)
            }

            recomputed = _hash_entry(verifiable, prev_hash)

            checked += 1

            if claimed_prev != prev_hash or claimed_self != recomputed:
                if first_break_line is None:
                    first_break_line = line_no

            prev_hash = claimed_self or recomputed

    return {
        "ok": first_break_line is None,
        "checked": checked,
        "first_break_line": first_break_line,
    }



# GLOBAL STATE


# Serialises writers within one process. Cross-process ordering is
# handled by fcntl.flock inside log_event() where available.
_write_lock = threading.Lock()

# Lazily initialised log file path and current trace identifier.
_log_path: Optional[Path] = None
_current_trace_id: Optional[str] = None



# LOG PATH


def _get_log_path() -> Path:
    """
    Return the audit log path, resolving it on first use.

    The path is read from the AUDIT_LOG_PATH environment variable
    (default "./logs/audit.jsonl") and cached for the lifetime of the
    process. The parent directory is created if it does not exist.
    """
    global _log_path

    if _log_path is None:

        _log_path = Path(
            os.getenv(
                "AUDIT_LOG_PATH",
                "./logs/audit.jsonl"
            )
        )

        _log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    return _log_path



# TRACE MANAGEMENT


def start_trace(trace_id: Optional[str] = None) -> str:
    """
    Start a new request trace and make it the current trace.

    Events logged afterwards without an explicit `trace_id` are
    associated with this trace, which allows the records produced by
    the different pipeline stages for one request to be correlated.

    Args:
        trace_id: Optional caller-supplied identifier. If omitted, a
            full UUID4 string is generated. The full-length UUID is used
            (rather than a shortened form) so that accidental
            collisions are negligible at realistic trace volumes, which
            the trace-based forensic correlation relies on. A short
            human-readable label, if needed, can be stored in `details`
            instead of shortening the identifier.

    Returns:
        The trace identifier now in effect.
    """

    global _current_trace_id

    _current_trace_id = (
        trace_id
        or str(uuid.uuid4())
    )

    return _current_trace_id


def get_trace_id() -> Optional[str]:
    """Return the current trace identifier, or None if no trace has been started."""
    return _current_trace_id



# MAIN LOGGER


def log_event(
    stage: str,
    status: str,
    details: dict,
    *,
    risk_level: str = "low",
    model_name: Optional[str] = None,
    latency_ms: Optional[float] = None,
    user_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    redact_pii: bool = True,
) -> dict:
    """
    Append one structured, hash-chained audit event to the JSONL log.

    Args:
        stage: Name of the pipeline stage or component emitting the event.
        status: One of PASS, BLOCK, ERROR, SOFT, ALLOW.
        details: Stage-specific payload (must be JSON-serialisable). It
            must not use any of the reserved entry field names.
        risk_level: One of low, medium, high, critical.
        model_name: Optional name of the model involved.
        latency_ms: Optional stage latency in milliseconds (stored
            rounded to three decimals).
        user_id: Optional user identifier (not redacted).
        trace_id: Optional trace identifier. Defaults to the current
            trace set by start_trace().
        session_id: Optional session identifier (not redacted).
        redact_pii: If True (default), `details` is passed through
            redact_details() and the entry is flagged `redacted: true`
            when anything was changed. Pass False only when raw values
            are required and the destination is known to be inside the
            data-residency boundary.

    Returns:
        The entry dict that was written, including `event_id`,
        `timestamp` (UTC, ISO 8601), `prev_hash` and `entry_hash`.

    Raises:
        ValueError: If `status` or `risk_level` is not an allowed value,
            or if `details` contains a reserved key.
    """

   
    # VALIDATION
   

    if status not in _VALID_STATUSES:
        raise ValueError(
            f"status must be one of "
            f"{_VALID_STATUSES}, got {status!r}"
        )

    if risk_level not in _VALID_RISK_LEVELS:
        raise ValueError(
            f"risk_level must be one of "
            f"{_VALID_RISK_LEVELS}, got {risk_level!r}"
        )

   
    # RESERVED KEYS
    # Top-level `details` keys that would collide with entry fields.
   

    _RESERVED = {
        "event_id",
        "timestamp",
        "trace_id",
        "session_id",
        "stage",
        "status",
        "risk_level",
        "user_id",
        "model_name",
        "latency_ms",
        "prev_hash",
        "entry_hash",
        "redacted",
    }

    reserved_clash = _RESERVED & details.keys()

    if reserved_clash:
        raise ValueError(
            f"details contains reserved keys: "
            f"{reserved_clash}"
        )

   
    # REDACTION
   

    was_redacted = False

    if redact_pii:
        details, was_redacted = redact_details(details)

   
    # BUILD ENTRY (mandatory fields)
   

    entry = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "trace_id": (
            trace_id
            or _current_trace_id
        ),

        "stage": stage,
        "status": status,
        "risk_level": risk_level,
    }

   
    # OPTIONAL FIELDS (present only when supplied)
   

    if session_id is not None:
        entry["session_id"] = session_id

    if user_id is not None:
        entry["user_id"] = user_id

    if model_name is not None:
        entry["model_name"] = model_name

    if latency_ms is not None:
        entry["latency_ms"] = round(
            latency_ms,
            3
        )

    if was_redacted:
        entry["redacted"] = True

   
    # PAYLOAD
   

    entry["details"] = details

   
    # HASH CHAIN AND WRITE
    #
    # Reading the previous hash, computing the new hash and writing the
    # line all happen under the same locks, so the chain order always
    # matches the on-disk order and no concurrent writer can compute
    # its hash from a stale prev_hash.
    #
    # Multi-process writers: the in-memory `_last_hash` is valid only
    # within one process. If several processes append to the same file,
    # a cached value would become stale and verify_chain() would report
    # a break although no tampering occurred. Where fcntl is available,
    # an exclusive flock is therefore held across read-last-hash,
    # compute and write, and the previous hash is re-read from the file
    # tail instead of being taken from the cache. Without fcntl (e.g.
    # Windows), only single-process operation is supported.
   

    with _write_lock:

        global _last_hash
        log_path = _get_log_path()

        with open(
            log_path,
            "a",
            encoding="utf-8"
        ) as f:

            if _HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            try:
                with _hash_lock:
                    if _HAS_FCNTL:
                        # With the exclusive lock held, the file is the
                        # authoritative source for the previous hash.
                        _last_hash = _read_last_entry_hash_tail(log_path)
                    elif _last_hash is None:
                        # Without fcntl: restore the chain once per
                        # process, then use the in-memory cache.
                        _last_hash = _load_last_hash(log_path)

                    entry["prev_hash"] = _last_hash
                    entry["entry_hash"] = _hash_entry(entry, _last_hash)
                    _last_hash = entry["entry_hash"]

                line = (
                    json.dumps(
                        entry,
                        ensure_ascii=False
                    )
                    + "\n"
                )
                f.write(line)
                f.flush()
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return entry



# STAGE TIMER


class StageTimer:
    """
    Context manager that times a pipeline stage and logs one audit event.

    On exit it calls log_event() with the measured latency. The status
    defaults to the value given at construction and may be changed
    inside the block with override_status(). If the block raises, the
    status is set to ERROR and the exception message and type are added
    to `details` (without overwriting existing "error" / "error_type"
    keys). The exception is never suppressed.

    Usage:
        with StageTimer("stage_name", risk_level="medium") as t:
            ...
            t.details["key"] = value
            t.override_status("BLOCK")

    Any extra keyword arguments (e.g. user_id, session_id, trace_id,
    model_name, redact_pii) are forwarded to log_event().
    """

    def __init__(
        self,
        stage: str,
        status: str = "PASS",
        risk_level: str = "low",
        **log_kwargs
    ):

        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {status}"
            )

        self.stage = stage
        self._status = status
        self.risk_level = risk_level
        self.log_kwargs = log_kwargs

        # Payload to be logged; the caller may add keys inside the block.
        self.details: dict = {}

        self._start: float = 0.0

   

    def override_status(
        self,
        status: str
    ) -> None:
        """Set the status that will be logged on exit (must be a valid status)."""

        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {status}"
            )

        self._status = status

   

    def __enter__(self):

        self._start = time.perf_counter()

        return self

   

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ) -> bool:

        latency_ms = (
            time.perf_counter()
            - self._start
        ) * 1000

        # -------------------------------------------------
        # Record an exception raised inside the block
        # -------------------------------------------------

        if exc_type is not None:

            self._status = "ERROR"

            self.details.setdefault(
                "error",
                str(exc_val)
            )

            self.details.setdefault(
                "error_type",
                exc_type.__name__
            )

        # -------------------------------------------------
        # Emit the audit event
        # -------------------------------------------------

        log_event(
            stage=self.stage,
            status=self._status,
            details=self.details,
            risk_level=self.risk_level,
            latency_ms=latency_ms,
            **self.log_kwargs,
        )

        # Exceptions raised in the block are always propagated.
        return False
