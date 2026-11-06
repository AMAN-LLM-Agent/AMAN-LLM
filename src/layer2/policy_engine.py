
# AMAN-LLM
# Layer 2 — Policy Engine Orchestrator (Tool-Call Governance)

#
## PURPOSE
#
# This module implements the Layer 2 policy engine described in the AMAN-LLM
# architecture: a sequential gate pipeline that governs tool calls proposed
# by the LLM Agent Core before they are executed. Each candidate tool call is
# evaluated against rate limiting, schema validation, injection detection,
# and configurable policy rules, and the pipeline returns an explicit
# ALLOW/BLOCK decision together with an audit trail.
#
## ARCHITECTURAL ROLE
#
# Layer 2 sits between the LLM Agent Core (which proposes a tool call) and
# the tool execution environment. It receives a PlannedToolCall and is
# responsible for enforcing policy before any tool side effect occurs. It is
# distinct from Layer 1 (input-level ethical filtering, upstream of the
# agent) and Layer 3 (output validation, downstream of tool execution).
# The Audit Logger is a cross-cutting component consumed by this module
# rather than a sequential gate: every decision, block or allow, is emitted
# as a structured JSON record via `_audit()` regardless of which gate ran.
#
## GATE PIPELINE (strict order — the first gate that blocks wins)
#
#   PlannedToolCall
#       │
#       ▼
#   [GATE 0] Rate limiter      PR-08  — token-bucket per context_id
#       ▼
#   [GATE 1] Schema validation — Pydantic models  (schema_validator.py)
#       ▼
#   [GATE 2] Injection scanner — multi-signal heuristic (injection_detector.py)
#       ▼
#   [GATE 3] YAML rules        — PR-01 → PR-13  (policy_rules.yaml)
#       ▼
#   PolicyDecision(decision="ALLOW")
#
## INPUTS
#
#   - PlannedToolCall: the tool name, arguments, context_id, and session
#     role proposed by the LLM Agent Core.
#   - config/policy_rules.yaml: the externally configurable rule set
#     (PR-01 .. PR-13), rejection strategies, and role/tool allowlists.
#     Path discovery is repo-layout-agnostic (see `_YAML_PATH_CANDIDATES`).
#
## OUTPUTS
#
#   - PolicyDecision: the ALLOW/BLOCK verdict, matched rule and violation
#     type (on BLOCK), composite risk score/band, OWASP LLM Top-10 and
#     MITRE ATT&CK mapping, bilingual user-facing messages, and a per-gate
#     decision trace.
#   - One structured JSON audit record per call, written through the
#     module-level `logger` (see `_audit()`).
#   - Optional benchmark output (`BenchmarkReport`) when `run_benchmark()`
#     is used to evaluate the pipeline against a labelled case set.
#
## IMPLEMENTATION NOTES
#
# Beyond the minimal four-gate pipeline, this implementation adds: a
# composite 0-100 risk score with a low/medium/high/critical band; a
# per-gate explainability trace; severity escalation for repeated attacks
# from the same context within a time window; in-memory attack telemetry
# exposed via `get_telemetry()`; policy hot-reload via `enable_hot_reload()`;
# a per-tool safe-pattern allowlist that skips YAML rule evaluation for
# calls matching a known-safe shape (false-positive mitigation only —
# Gates 0-2 still run unconditionally); and bilingual (Arabic/English)
# user-facing response templates. Every YAML rule and heuristic violation
# type is mapped to an OWASP LLM Top-10 category and a MITRE ATT&CK
# technique in `OWASP_MITRE_MAP`, for reference only — this mapping does
# not affect the ALLOW/BLOCK decision itself.
#
## REPRODUCIBILITY
#
# `run_layer2()` is deterministic given a fixed `config/policy_rules.yaml`,
# the rate-limiter and escalation-tracker in-memory state, and the outputs
# of `schema_validator.validate_tool_call()` and
# `injection_detector.detect_injection()`. Rate-limit and escalation state
# are process-local and time-dependent (wall-clock / monotonic clock); call
# `reset_rate_bucket()` / `reset_telemetry()` between independent test runs
# or benchmark cases to avoid cross-case interference. `run_benchmark()`
# reports precision/recall/F1 and latency statistics over a caller-supplied
# labelled case set; it does not itself define or ship a benchmark dataset.
#
## PUBLIC API
#
#   run_layer2(call)              → PolicyDecision   (main entry point)
#   get_telemetry()               → TelemetrySnapshot
#   reset_telemetry()             → None
#   reset_rate_bucket(ctx)        → None
#   reload_policy()               → None
#   enable_hot_reload(interval_s) → None
#   run_benchmark(cases)          → BenchmarkReport
#


from __future__ import annotations

import collections
import json
import logging
import re
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from schema_validator   import validate_tool_call
    from injection_detector import detect_injection
except ImportError:
    from src.layer2.schema_validator import validate_tool_call
    from src.layer2.injection_detector import detect_injection

# ---------------------------------------------------------------------------
# Logging
#
# All decision, audit, and diagnostic records for this module are emitted
# through this logger name, so downstream log processing (including the
# Audit Logger's JSON records; see `_audit()`) can filter on it directly.
# ---------------------------------------------------------------------------
logger = logging.getLogger("aman.layer2.policy_engine")

# ---------------------------------------------------------------------------
# Policy YAML path discovery
#
# `config/policy_rules.yaml` is searched for at several relative locations
# so that the module resolves correctly regardless of the working directory
# it is imported from (repo root, src/, src/layer2/, or a tests/ directory).
# The first existing candidate path is used; see `_find_yaml()`.
# ---------------------------------------------------------------------------
_YAML_PATH_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent / "config" / "policy_rules.yaml",
    Path(__file__).resolve().parent.parent        / "config" / "policy_rules.yaml",
    Path(__file__).resolve().parent               / "policy_rules.yaml",
    Path("config") / "policy_rules.yaml",
    Path("policy_rules.yaml"),
]

# ---------------------------------------------------------------------------
# Security framework mappings
#
# Each violation type recognised by this module is associated with an
# OWASP LLM Top-10 (2025) category and a MITRE ATT&CK for Enterprise
# technique, for reporting and cross-referencing purposes. This mapping is
# descriptive metadata attached to a BLOCK decision (see `PolicyDecision`);
# it does not participate in the detection or blocking logic itself.
#   OWASP LLM Top-10: https://owasp.org/www-project-top-10-for-large-language-model-applications/
#   MITRE ATT&CK for Enterprise: https://attack.mitre.org/
# ---------------------------------------------------------------------------
OWASP_MITRE_MAP: dict[str, dict[str, str]] = {
    "injection":           {"owasp": "LLM01:PromptInjection",       "mitre": "T1059 - Command and Scripting Interpreter"},
    "prompt_leakage":      {"owasp": "LLM01:PromptInjection",       "mitre": "T1530 - Data from Cloud Storage"},
    "permission":          {"owasp": "LLM05:ImproperOutputHandling", "mitre": "T1078 - Valid Accounts"},
    "role_violation":      {"owasp": "LLM05:ImproperOutputHandling", "mitre": "T1078.004 - Cloud Accounts"},
    "path_traversal":      {"owasp": "LLM02:InsecureOutputHandling", "mitre": "T1083 - File and Directory Discovery"},
    "command_injection":   {"owasp": "LLM02:InsecureOutputHandling", "mitre": "T1059.004 - Unix Shell"},
    "privacy_violation":   {"owasp": "LLM06:SensitiveInfoDisclosure","mitre": "T1552 - Unsecured Credentials"},
    "abuse":               {"owasp": "LLM04:ModelDenialOfService",   "mitre": "T1499 - Endpoint Denial of Service"},
    "ssrf_attempt":        {"owasp": "LLM07:SystemPromptLeakage",    "mitre": "T1090 - Proxy"},
    "sensitive_file_access":{"owasp":"LLM06:SensitiveInfoDisclosure","mitre": "T1552.001 - Credentials In Files"},
    "data_exfiltration":   {"owasp": "LLM06:SensitiveInfoDisclosure","mitre": "T1041 - Exfiltration Over C2 Channel"},
    "token_theft":         {"owasp": "LLM06:SensitiveInfoDisclosure","mitre": "T1528 - Steal Application Access Token"},
    "schema_violation":    {"owasp": "LLM04:ModelDenialOfService",   "mitre": "T1499.002 - Service Exhaustion Flood"},
}

# ---------------------------------------------------------------------------
# Risk scoring — base score (0-100) assigned per violation type on a BLOCK
# decision. Severity escalation (`_EscalationTracker`) adds a fixed boost
# to this base value for repeated attacks from the same context within a
# time window; see `_block()`. These values are configurable constants
# reflecting the relative severity ordering used by this implementation;
# their calibration is not documented in this module.
# ---------------------------------------------------------------------------
_BASE_RISK_SCORE: dict[str, int] = {
    "injection":            90,
    "prompt_leakage":       88,
    "command_injection":    95,
    "path_traversal":       85,
    "ssrf_attempt":         92,
    "sensitive_file_access":88,
    "data_exfiltration":    90,
    "token_theft":          95,
    "privacy_violation":    85,
    "role_violation":       60,
    "permission":           55,
    "abuse":                50,
    "schema_violation":     35,
}

def _risk_band(score: int) -> str:
    """
    Map a composite risk score to a qualitative severity band.

    Parameters
    ----------
    score : int
        Composite risk score in the range 0-100 (see `_BASE_RISK_SCORE`
        and `_EscalationTracker.ESCALATION_BOOST`).

    Returns
    -------
    str
        One of "low", "medium", "high", "critical", using fixed score
        thresholds. The bands are a reporting convenience and do not by
        themselves change the ALLOW/BLOCK decision.
    """
    if score >= 80: return "critical"
    if score >= 60: return "high"
    if score >= 35: return "medium"
    return "low"

# ---------------------------------------------------------------------------
# Bilingual (Arabic / English) user-facing response templates.
# `response_ar` is intended for display to the end user; `response_en` is
# intended for logs and debugging. Placeholders (e.g. {allowed_tools}) are
# filled in by `_build_responses()`.
# ---------------------------------------------------------------------------
_RESPONSE_TEMPLATES: dict[str, dict[str, str]] = {
    "permission_ask": {
        "ar": "هذه الأداة تتطلب تسجيل الدخول. يرجى المصادقة أولاً أو استخدام إحدى الأدوات المتاحة: {allowed_tools}.",
        "en": "This tool requires authentication. Please log in or use an available tool: {allowed_tools}.",
    },
    "attack_decline": {
        "ar": "تم رفض الطلب. تم اكتشاف نمط هجوم في المعاملات المُرسَلة.",
        "en": "Request declined. An attack pattern was detected in the submitted arguments.",
    },
    "role_ask": {
        "ar": "هذه الأداة غير متاحة لدورك الحالي. الأدوات المتاحة لك: {allowed_tools}.",
        "en": "This tool is not available for your current role. Your available tools: {allowed_tools}.",
    },
    "schema_ask": {
        "ar": "الحمولة المُرسَلة غير صالحة أو تتجاوز الحجم المسموح به. يرجى تبسيط الطلب أو تقسيمه.",
        "en": "The payload is invalid or exceeds the size limit. Please simplify or split your request.",
    },
    "privacy_decline": {
        "ar": "تم رفض الطلب. تم الكشف عن معلومات شخصية حساسة في المعاملات.",
        "en": "Request declined. Sensitive personal information (PII) was detected.",
    },
    "rate_limit_ask": {
        "ar": "تجاوزت الحد الأقصى لاستدعاءات الأدوات ({max_calls} في {window_seconds} ثانية). يرجى الانتظار.",
        "en": "Rate limit exceeded ({max_calls} calls per {window_seconds}s). Please wait before retrying.",
    },
    "sensitive_file_decline": {
        "ar": "تم رفض الوصول. المسار يستهدف ملف اعتمادات أو أسرار حساسة.",
        "en": "Access declined. The path targets a sensitive credentials or secrets file.",
    },
    "prompt_leakage_decline": {
        "ar": "تم رفض الطلب. تم الكشف عن محاولة لاستخراج تعليمات النظام الداخلية.",
        "en": "Request declined. A system prompt extraction attempt was detected.",
    },
    "data_exfiltration_decline": {
        "ar": "تم رفض الطلب. تم الكشف عن استعلام لاستخراج بيانات بالجملة.",
        "en": "Request declined. A bulk data exfiltration query was detected.",
    },
    "token_theft_decline": {
        "ar": "تم رفض الطلب. تم الكشف عن رمز وصول أو مفتاح API في المعاملات.",
        "en": "Request declined. A Bearer token or API key was detected in the arguments.",
    },
}

# ---------------------------------------------------------------------------
# Safe-pattern allowlist — false-positive mitigation for Gate 3 (YAML rules)
#
# Maps a tool name to a list of compiled regular expressions. If the
# serialised call arguments match any pattern for that tool, Gate 3 (YAML
# rule evaluation) is skipped for that call. Gates 0-2 (rate limiting,
# schema validation, injection detection) always run regardless of an
# allowlist match; the allowlist narrows only the YAML rule set, not the
# earlier detection gates.
# ---------------------------------------------------------------------------
_SAFE_PATTERNS: dict[str, list[re.Pattern]] = {
    # web_search intentionally empty — PR-07/PR-11/PR-13 must still evaluate
    # on all web_search calls regardless of argument shape.
    "database_query": [
        # Parameterised SELECT WHERE field=? LIMIT n — safe form.
        # Does NOT match SELECT * (PR-12), missing LIMIT, or destructive keywords.
        re.compile(
            r"SELECT\s+[\w\s,]+\s+FROM\s+\w+\s+WHERE\s+\w+\s*=\s*\?\s+LIMIT\s+\d+",
            re.IGNORECASE,
        ),
    ],
}



# Data classes


@dataclass
class GateTrace:
    """
    One entry in the per-call decision trace.

    Records which gate executed, its outcome, and the time it took, so
    that a `PolicyDecision` can be inspected after the fact to determine
    exactly what the pipeline evaluated for a given call. Used only for
    explainability and auditing; it does not affect the pipeline outcome.
    """
    gate:       str              # "rate" | "schema" | "injection" | "rule" | "allowlist"
    result:     str              # "pass" | "block" | "skip"
    rule_id:    Optional[str]   = None
    reason:     str              = ""
    latency_ms: float            = 0.0


@dataclass
class PlannedToolCall:
    """
    A tool call proposed by the LLM Agent Core, prior to execution.

    This is the sole input to `run_layer2()`. `arguments` is the raw,
    schema-unvalidated argument mapping as proposed by the agent;
    validation against the tool's expected schema happens in Gate 1
    (see `schema_validator.validate_tool_call`).
    """
    tool_name:    str
    arguments:    dict[str, Any]
    context_id:   str
    session_role: str = "anonymous"


@dataclass
class PolicyDecision:
    """
    Full result of running the Layer 2 pipeline on one PlannedToolCall.

    Returned by `run_layer2()`. Also written to the audit log and used to
    update in-memory telemetry (see `_audit()` and `_Telemetry.record()`).

    Core fields (always populated)
    --------------------------------
    decision        "ALLOW" | "BLOCK"
    audit_id        UUID — cross-reference with JSON audit log
    gate            which gate blocked (or "passed")
    latency_ms      total pipeline latency in milliseconds

    On BLOCK
    --------
    rule_id         matched rule e.g. "PR-03"
    violation_type  e.g. "injection"
    reason          human-readable explanation
    strategy        "decline" | "ask"
    risk_score      0–100 composite score
    risk_band       "low" | "medium" | "high" | "critical"
    owasp_id        OWASP LLM Top-10 category
    mitre_id        MITRE ATT&CK technique
    response_ar     Arabic message for the end user
    response_en     English message for logs / debug
    escalated       True if severity was upgraded (repeated attack)
    scrubbed        True if arguments were redacted (PR-13)
    suggest_tools   tools available to this role (for "ask" strategy)
    signals         injection signals list (Gate 2)
    trace           per-gate explainability log
    """
    decision:        str
    audit_id:        str              = field(default_factory=lambda: str(uuid.uuid4()))
    gate:            str              = ""
    latency_ms:      float            = 0.0
    rule_id:         Optional[str]    = None
    violation_type:  Optional[str]    = None
    reason:          str              = ""
    strategy:        str              = ""
    risk_score:      int              = 0
    risk_band:       str              = "low"
    owasp_id:        str              = ""
    mitre_id:        str              = ""
    response_ar:     str              = ""
    response_en:     str              = ""
    escalated:       bool             = False
    scrubbed:        bool             = False
    suggest_tools:   list[str]        = field(default_factory=list)
    signals:         list[dict]       = field(default_factory=list)
    trace:           list[GateTrace]  = field(default_factory=list)
    policy_version:  str              = ""



# Telemetry


@dataclass
class TelemetrySnapshot:
    """
    Point-in-time aggregate statistics over calls processed by `run_layer2()`.

    Exposed via `get_telemetry()` for monitoring and reporting. `precision`,
    `recall`, and `f1` are populated only when at least one call has been
    scored against a known label through `run_benchmark()` /
    `_Telemetry.record_labelled()`; otherwise they are `None`.
    """
    total_calls:       int
    total_blocks:      int
    total_allows:      int
    block_rate:        float               # 0.0 – 1.0
    by_violation:      dict[str, int]      # violation_type → count
    by_rule:           dict[str, int]      # rule_id → count
    by_risk_band:      dict[str, int]      # "low"/"medium"/"high"/"critical" → count
    escalations:       int
    avg_latency_ms:    float
    p95_latency_ms:    float
    precision:         Optional[float]     # requires labelled evaluation calls
    recall:            Optional[float]
    f1:                Optional[float]


class _Telemetry:
    """
    Thread-safe, in-memory telemetry store for the Layer 2 pipeline.

    Accumulates counts and latencies across calls to `run_layer2()` for the
    lifetime of the process. State is not persisted; it is reset by
    `reset()` (e.g. between benchmark runs) and read via `snapshot()`.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with threading.Lock():
            self._total       = 0
            self._blocks      = 0
            self._allows      = 0
            self._by_vtype:   dict[str, int]   = collections.defaultdict(int)
            self._by_rule:    dict[str, int]   = collections.defaultdict(int)
            self._by_band:    dict[str, int]   = collections.defaultdict(int)
            self._escalations = 0
            self._latencies:  list[float]      = []
            # For evaluation metrics (labelled calls)
            self._tp = self._fp = self._tn = self._fn = 0

    def record(self, decision: PolicyDecision):
        """Update aggregate counters and latency history with one decision."""
        with self._lock:
            self._total += 1
            self._latencies.append(decision.latency_ms)
            if decision.decision == "BLOCK":
                self._blocks += 1
                if decision.violation_type:
                    self._by_vtype[decision.violation_type] += 1
                if decision.rule_id:
                    self._by_rule[decision.rule_id] += 1
                self._by_band[decision.risk_band] += 1
                if decision.escalated:
                    self._escalations += 1
            else:
                self._allows += 1

    def record_labelled(self, decision: PolicyDecision, expected_block: bool):
        """Record label counters only — run_layer2 already called record()."""
        with self._lock:
            actual_block = decision.decision == "BLOCK"
            if actual_block and expected_block:     self._tp += 1
            elif actual_block and not expected_block: self._fp += 1
            elif not actual_block and not expected_block: self._tn += 1
            else:                                   self._fn += 1

    def snapshot(self) -> TelemetrySnapshot:
        """
        Compute and return a `TelemetrySnapshot` from the current counters.

        p95 latency is computed via nearest-rank on the sorted latency
        sample; precision/recall/F1 are computed from labelled-call
        counters and are `None` until at least one labelled call has been
        recorded (see `record_labelled`).
        """
        with self._lock:
            lats = self._latencies or [0.0]
            avg  = statistics.mean(lats)
            p95  = sorted(lats)[int(len(lats) * 0.95)] if len(lats) > 1 else lats[0]
            total_eval = self._tp + self._fp + self._tn + self._fn
            if total_eval > 0:
                prec = self._tp / (self._tp + self._fp) if (self._tp + self._fp) else None
                rec  = self._tp / (self._tp + self._fn) if (self._tp + self._fn) else None
                f1   = (2 * prec * rec / (prec + rec)
                        if prec is not None and rec is not None and (prec + rec) > 0
                        else None)
            else:
                prec = rec = f1 = None

            return TelemetrySnapshot(
                total_calls    = self._total,
                total_blocks   = self._blocks,
                total_allows   = self._allows,
                block_rate     = self._blocks / self._total if self._total else 0.0,
                by_violation   = dict(self._by_vtype),
                by_rule        = dict(self._by_rule),
                by_risk_band   = dict(self._by_band),
                escalations    = self._escalations,
                avg_latency_ms = round(avg, 3),
                p95_latency_ms = round(p95, 3),
                precision      = round(prec, 4) if prec is not None else None,
                recall         = round(rec, 4)  if rec  is not None else None,
                f1             = round(f1, 4)   if f1   is not None else None,
            )


_telemetry = _Telemetry()



# Severity escalation tracker


class _EscalationTracker:
    """
    Tracks repeated policy violations from the same context.

    Maintains, per (context_id, violation_type) pair, a sliding-window
    history of violation timestamps. When the number of violations within
    the window reaches ESCALATION_THRESHOLD, subsequent BLOCK decisions for
    that pair are treated as escalated and receive a risk-score boost (see
    `_block()`). This implements a simple repeated-attack heuristic; it
    does not distinguish between related and unrelated repeated calls
    beyond the (context_id, violation_type) grouping.
    """
    ESCALATION_THRESHOLD = 3      # Number of violations within the window that triggers escalation.
    ESCALATION_WINDOW_S  = 300    # Sliding window length, in seconds (5 minutes).
    ESCALATION_BOOST     = 10     # Risk-score points added to an escalated decision.

    def __init__(self):
        self._lock    = threading.Lock()
        # context_id → {violation_type → deque[timestamp]}
        self._history: dict[str, dict[str, collections.deque]] = (
            collections.defaultdict(lambda: collections.defaultdict(collections.deque))
        )

    def record_and_check(self, context_id: str, violation_type: str) -> bool:
        """
        Record one violation and report whether escalation applies.

        Parameters
        ----------
        context_id : str
            Identifier of the calling session/context (from PlannedToolCall).
        violation_type : str
            The violation type of the current BLOCK decision.

        Returns
        -------
        bool
            True if the number of violations for this (context_id,
            violation_type) pair within ESCALATION_WINDOW_S, including the
            one just recorded, has reached ESCALATION_THRESHOLD.

        Notes
        -----
        Timestamps older than the window are evicted before the count is
        checked, using the monotonic clock (`time.monotonic()`), which is
        unaffected by system clock adjustments.
        """
        with self._lock:
            now    = time.monotonic()
            cutoff = now - self.ESCALATION_WINDOW_S
            bucket = self._history[context_id][violation_type]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            bucket.append(now)
            return len(bucket) >= self.ESCALATION_THRESHOLD


_escalation = _EscalationTracker()



# Policy YAML loading and hot-reload
#
# The parsed policy (PR-01 .. PR-13 rules, rejection strategies, and
# role/tool allowlists) is cached in-process after first load and reused
# across calls to `run_layer2()`. `reload_policy()` and
# `enable_hot_reload()` allow the cache to be refreshed at runtime without
# restarting the process.


_POLICY_CACHE:    dict | None      = None
_POLICY_LOCK:     threading.Lock   = threading.Lock()
_HOT_RELOAD_THREAD: threading.Thread | None = None


def _find_yaml() -> Path:
    """
    Locate `policy_rules.yaml` using `_YAML_PATH_CANDIDATES`.

    Returns
    -------
    Path
        The first candidate path that exists on disk.

    Raises
    ------
    FileNotFoundError
        If none of the candidate paths exist.
    """
    for p in _YAML_PATH_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "policy_rules.yaml not found. Searched:\n"
        + "\n".join(f"  {p}" for p in _YAML_PATH_CANDIDATES)
    )


def _load_policy() -> dict:
    """
    Return the parsed policy dict, loading and caching it on first use.

    Subsequent calls return the cached dict without re-reading the YAML
    file, until `reload_policy()` clears the cache. Thread-safe via
    `_POLICY_LOCK`.
    """
    global _POLICY_CACHE
    with _POLICY_LOCK:
        if _POLICY_CACHE is not None:
            return _POLICY_CACHE
        path = _find_yaml()
        with open(path, encoding="utf-8") as fh:
            _POLICY_CACHE = yaml.safe_load(fh)
        ver = _POLICY_CACHE.get("schema_version", "unversioned")
        logger.info("Policy loaded  path=%s  version=%s", path, ver)
        return _POLICY_CACHE


def reload_policy() -> None:
    """
    Clear the cached policy and reload it from disk immediately.

    Intended to be safe to call while the pipeline is in use; the cache is
    cleared and repopulated under `_POLICY_LOCK`.
    """
    global _POLICY_CACHE
    with _POLICY_LOCK:
        _POLICY_CACHE = None
    _load_policy()
    logger.info("Policy hot-reloaded")


def enable_hot_reload(interval_s: float = 30.0) -> None:
    """
    Start a background daemon thread that watches the policy file for changes.

    The thread polls the resolved policy file's modification time every
    `interval_s` seconds and calls `reload_policy()` when it changes.
    Errors during a poll are logged and do not stop the watcher thread.

    Parameters
    ----------
    interval_s : float
        Polling interval, in seconds. Default 30.0.

    Notes
    -----
    Idempotent with respect to thread creation: if a watcher thread is
    already running, calling this again is a no-op and does not change the
    interval of the existing thread.
    """
    global _HOT_RELOAD_THREAD

    def _watcher():
        yaml_path   = _find_yaml()
        last_mtime  = yaml_path.stat().st_mtime
        while True:
            time.sleep(interval_s)
            try:
                mtime = yaml_path.stat().st_mtime
                if mtime != last_mtime:
                    reload_policy()
                    last_mtime = mtime
            except Exception as exc:
                logger.warning("Hot-reload check failed: %s", exc)

    if _HOT_RELOAD_THREAD and _HOT_RELOAD_THREAD.is_alive():
        return
    _HOT_RELOAD_THREAD = threading.Thread(target=_watcher, daemon=True,
                                           name="aman-policy-hot-reload")
    _HOT_RELOAD_THREAD.start()
    logger.info("Hot-reload enabled  interval=%.0fs", interval_s)



# Rate limiter — Gate 0 (PR-08)
#
# Implements a sliding-window call counter per context_id, using the
# PR-08 rule's `rate_limit` configuration (max_calls, window_seconds) from
# the loaded policy YAML. This is the first gate evaluated by
# `run_layer2()`, ahead of schema and content-based checks, so that an
# excessive call volume from a given context is rejected before the more
# expensive validation and detection gates run.


_rate_buckets: dict[str, collections.deque] = collections.defaultdict(collections.deque)


def _check_rate_limit(context_id: str, policy: dict) -> tuple[bool, str, dict]:
    """
    Check and update the sliding-window call count for one context.

    Parameters
    ----------
    context_id : str
        Identifier of the calling session/context.
    policy : dict
        The loaded policy dict; the PR-08 rule's `rate_limit` sub-config
        supplies `max_calls` and `window_seconds` (defaults: 20 / 60s).

    Returns
    -------
    tuple[bool, str, dict]
        `(exceeded, reason, rate_params)`, where `exceeded` is True if the
        call would exceed the configured limit, `reason` is a
        human-readable explanation (empty if not exceeded), and
        `rate_params` echoes the resolved `max_calls` / `window_seconds`
        for use in user-facing messages.

    Notes
    -----
    If the limit is not exceeded, the current call is recorded in the
    bucket as a side effect of this check.
    """
    pr08 = next(
        (r for r in policy.get("layer2_rules", []) if r.get("id") == "PR-08"),
        None,
    )
    rate_cfg    = (pr08 or {}).get("rate_limit", {})
    window_secs = int(rate_cfg.get("window_seconds", 60))
    max_calls   = int(rate_cfg.get("max_calls", 20))

    now    = time.monotonic()
    bucket = _rate_buckets[context_id]
    while bucket and (now - bucket[0]) > window_secs:
        bucket.popleft()

    if len(bucket) >= max_calls:
        return (
            True,
            f"Rate limit exceeded: {len(bucket)} calls in last {window_secs}s (max {max_calls})",
            {"max_calls": max_calls, "window_seconds": window_secs},
        )
    bucket.append(now)
    return False, "", {"max_calls": max_calls, "window_seconds": window_secs}


def reset_rate_bucket(context_id: str) -> None:
    """Clear a session's rate bucket (used in tests and benchmarks)."""
    _rate_buckets[context_id] = collections.deque()



# Credential scrubber — PR-13
#
# Detects and redacts common credential formats (bearer/authorization
# headers, JWTs, provider-prefixed API keys such as sk-/ghp_/AKIA, and
# generic password/api_key/secret assignments) from tool-call arguments
# before they are written to the structured audit log. This is a
# pattern-based redaction mechanism: it targets recognised credential
# formats and is not a general-purpose secret scanner.


_SCRUB_PLACEHOLDER = "<REDACTED-CREDENTIAL>"
_CREDENTIAL_RE = re.compile(
    r"(?i)("
    r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"
    r"|Authorization:\s*(?:Basic|Bearer|Token)\s+\S+"
    r"|\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*"
    r"|\bsk-[A-Za-z0-9]{20,}"
    r"|\bpk-[A-Za-z0-9]{20,}"
    r"|\bghp_[A-Za-z0-9]{36}"
    r"|\bgho_[A-Za-z0-9]{36}"
    r"|\bghs_[A-Za-z0-9]{36}"
    r"|\bglpat-[A-Za-z0-9_\-]{20,}"
    r"|\bxoxb-[0-9A-Za-z]+(?:-[0-9A-Za-z]+)+"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bAIza[0-9A-Za-z\-_]{35}"
    r"|password\s*[=:]+\s*\S{8,}"
    r"|api[_\-]?key\s*[=:]+\s*\S{8,}"
    r"|secret[_\-]?key\s*[=:]+\s*\S{8,}"
    r"|private[_\-]?token\s*[=:]+\s*\S{8,}"
    r")"
)


def _scrub(text: str) -> str:
    """Replace every credential-pattern match in `text` with the placeholder."""
    return _CREDENTIAL_RE.sub(_SCRUB_PLACEHOLDER, text)


def _scrub_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """
    Return a credential-redacted copy of a tool call's arguments.

    Serialises `arguments` to JSON, applies `_scrub()`, and re-parses the
    result. If redaction leaves the text malformed as JSON (for example,
    a placeholder substituted inside a quoted string boundary), a minimal
    placeholder dict is returned instead of the malformed structure, so
    that a redaction failure cannot itself leak the original arguments.

    Returns
    -------
    tuple[dict[str, Any], bool]
        `(safe_arguments, changed)`, where `changed` is True if any
        redaction was applied.
    """
    serialised = json.dumps(arguments, ensure_ascii=False)
    redacted   = _scrub(serialised)
    changed    = redacted != serialised
    try:
        return json.loads(redacted), changed
    except json.JSONDecodeError:
        return {"__scrubbed__": True}, True



# Bulk-export heuristics — PR-12 ('bulk_column_export' and
# 'unlimited_select_star' DSL verbs)
#
# These two functions implement dedicated Python-side checks for the PR-12
# rule's DSL verbs, invoked from `_matches_rule()`. They operate on the
# raw SQL text of a `database_query` tool call's `query` argument and flag
# query shapes consistent with an unscoped bulk data export, as opposed to
# a bounded, single-record lookup.


_BULK_SELECT_RE = re.compile(r"(?is)^\s*SELECT\s+(.+?)\s+FROM\s+\S+")
_BULK_WHERE_RE  = re.compile(r"(?i)\bWHERE\b")
_BULK_LIMIT_RE  = re.compile(r"(?i)\bLIMIT\s+(\d+)")
_BULK_LIMIT_MAX = 100  # a LIMIT above this is treated as effectively unbounded

_SELECT_STAR_RE = re.compile(r"(?i)SELECT\s+\*\s+FROM")
_ANY_LIMIT_RE   = re.compile(r"(?i)\bLIMIT\s+\d+\b")


def _check_unlimited_select_star(query: str) -> bool:
    """
    Flag 'SELECT * FROM ...' as an unbounded bulk export UNLESS the query is
    bounded by a LIMIT clause *anywhere* in the statement. Checking the whole
    string (rather than only the text after the SELECT * match) avoids a
    false positive on CTEs where an inner subquery's LIMIT already bounds the
    row set the outer 'SELECT * FROM <cte>' draws from, e.g.:
        WITH t AS (SELECT name FROM cities LIMIT 10) SELECT * FROM t
    """
    if not _SELECT_STAR_RE.search(query):
        return False
    return not _ANY_LIMIT_RE.search(query)


def _check_bulk_export(query: str) -> bool:
    """
    Flag a database query as an unscoped bulk export when it:
      • projects 2+ explicit columns (comma-separated, not SELECT *), AND
      • has no WHERE clause narrowing the result set, AND
      • either omits LIMIT entirely or requests a LIMIT above _BULK_LIMIT_MAX.

    A single-column projection, an aggregate (COUNT(*) etc. — no comma), or
    any query scoped by a WHERE clause is left alone; those are the normal
    shape of a legitimate, bounded lookup.
    """
    m = _BULK_SELECT_RE.match(query)
    if not m:
        return False
    columns = m.group(1)
    if "," not in columns:
        return False
    if _BULK_WHERE_RE.search(query):
        return False
    limit_m = _BULK_LIMIT_RE.search(query)
    if limit_m:
        return int(limit_m.group(1)) > _BULK_LIMIT_MAX
    return True



# YAML rule condition evaluator (Gate 3)
#
# `_matches_rule()` interprets the small condition DSL used by
# `config/policy_rules.yaml` (PR-01 .. PR-13), matching each condition
# string against fixed patterns and dispatching to the corresponding
# check. This keeps the rule file declarative while allowing conditions to
# reference call arguments, session role, and role/tool allowlists.


def _serialise(arguments: dict[str, Any]) -> str:
    """Serialise tool-call arguments to a JSON string for text-based rule checks."""
    return json.dumps(arguments, ensure_ascii=False)


def _get_field(args_str: str, fname: str) -> str:
    """
    Extract one field's value from serialised arguments as a string.

    Returns an empty string if the field is absent, `null`, or if
    `args_str` is not valid JSON, so that callers can treat a missing or
    malformed field as simply not matching a condition rather than raising.
    """
    try:
        val = json.loads(args_str).get(fname, "")
        return str(val) if val is not None else ""
    except (json.JSONDecodeError, AttributeError):
        return ""


def _matches_rule(
    rule:    dict,
    call:    PlannedToolCall,
    policy:  dict,
    args_str: str,
) -> tuple[bool, str]:
    """
    Evaluate one YAML rule's `condition` string against a tool call.

    Parameters
    ----------
    rule : dict
        One rule entry from `policy["layer2_rules"]`, as loaded from
        `config/policy_rules.yaml`.
    call : PlannedToolCall
        The tool call under evaluation.
    policy : dict
        The full loaded policy dict (used to resolve role/tool allowlists).
    args_str : str
        `call.arguments` pre-serialised to JSON (see `_serialise`), passed
        in to avoid re-serialising per rule.

    Returns
    -------
    tuple[bool, str]
        `(matched, evidence_snippet)`. `evidence_snippet` is a short,
        human-readable excerpt supporting a match; it is empty when the
        rule does not match.

    Notes
    -----
    Recognised DSL verbs:
      session_role == 'X'
      tool_name not in role_allowed_tools[session_role]
      arguments.FIELD contains 'LITERAL'
      arguments.FIELD matches 'REGEX'
      arguments         matches 'REGEX'   (full JSON payload)
      len(arguments)    > N               (UTF-8 bytes, full payload)
      len(arguments.FIELD) > N            (UTF-8 bytes, one field)
      arguments.FIELD   > N               (numeric field comparison)
      bulk_column_export                  (PR-12 heuristic; see `_check_bulk_export`)
      unlimited_select_star               (PR-12 heuristic; see `_check_unlimited_select_star`)
      tool_calls_per_minute > N           (delegated to Gate 0; always returns no match here)

    A condition string that does not match any recognised verb is logged
    as unrecognised and treated as not matched (fails open for that single
    rule, not for the pipeline as a whole).
    """
    cond = rule.get("condition", "").strip()
    if not cond:
        return False, ""

    # PR-08 rate-limit sub-rule delegated to Gate 0
    if "tool_calls_per_minute" in cond:
        return False, ""

    # bulk_column_export — dedicated Python heuristic (PR-12)
    if cond == "bulk_column_export":
        query = str(call.arguments.get("query", ""))
        hit = _check_bulk_export(query)
        return hit, (f"unscoped multi-column export: {query[:100]}" if hit else "")

    # unlimited_select_star — dedicated Python heuristic (PR-12)
    if cond == "unlimited_select_star":
        query = str(call.arguments.get("query", ""))
        hit = _check_unlimited_select_star(query)
        return hit, (f"SELECT * with no bounding LIMIT anywhere: {query[:100]}" if hit else "")

    # session_role == 'X'
    m = re.fullmatch(r"session_role\s*==\s*['\"](\w+)['\"]", cond)
    if m:
        match = call.session_role == m.group(1)
        return match, (f"session_role='{call.session_role}'" if match else "")

    # tool_name not in role_allowed_tools[session_role]
    if "tool_name not in role_allowed_tools" in cond:
        allowed = policy.get("role_allowed_tools", {}).get(call.session_role, [])
        match   = call.tool_name not in allowed
        return match, (f"tool '{call.tool_name}' not in {allowed}" if match else "")

    # len(arguments.FIELD) > N   (byte length of one field, not the whole payload)
    m = re.fullmatch(r"len\(arguments\.(\w+)\)\s*>\s*(\d+)", cond)
    if m:
        fname, limit = m.group(1), int(m.group(2))
        fval  = call.arguments.get(fname)
        size  = len(str(fval).encode("utf-8")) if fval is not None else 0
        match = fval is not None and size > limit
        return match, (f"{fname}: {size} bytes > limit {limit}" if match else "")

    # arguments.FIELD > N   (numeric field comparison)
    m = re.fullmatch(r"arguments\.(\w+)\s*>\s*(\d+)", cond)
    if m:
        fname, limit = m.group(1), int(m.group(2))
        fval = call.arguments.get(fname)
        try:
            match = fval is not None and float(fval) > limit
        except (TypeError, ValueError):
            match = False
        return match, (f"{fname}={fval} > limit {limit}" if match else "")

    # len(arguments) > N
    m = re.fullmatch(r"len\(arguments\)\s*>\s*(\d+)", cond)
    if m:
        size  = len(args_str.encode("utf-8"))
        limit = int(m.group(1))
        match = size > limit
        return match, (f"{size} bytes > limit {limit}" if match else "")

    # arguments.FIELD contains 'LITERAL'
    m = re.fullmatch(r"arguments\.(\w+)\s+contains\s+'([^']+)'", cond)
    if m:
        fname, literal = m.group(1), m.group(2)
        fval  = _get_field(args_str, fname)
        match = literal in fval
        if match:
            i = fval.index(literal)
            return True, fval[max(0, i-10): i+len(literal)+10]
        return False, ""

    # arguments.FIELD matches 'REGEX'
    m = re.fullmatch(r"arguments\.(\w+)\s+matches\s+'(.+)'", cond, re.DOTALL)
    if m:
        fname, pat = m.group(1), m.group(2)
        fval = _get_field(args_str, fname)
        try:
            hit = re.search(pat, fval, re.IGNORECASE | re.UNICODE | re.DOTALL)
        except re.error as e:
            logger.warning("Bad regex in rule %s: %s", rule.get("id"), e)
            return False, ""
        return (True, hit.group(0)[:120]) if hit else (False, "")

    # arguments matches 'REGEX'  (full payload)
    m = re.fullmatch(r"arguments\s+matches\s+'(.+)'", cond, re.DOTALL)
    if m:
        try:
            hit = re.search(m.group(1), args_str, re.IGNORECASE | re.UNICODE | re.DOTALL)
        except re.error as e:
            logger.warning("Bad regex in rule %s: %s", rule.get("id"), e)
            return False, ""
        return (True, hit.group(0)[:120]) if hit else (False, "")

    logger.warning("Unrecognised DSL in rule %s: '%s'", rule.get("id", "?"), cond)
    return False, ""



# User-facing response builder


def _build_responses(
    violation_type: str,
    strategy_cfg:   dict,
    rate_params:    dict,
    allowed_tools:  list[str],
) -> tuple[str, str]:
    """
    Build bilingual user-facing messages for a BLOCK decision.

    Parameters
    ----------
    violation_type : str
        The violation type of the current BLOCK decision (used only for
        logging context here; the template is selected via `strategy_cfg`).
    strategy_cfg : dict
        The rejection-strategy config for this violation type, as defined
        in the policy YAML; its `response_key` selects the template from
        `_RESPONSE_TEMPLATES` (defaults to "attack_decline").
    rate_params : dict
        Resolved `max_calls` / `window_seconds`, used to fill the
        rate-limit template's placeholders.
    allowed_tools : list[str]
        Tools available to the caller's role, used to fill the
        `{allowed_tools}` placeholder on "ask"-style templates.

    Returns
    -------
    tuple[str, str]
        `(response_ar, response_en)`, the formatted Arabic and English
        messages.
    """
    rkey = strategy_cfg.get("response_key", "attack_decline")
    tpl  = _RESPONSE_TEMPLATES.get(rkey, {})

    def _fmt(text: str) -> str:
        return text.format(
            allowed_tools  = ", ".join(allowed_tools) if allowed_tools else "—",
            max_calls      = rate_params.get("max_calls", 20),
            window_seconds = rate_params.get("window_seconds", 60),
        )

    return _fmt(tpl.get("ar", "تم رفض الطلب.")), _fmt(tpl.get("en", "Request declined."))



# Audit Logger integration — structured JSON audit record
#
# `_audit()` is the point of integration with the AMAN-LLM Audit Logger, a
# cross-cutting component rather than a sequential gate: it is invoked for
# every decision reached by `run_layer2()`, both ALLOW and BLOCK, so that
# the audit trail is complete regardless of which gate (if any) blocked
# the call. Records are observational only — emitting an audit record has
# no effect on the ALLOW/BLOCK decision it describes.


def _audit(
    call:     PlannedToolCall,
    decision: PolicyDecision,
    safe_args: dict | None = None,
) -> None:
    """
    Emit one structured JSON audit record via the module logger.

    Parameters
    ----------
    call : PlannedToolCall
        The tool call that was evaluated.
    decision : PolicyDecision
        The resulting decision, including its per-gate trace.
    safe_args : dict | None
        Credential-redacted arguments to record instead of `call.arguments`
        (set when PR-13 has scrubbed the payload; see `_scrub_arguments`).
        When None, the original `call.arguments` are recorded as-is.

    Notes
    -----
    The record is written as a single JSON object via `logger.info`,
    prefixed with "AUDIT ", so it can be extracted from general log output
    by downstream log processing.
    """
    record = {
        "audit_id":       decision.audit_id,
        "ts":             time.time(),
        "policy_version": decision.policy_version,
        "decision":       decision.decision,
        "gate":           decision.gate,
        "tool":           call.tool_name,
        "session_role":   call.session_role,
        "context_id":     call.context_id,
        "rule_id":        decision.rule_id,
        "violation_type": decision.violation_type,
        "risk_score":     decision.risk_score,
        "risk_band":      decision.risk_band,
        "owasp_id":       decision.owasp_id,
        "mitre_id":       decision.mitre_id,
        "strategy":       decision.strategy,
        "escalated":      decision.escalated,
        "scrubbed":       decision.scrubbed,
        "latency_ms":     round(decision.latency_ms, 3),
        "reason":         decision.reason,
        "trace":          [
            {"gate": t.gate, "result": t.result, "rule_id": t.rule_id,
             "reason": t.reason, "latency_ms": round(t.latency_ms, 3)}
            for t in decision.trace
        ],
        "arguments":      safe_args if safe_args is not None else call.arguments,
    }
    logger.info("AUDIT %s", json.dumps(record, ensure_ascii=False))



# Main pipeline


def run_layer2(call: PlannedToolCall) -> PolicyDecision:
    """
    Run the Layer 2 gate pipeline on one proposed tool call.

    Evaluates, in order, the rate limiter (Gate 0), schema validation
    (Gate 1), injection detection (Gate 2), and YAML policy rules
    (Gate 3, subject to the safe-pattern allowlist). The first gate that
    blocks determines the returned decision; if every gate passes, the
    call is ALLOWed. Every decision (ALLOW or BLOCK) is recorded to the
    structured audit log and to in-memory telemetry before being returned.

    Parameters
    ----------
    call : PlannedToolCall
        The tool call proposed by the LLM Agent Core.

    Returns
    -------
    PolicyDecision
        The pipeline's verdict, including the per-gate trace. This
        function does not raise for policy violations; exceptions
        propagate only for conditions outside normal gate evaluation
        (e.g. the policy YAML file being missing on first load).
    """
    t_start  = time.perf_counter()
    policy   = _load_policy()
    audit_id = str(uuid.uuid4())
    trace:   list[GateTrace] = []
    policy_version = policy.get("schema_version", "unversioned")

    strategies   = policy.get("rejection_strategies", {})
    allowed_map  = policy.get("role_allowed_tools", {})
    allowed_tools = allowed_map.get(call.session_role, [])
    args_str     = _serialise(call.arguments)

    # ── Shared helper: assemble, log, audit, and return a BLOCK decision ──
    #
    # Defined as a closure over this call's `trace`, `strategies`,
    # `allowed_tools`, `audit_id`, and `t_start` so that every gate can
    # produce a fully-populated PolicyDecision (risk score/band, OWASP/
    # MITRE mapping, bilingual responses, audit record, telemetry update)
    # through a single code path, rather than duplicating that assembly
    # logic at each gate's block site.
    def _block(
        gate:           str,
        rule_id:        str,
        violation_type: str,
        reason:         str,
        evidence:       str         = "",
        signals:        list[dict]  = None,
        safe_args:      dict | None = None,
        rate_params:    dict        = None,
    ) -> PolicyDecision:
        rate_params = rate_params or {}
        signals     = signals or []
        cfg         = strategies.get(violation_type, {})
        strategy    = cfg.get("strategy", "decline")

        # Risk scoring + escalation
        base_score = _BASE_RISK_SCORE.get(violation_type, 40)
        escalated  = _escalation.record_and_check(call.context_id, violation_type)
        score      = min(100, base_score + (_escalation.ESCALATION_BOOST if escalated else 0))
        band       = _risk_band(score)

        # Framework mapping
        fw = OWASP_MITRE_MAP.get(violation_type, {})

        # Suggest tools for "ask" strategies
        suggest: list[str] = []
        if cfg.get("suggest_alternatives"):
            suggest = allowed_tools

        # Bilingual responses
        ar, en = _build_responses(violation_type, cfg, rate_params, suggest or allowed_tools)

        # Gate trace entry
        trace.append(GateTrace(
            gate=gate, result="block", rule_id=rule_id,
            reason=reason[:200],
            latency_ms=round((time.perf_counter() - t_start) * 1000, 3),
        ))

        d = PolicyDecision(
            decision        = "BLOCK",
            audit_id        = audit_id,
            gate            = gate,
            latency_ms      = round((time.perf_counter() - t_start) * 1000, 3),
            rule_id         = rule_id,
            violation_type  = violation_type,
            reason          = reason,
            strategy        = strategy,
            risk_score      = score,
            risk_band       = band,
            owasp_id        = fw.get("owasp", ""),
            mitre_id        = fw.get("mitre", ""),
            response_ar     = ar,
            response_en     = en,
            escalated       = escalated,
            scrubbed        = safe_args is not None,
            suggest_tools   = suggest,
            signals         = signals,
            trace           = trace,
            policy_version  = policy_version,
        )

        log_level = cfg.get("log_level", "WARNING")
        getattr(logger, log_level.lower(), logger.warning)(
            "BLOCK audit=%s rule=%s vtype=%s risk=%s/%s tool=%s role=%s",
            audit_id[:8], rule_id, violation_type, score, band,
            call.tool_name, call.session_role,
        )
        _audit(call, d, safe_args=safe_args)
        _telemetry.record(d)
        return d

    # ══════════════════════════════════════════════════════════════════════
    # GATE 0 — Rate limiter  (PR-08)
    # ══════════════════════════════════════════════════════════════════════
    g0_start = time.perf_counter()
    exceeded, rate_reason, rate_params = _check_rate_limit(call.context_id, policy)
    trace.append(GateTrace(
        gate="rate", result="block" if exceeded else "pass",
        rule_id="PR-08" if exceeded else None,
        reason=rate_reason,
        latency_ms=round((time.perf_counter() - g0_start) * 1000, 3),
    ))
    if exceeded:
        return _block("rate", "PR-08", "abuse", rate_reason, rate_params=rate_params)

    # ══════════════════════════════════════════════════════════════════════
    # GATE 1 — Schema validation  (Pydantic)
    # ══════════════════════════════════════════════════════════════════════
    g1_start    = time.perf_counter()
    validation  = validate_tool_call(call.tool_name, call.arguments)
    g1_latency  = round((time.perf_counter() - g1_start) * 1000, 3)
    if not validation.valid:
        err_summary = "; ".join(validation.errors[:3])
        trace.append(GateTrace(
            gate="schema", result="block", rule_id="SCHEMA",
            reason=err_summary[:200], latency_ms=g1_latency,
        ))
        return _block("schema", "SCHEMA", "schema_violation",
                      f"Schema validation failed: {err_summary}")
    trace.append(GateTrace(gate="schema", result="pass", latency_ms=g1_latency))

    # ══════════════════════════════════════════════════════════════════════
    # GATE 2 — Injection detection
    # ══════════════════════════════════════════════════════════════════════
    g2_start   = time.perf_counter()
    injection  = detect_injection(call.arguments)
    g2_latency = round((time.perf_counter() - g2_start) * 1000, 3)
    if injection.detected:
        _VECTOR_VTYPE = {
            "sql":      "injection",
            "prompt":   "prompt_leakage",
            "unicode":  "injection",
            "encoding": "injection",
            "polyglot": "injection",
        }
        top_vector = (
            max(injection.signals, key=lambda s: s.confidence).vector
            if injection.signals else "injection"
        )
        vtype   = _VECTOR_VTYPE.get(top_vector, "injection")
        sigs    = [{"detector": s.detector, "confidence": s.confidence,
                    "vector": s.vector, "evidence": s.evidence}
                   for s in injection.signals]
        reason  = f"Injection detected (conf={injection.confidence:.2f}): {injection.reason}"
        trace.append(GateTrace(
            gate="injection", result="block", rule_id="INJECT",
            reason=reason[:200], latency_ms=g2_latency,
        ))
        return _block("injection", "INJECT", vtype, reason, signals=sigs)
    trace.append(GateTrace(gate="injection", result="pass", latency_ms=g2_latency))

    # ══════════════════════════════════════════════════════════════════════
    # GATE 3 — YAML rules  (PR-01 → PR-13, ordered for correctness)
    # PR-01 is checked explicitly BEFORE PR-05 to preserve correct rule IDs
    # ══════════════════════════════════════════════════════════════════════

    # Safe-pattern allowlist check (FP mitigation) — runs before YAML rules
    safe_patterns = _SAFE_PATTERNS.get(call.tool_name, [])
    for sp in safe_patterns:
        if sp.search(args_str):
            trace.append(GateTrace(
                gate="allowlist", result="skip", rule_id=None,
                reason="Matched safe pattern — YAML rules skipped",
                latency_ms=round((time.perf_counter() - t_start) * 1000, 3),
            ))
            break  # allowlist matched — skip YAML rules but still proceed to ALLOW below

    else:
        # No allowlist match — evaluate all YAML rules
        g3_start = time.perf_counter()
        for rule in policy.get("layer2_rules", []):
            rule_id = rule.get("id", "?")
            if not rule.get("enabled", True): continue
            if "tool_calls_per_minute" in rule.get("condition", ""):
                continue  # this specific PR-08 sub-rule is handled by Gate 0

            # Tool scope filter
            rt = rule.get("tool"); rts = rule.get("tools")
            if rt  and call.tool_name != rt:            continue
            if rts and call.tool_name not in rts:       continue

            matched, evidence = _matches_rule(rule, call, policy, args_str)
            if not matched: continue

            vtype    = rule.get("violation_type", "unknown")
            safe_args = None

            # PR-13: scrub credentials from both evidence and audit payload
            if rule_id == "PR-13":
                scrubbed_dict, _ = _scrub_arguments(call.arguments)
                safe_args = scrubbed_dict
                evidence  = _SCRUB_PLACEHOLDER

            reason = evidence or rule.get("description", rule_id)
            return _block("rule", rule_id, vtype, reason, evidence=evidence,
                          safe_args=safe_args)

        # All YAML rules passed
        trace.append(GateTrace(
            gate="rule", result="pass",
            latency_ms=round((time.perf_counter() - g3_start) * 1000, 3),
        ))

    # ══════════════════════════════════════════════════════════════════════
    # ALL GATES PASSED → ALLOW
    # ══════════════════════════════════════════════════════════════════════
    total_ms = round((time.perf_counter() - t_start) * 1000, 3)
    d = PolicyDecision(
        decision       = "ALLOW",
        audit_id       = audit_id,
        gate           = "passed",
        latency_ms     = total_ms,
        trace          = trace,
        policy_version = policy_version,
    )
    logger.debug("ALLOW audit=%s tool=%s role=%s latency=%.1fms",
                 audit_id[:8], call.tool_name, call.session_role, total_ms)
    _audit(call, d)
    _telemetry.record(d)
    return d



# Public helpers


def get_telemetry() -> TelemetrySnapshot:
    """Return a point-in-time `TelemetrySnapshot` of pipeline statistics."""
    return _telemetry.snapshot()


def reset_telemetry() -> None:
    """
    Reset all telemetry counters to zero.

    Intended for use between independent evaluation or benchmark runs, so
    that statistics from an earlier run do not carry over into a new one.
    """
    _telemetry.reset()



# Benchmark runner
#
# Supports evaluating this pipeline's detection behaviour against a
# caller-supplied, labelled set of tool calls. The benchmark itself
# (dataset construction, case selection) is not defined here; this module
# only executes labelled cases through `run_layer2()` and aggregates the
# resulting precision/recall/F1 and latency statistics.


@dataclass
class BenchmarkCase:
    """
    One labelled test case for `run_benchmark()`.

    `expected_block` is the ground-truth label (True if the call should be
    blocked). `expected_rule`, if given, additionally requires the exact
    rule_id that blocked the call to match, not just the ALLOW/BLOCK
    outcome.
    """
    label:          str
    call:           PlannedToolCall
    expected_block: bool
    expected_rule:  Optional[str] = None   # None means "don't check rule ID"


@dataclass
class BenchmarkReport:
    """
    Aggregate result of one `run_benchmark()` run.

    `precision`, `recall`, and `f1` are computed over the labelled cases
    in this run (via `_Telemetry.record_labelled`); `wrong` lists the
    labels of cases whose decision, or rule_id when checked, did not match
    the expectation.
    """
    total:          int
    correct:        int
    wrong:          list[str]        # labels of wrong cases
    precision:      Optional[float]
    recall:         Optional[float]
    f1:             Optional[float]
    avg_latency_ms: float
    p95_latency_ms: float
    telemetry:      TelemetrySnapshot


def run_benchmark(cases: list[BenchmarkCase]) -> BenchmarkReport:
    """
    Run a labelled set of tool calls through `run_layer2()` and report results.

    Parameters
    ----------
    cases : list[BenchmarkCase]
        Labelled cases to evaluate, in order.

    Returns
    -------
    BenchmarkReport
        Aggregate precision/recall/F1 (derived from the labels in `cases`),
        latency statistics, the list of mismatched case labels, and the
        final `TelemetrySnapshot`.

    Notes
    -----
    Calls `reset_telemetry()` at the start, so any telemetry accumulated
    before this call is discarded. For each case, the rate-limit bucket
    for that case's `context_id` is reset before evaluation, so that
    benchmark cases do not trigger Gate 0 (rate limiting) against each
    other purely due to case ordering.
    """
    reset_telemetry()
    wrong:    list[str]   = []
    latencies: list[float] = []

    for bc in cases:
        reset_rate_bucket(bc.call.context_id)
        d = run_layer2(bc.call)
        _telemetry.record_labelled(d, bc.expected_block)
        latencies.append(d.latency_ms)

        actual_block = d.decision == "BLOCK"
        decision_ok  = actual_block == bc.expected_block
        rule_ok      = (bc.expected_rule is None) or (d.rule_id == bc.expected_rule)
        if not (decision_ok and rule_ok):
            wrong.append(
                f"{bc.label}: expected={'BLOCK' if bc.expected_block else 'ALLOW'} "
                f"rule={bc.expected_rule} | got decision={d.decision} rule={d.rule_id}"
            )

    snap = _telemetry.snapshot()
    return BenchmarkReport(
        total          = len(cases),
        correct        = len(cases) - len(wrong),
        wrong          = wrong,
        precision      = snap.precision,
        recall         = snap.recall,
        f1             = snap.f1,
        avg_latency_ms = snap.avg_latency_ms,
        p95_latency_ms = snap.p95_latency_ms,
        telemetry      = snap,
    )
