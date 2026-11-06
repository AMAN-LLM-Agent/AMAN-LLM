
# src/layer2/injection_detector.py
# AMAN-LLM
# Layer 2 — Policy Engine Orchestrator / Deep Injection Content Scanner


## PURPOSE

# Performs multi-signal, pattern-based detection of injection attempts inside
# tool-call arguments. This module runs after schema validation has already
# confirmed that the arguments conform to the expected structure, and before
# the Policy Engine Orchestrator dispatches the call to the tool sandbox.

## ARCHITECTURAL ROLE

# Within the AMAN-LLM pipeline, this module implements the content-inspection
# stage of Layer 2 (Policy Engine / Tool-Call Enforcement). It is applied
# after schema validation and declarative policy rules have already run, and
# it inspects the textual content of arguments for patterns associated with
# SQL injection, prompt injection, Unicode-based obfuscation, encoding-based
# obfuscation, and multi-vector ("polyglot") payloads. It complements, rather
# than replaces, upstream schema validation and rule-based policy checks.

## INPUTS
#
# A schema-validated tool-call argument payload (`dict[str, Any]`), which may
# contain nested dictionaries, lists, and string values.

## OUTPUTS

# An `InjectionResult` describing whether an injection attempt was detected,
# the associated confidence score, a human-readable summary for the audit
# log, and the full list of fired `InjectionSignal` records for forensic
# inspection.
#
## IMPLEMENTATION NOTES
#
# - The module is implemented in pure Python, using regular-expression and
#   heuristic checks rather than a machine-learning classifier.
# - Each individual detector returns zero or more `InjectionSignal` objects,
#   each carrying a heuristic confidence value in the range [0.0, 1.0].
# - `detect_injection()` aggregates all fired signals and applies the
#   threshold logic described below to produce the final decision.
# - Confidence values are heuristic scores assigned per detection rule; they
#   are not calibrated probabilities.
# - Detection thresholds (`HARD_THRESHOLD`, `SOFT_COMBINED_THRESHOLD`) are
#   module-level constants and can be adjusted without modifying detector
#   logic.
#
## REPRODUCIBILITY

# All detection rules in this module are deterministic (no sampling or
# randomness is involved). The corresponding test suite is located at
# tests/test_layer2_rules.py. Python version and dependency requirements are
# not specified in this module; refer to the repository-level environment
# configuration.
#
## RETURN CONTRACT

# - `InjectionResult.detected == False` → the caller proceeds to the sandbox
#   execution stage.
# - `InjectionResult.detected == True`  → the caller (Policy Engine
#   Orchestrator) is expected to reject the call.
#   - `.signals`    : list[InjectionSignal] — all fired signals with scores.
#   - `.reason`     : str   — human-readable summary for the audit log.
#   - `.confidence` : float — the maximum single-signal confidence, or the
#                     combined confidence when the soft threshold path fires.

# Note: this module implements a pattern-based detection mechanism. It
# detects the categories of injection patterns enumerated below; it does not
# constitute a complete or provably exhaustive defense against all possible
# injection strategies, and previously unseen attack formulations may not be
# captured by the predefined patterns.


from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


# Confidence thresholds

# The two thresholds below implement the aggregation policy used by
# detect_injection(): a single high-confidence signal is sufficient to
# trigger a block, while multiple lower-confidence signals can jointly
# trigger a block once their combined score is high enough. Both values are
# configurable module-level constants; their calibration procedure, if any,
# is not specified in this module.

# A single signal at or above this score triggers an immediate block.
HARD_THRESHOLD: float = 0.85

# Two or more lower-confidence ("soft") signals whose combined score reaches
# this threshold also trigger a block.
SOFT_COMBINED_THRESHOLD: float = 0.80



# Data classes


@dataclass
class InjectionSignal:
    """
    A single fired detection signal produced by one of the detectors below.

    Attributes
    ----------
    detector : str
        Identifier of the detector rule that fired (e.g. "sql_tautology").
    confidence : float
        Heuristic confidence score in the range [0.0, 1.0] assigned to this
        rule; not a calibrated probability.
    evidence : str
        Truncated snippet or description of the matched content, intended
        for audit logging.
    vector : str
        Category of the fired signal: one of 'sql', 'prompt', 'unicode',
        'encoding', or 'polyglot'.
    """
    detector:   str
    confidence: float
    evidence:   str
    vector:     str


@dataclass
class InjectionResult:
    """
    Aggregated detection result returned by detect_injection() to the
    Policy Engine Orchestrator.

    Attributes
    ----------
    detected : bool
        Whether the aggregated evidence met the block threshold.
    confidence : float
        Maximum single-signal confidence, or the combined confidence when
        the soft-threshold path fired. Defaults to 0.0 when no signal fired.
    reason : str
        Human-readable summary of the triggering evidence, for the audit
        log. Empty when no signal fired.
    signals : list[InjectionSignal]
        All fired signals, including those below the block threshold, kept
        for forensic and audit purposes.
    """
    detected:   bool
    confidence: float                          = 0.0
    reason:     str                            = ""
    signals:    list[InjectionSignal]          = field(default_factory=list)



# Helpers


def _snippet(text: str, maxlen: int = 80) -> str:
    """
    Truncate `text` to `maxlen` characters for inclusion in audit evidence,
    appending an ellipsis when truncation occurs. Used to bound the size of
    evidence strings recorded in InjectionSignal.evidence.
    """
    s = text[:maxlen]
    return s + "…" if len(text) > maxlen else s


def _normalise(text: str) -> str:
    """
    Pre-process `text` prior to pattern matching by removing zero-width and
    other invisible Unicode characters and normalising the result to
    Unicode NFC form, with whitespace collapsed.

    This step is applied before the SQL- and prompt-injection detectors so
    that payloads obfuscated with invisible separators collapse to a
    canonical form before regular-expression matching is attempted.
    """
    # Remove zero-width and other invisible Unicode characters that could
    # otherwise be used to split a sensitive token across codepoints and
    # evade direct substring or regex matching.
    cleaned = re.sub(
        r'[\u200b\u200c\u200d\u2060\ufeff\u00ad\u034f\u115f\u1160'
        r'\u17b4\u17b5\u180b-\u180d\u180e\ufe00-\ufe0f]',
        '', text
    )
    return unicodedata.normalize('NFC', cleaned)



# 1. SQL Injection Detector


# Ordered list of (detector name, confidence, compiled pattern) tuples used
# by _detect_sql_injection(). Confidence values reflect the specificity of
# each pattern: constructs with few legitimate use cases inside tool-call
# arguments (e.g. stacked queries, UNION SELECT) are assigned higher
# confidence than constructs that can also occur in benign input (e.g. a
# doubled quote character).
_SQL_SIGNALS: list[tuple[str, float, re.Pattern]] = [
    # Stacked / terminated queries, e.g. "'; DROP TABLE ..." or "1; SELECT ..."
    ("sql_stacked_query",       0.95,
     re.compile(r"['\"`]?\s*;\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|CALL)",
                re.IGNORECASE)),

    # Classic tautology, e.g. " OR 1=1 --" or " OR 'a'='a'"
    ("sql_tautology",            0.90,
     re.compile(r"(\bOR\b|\bAND\b)\s+(\d+\s*=\s*\d+|['\"].+['\"]\s*=\s*['\"].+['\"]|true|false)\s*(--|#|/\*)",
                re.IGNORECASE)),

    # Comment-stripping delimiters used to truncate the remainder of a query
    ("sql_comment_strip",        0.85,
     re.compile(r"(/\*.*?\*/|--[^\n]*|#[^\n]*)", re.IGNORECASE | re.DOTALL)),

    # UNION-based extraction, e.g. "UNION SELECT NULL, ..."
    ("sql_union_extract",        0.95,
     re.compile(r"\bUNION\s+(ALL\s+)?SELECT\b", re.IGNORECASE)),

    # Hex / CHAR() encoding used to obfuscate literals, e.g. CHAR(65,66) or 0x41424344
    ("sql_hex_encoding",         0.80,
     re.compile(r"(CHAR\s*\(\s*\d+|\bCHARINDEX\b|0x[0-9a-fA-F]{4,})", re.IGNORECASE)),

    # Time-based blind indicators, e.g. SLEEP(5), BENCHMARK(...), WAITFOR DELAY
    ("sql_time_based_blind",     0.90,
     re.compile(r"(\bSLEEP\s*\(|\bBENCHMARK\s*\(|\bWAITFOR\s+DELAY\b|\bPG_SLEEP\s*\()",
                re.IGNORECASE)),

    # Error-based blind indicators, e.g. EXTRACTVALUE(...), UPDATEXML(...)
    ("sql_error_based_blind",    0.90,
     re.compile(r"(\bEXTRACTVALUE\s*\(|\bUPDATEXML\s*\(|\bEXP\s*\(~)", re.IGNORECASE)),

    # Subquery-based exfiltration pattern, e.g. "(SELECT ... FROM ...)"
    ("sql_subquery_exfil",       0.75,
     re.compile(r"\(\s*SELECT\s+.+\s+FROM\s+\w+", re.IGNORECASE | re.DOTALL)),

    # Quote-escape sequences, e.g. '' or \' — lower confidence since these
    # can also appear in legitimate free-text input.
    ("sql_quote_escape",         0.60,
     re.compile(r"('')|(\\')", re.IGNORECASE)),
]


def _detect_sql_injection(text: str) -> list[InjectionSignal]:
    """
    Apply the ordered `_SQL_SIGNALS` patterns to `text` (after
    normalisation) and return the list of signals whose pattern matched.
    More than one signal may fire for the same input.
    """
    signals: list[InjectionSignal] = []
    normalised = _normalise(text)

    for name, confidence, pattern in _SQL_SIGNALS:
        m = pattern.search(normalised)
        if m:
            signals.append(InjectionSignal(
                detector=name,
                confidence=confidence,
                evidence=_snippet(m.group(0)),
                vector="sql",
            ))
    return signals



# 2. Prompt Injection Detector


# Ordered list of (pattern, confidence, label) tuples covering direct
# instruction-override attempts, persona/role hijacking, system-prompt
# extraction attempts, delimiter-based injection, indirect injection via
# tool arguments, and Arabic-language variants of the above categories.
_PROMPT_SIGNALS: list[tuple[re.Pattern, float, str]] = [

    # -- Direct override instructions --------------------------------------
    (re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?|context)",
        re.IGNORECASE), 0.95, "prompt_override_en"),

    (re.compile(
        r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|rules?|guidelines?)",
        re.IGNORECASE), 0.95, "prompt_disregard_en"),

    (re.compile(
        r"forget\s+(all\s+)?(your\s+)?(previous\s+)?(instructions?|training|rules?|guidelines?)",
        re.IGNORECASE), 0.90, "prompt_forget_en"),

    # -- Role / persona hijacking -------------------------------------------
    (re.compile(
        r"(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)|roleplay\s+as)"
        r"\s+(an?\s+)?(different|new|unrestricted|jailbroken|evil|DAN)",
        re.IGNORECASE), 0.90, "prompt_persona_hijack"),

    (re.compile(
        r"\bDAN\b|\bDO\s+ANYTHING\s+NOW\b|jailbreak\s+mode|developer\s+mode\s+enabled",
        re.IGNORECASE), 0.95, "prompt_jailbreak_mode"),

    # -- System prompt extraction --------------------------------------------
    (re.compile(
        r"(reveal|print|show|output|display|repeat|tell\s+me)\s+"
        r"(your\s+)?(system\s+prompt|hidden\s+(rules?|instructions?)|"
        r"initial\s+prompt|original\s+(instructions?|prompt)|full\s+prompt)",
        re.IGNORECASE), 0.95, "prompt_extraction_en"),

    (re.compile(
        r"what\s+(are\s+)?(your\s+)?(instructions?|rules?|system\s+prompt|directives?)",
        re.IGNORECASE), 0.80, "prompt_probe_en"),

    # -- Instruction injection via delimiter tricks --------------------------
    (re.compile(
        r"(###\s*SYSTEM|###\s*INSTRUCTIONS|<\|system\|>|<\|im_start\|>system"
        r"|\[INST\]|\[SYS\]|<<SYS>>)",
        re.IGNORECASE), 0.90, "prompt_delimiter_inject"),

    # -- Arabic-language prompt injection / override -------------------------
    (re.compile(
        r"(تجاهل|أهمل|تناسى|انسَ)\s*.{0,10}"
        r"(التعليمات|الأوامر|السياق|القواعد|النظام|ما\s*سبق)",
        re.IGNORECASE | re.UNICODE), 0.95, "prompt_override_ar"),

    (re.compile(
        r"(أنت\s+الآن|تصرف\s+ك|العب\s+دور)\s*.{0,15}"
        r"(نظام\s+مختلف|ذكاء\s+اصطناعي\s+آخر|بدون\s+قيود|غير\s+محدود)",
        re.IGNORECASE | re.UNICODE), 0.90, "prompt_persona_hijack_ar"),

    (re.compile(
        r"(اكشف|أظهر|اعرض|أخبرني\s+عن)\s*.{0,15}"
        r"(تعليماتك|برومبتك|النظام\s+الداخلي|القواعد\s+المخفية|إرشاداتك)",
        re.IGNORECASE | re.UNICODE), 0.95, "prompt_extraction_ar"),

    # -- Indirect injection via tool argument content ------------------------
    # Matches phrasing that attempts to address a future automated reader of
    # the argument content itself (e.g. instructions embedded in a document
    # that is expected to be processed by an LLM downstream).
    (re.compile(
        r"(when\s+you\s+(read|process|see)\s+this|upon\s+receiving\s+this"
        r"|after\s+reading\s+this\s+file)",
        re.IGNORECASE), 0.85, "prompt_indirect_inject"),
]


def _detect_prompt_injection(text: str) -> list[InjectionSignal]:
    """
    Apply the ordered `_PROMPT_SIGNALS` patterns to `text` (after
    normalisation) and return the list of signals whose pattern matched.
    """
    signals: list[InjectionSignal] = []
    normalised = _normalise(text)

    for pattern, confidence, label in _PROMPT_SIGNALS:
        m = pattern.search(normalised)
        if m:
            signals.append(InjectionSignal(
                detector=label,
                confidence=confidence,
                evidence=_snippet(m.group(0)),
                vector="prompt",
            ))
    return signals



# 3. Unicode Obfuscation Detector


# Zero-width and other invisible Unicode characters that can be used to
# split a sensitive token across codepoints, defeating straightforward
# substring or lexical matching.
_ZERO_WIDTH_RE = re.compile(
    r'[\u200b\u200c\u200d\u2060\ufeff\u00ad\u034f\u115f\u1160'
    r'\u17b4\u17b5\u180b-\u180d\u180e\ufe00-\ufe0f]'
)

# Bidirectional text override characters, which can alter the visual
# rendering order of text without changing its underlying codepoints.
_BIDI_RE = re.compile(r'[\u202a-\u202e\u2066-\u2069\u200e\u200f]')

# Mapping of selected Cyrillic, Greek, and Arabic-Indic characters to their
# visually similar Latin/ASCII equivalents ("homoglyphs"). Used to detect
# keywords that have been visually disguised via character substitution
# while remaining legible to a human reader.
_HOMOGLYPH_MAP = {
    '\u0430': 'a',  # Cyrillic а -> a
    '\u0435': 'e',  # Cyrillic е -> e
    '\u0456': 'i',  # Cyrillic і -> i
    '\u043e': 'o',  # Cyrillic о -> o
    '\u0440': 'r',  # Cyrillic р -> r
    '\u0441': 'c',  # Cyrillic с -> c
    '\u0445': 'x',  # Cyrillic х -> x
    '\u03bf': 'o',  # Greek ο -> o
    '\u03b1': 'a',  # Greek α -> a
    '\u0660': '0',  # Arabic-Indic 0 -> 0
    '\u0661': '1',  # Arabic-Indic 1 -> 1
    '\u0662': '2',  # Arabic-Indic 2 -> 2
    '\u0663': '3',  # Arabic-Indic 3 -> 3
}

# Keywords whose homoglyph-substituted form is treated as evidence of
# obfuscation. This list is not exhaustive; it is limited to keywords
# already relevant to the SQL- and prompt-injection detectors above.
_SENSITIVE_KEYWORDS = [
    'select', 'union', 'drop', 'delete', 'insert', 'update', 'exec',
    'script', 'alert', 'onerror', 'onload', 'javascript',
    'ignore', 'jailbreak', 'dan', 'bypass',
]


def _homoglyph_normalise(text: str) -> str:
    """
    Return a copy of `text` with each character present in `_HOMOGLYPH_MAP`
    replaced by its mapped ASCII equivalent; characters not present in the
    map are left unchanged.
    """
    return ''.join(_HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def _detect_unicode_obfuscation(text: str) -> list[InjectionSignal]:
    """
    Detect zero-width character usage, bidirectional-override character
    usage, and homoglyph-substituted sensitive keywords in `text`.
    """
    signals: list[InjectionSignal] = []

    # Zero-width character injection: presence alone is treated as a signal,
    # since legitimate tool arguments are not expected to contain these
    # characters.
    zw_matches = _ZERO_WIDTH_RE.findall(text)
    if zw_matches:
        signals.append(InjectionSignal(
            detector="unicode_zero_width",
            confidence=0.80,
            evidence=f"{len(zw_matches)} zero-width character(s) detected",
            vector="unicode",
        ))

    # Bidirectional override injection
    bidi_matches = _BIDI_RE.findall(text)
    if bidi_matches:
        signals.append(InjectionSignal(
            detector="unicode_bidi_override",
            confidence=0.85,
            evidence=f"{len(bidi_matches)} bidi-override character(s): "
                     + repr(bidi_matches[:3]),
            vector="unicode",
        ))

    # Homoglyph-substituted sensitive keyword detection: a keyword is
    # flagged only if it appears after homoglyph normalisation but was not
    # already present in the original (lower-cased) text, indicating that
    # substitution was necessary to reveal it.
    homoglyph_normalised = _homoglyph_normalise(text).lower()
    for kw in _SENSITIVE_KEYWORDS:
        if kw in homoglyph_normalised and kw not in text.lower():
            # Keyword appears only after homoglyph substitution -> obfuscated
            signals.append(InjectionSignal(
                detector="unicode_homoglyph",
                confidence=0.88,
                evidence=f"homoglyph-encoded keyword detected: '{kw}'",
                vector="unicode",
            ))
            break  # One signal per text, to avoid flooding the result set.

    return signals



# 4. Encoding Obfuscation Detector


# HTML entity sequences, e.g. &lt;script&gt;, &#x3C;, &#60;
_HTML_ENTITY_RE = re.compile(r'&(#\d{2,5}|#x[0-9a-fA-F]{2,4}|[a-zA-Z]{2,10});')

# Chained URL percent-encoding sequences (two or more consecutive tokens)
_URL_ENCODING_RE = re.compile(r'%[0-9a-fA-F]{2}(%[0-9a-fA-F]{2})+')

# Chained hex-escape sequences (three or more), e.g. \x41\x42\x43
_HEX_ESCAPE_RE = re.compile(r'(\\x[0-9a-fA-F]{2}){3,}')

# Candidate base64 blob: at least 20 characters drawn from the base64
# alphabet, bounded so as not to match substrings of longer alphanumeric
# tokens (e.g. identifiers or file paths).
_BASE64_RE = re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{20,}={0,2}(?![A-Za-z0-9+/])')

# Patterns checked against the *decoded* content of a candidate base64 blob;
# reuses vocabulary from the prompt- and SQL-injection detectors above.
_DANGEROUS_DECODED_RE = re.compile(
    r'(ignore\s+(all\s+)?(previous\s+)?instructions|DROP\s+TABLE|UNION\s+SELECT|<script|eval\(|exec\()',
    re.IGNORECASE,
)


def _try_decode_base64(token: str) -> str | None:
    """
    Attempt to base64-decode `token` (padding it as needed) and decode the
    result as UTF-8, ignoring undecodable bytes.

    Returns
    -------
    str | None
        The decoded string if `token` decodes to at least 4 characters;
        `None` if decoding fails or the result is too short to be
        meaningful evidence.
    """
    try:
        padded = token + '=' * (-len(token) % 4)
        decoded = base64.b64decode(padded).decode('utf-8', errors='ignore')
        return decoded if len(decoded) >= 4 else None
    except Exception:
        # Any decoding failure is treated as "not a base64 payload" rather
        # than as an error condition; the caller simply skips this token.
        return None


def _detect_encoding_obfuscation(text: str) -> list[InjectionSignal]:
    """
    Detect HTML-entity encoding, chained URL percent-encoding, chained hex
    escape sequences, and base64-encoded payloads that decode to content
    matching `_DANGEROUS_DECODED_RE`.
    """
    signals: list[InjectionSignal] = []

    # HTML entities: a single entity is common in legitimate text, so three
    # or more occurrences are required before this is treated as a signal.
    html_matches = _HTML_ENTITY_RE.findall(text)
    if len(html_matches) >= 3:
        signals.append(InjectionSignal(
            detector="encoding_html_entities",
            confidence=0.70,
            evidence=f"{len(html_matches)} HTML entities: {html_matches[:3]}",
            vector="encoding",
        ))

    # Chained URL encoding, e.g. %25%3C%73%63%72%69%70%74%3E. Multiple
    # chained percent-encoded tokens are uncommon in legitimate tool-call
    # arguments; the confidence assigned here is below HARD_THRESHOLD on its
    # own but can contribute to the soft combined threshold.
    url_matches = _URL_ENCODING_RE.findall(text)
    if url_matches:
        signals.append(InjectionSignal(
            detector="encoding_url_percent",
            confidence=0.55,
            evidence=_snippet(url_matches[0]),
            vector="encoding",
        ))

    # Hex escape sequences
    hex_matches = _HEX_ESCAPE_RE.findall(text)
    if hex_matches:
        signals.append(InjectionSignal(
            detector="encoding_hex_escape",
            confidence=0.78,
            evidence=_snippet(hex_matches[0]),
            vector="encoding",
        ))

    # Base64 blobs: only flagged when the decoded content itself matches a
    # dangerous pattern, to avoid false positives on arbitrary base64 data
    # (e.g. legitimate binary attachments or tokens).
    for m in _BASE64_RE.finditer(text):
        decoded = _try_decode_base64(m.group(0))
        if decoded and _DANGEROUS_DECODED_RE.search(decoded):
            signals.append(InjectionSignal(
                detector="encoding_base64_payload",
                confidence=0.92,
                evidence=f"base64 blob decodes to dangerous payload: "
                         + _snippet(decoded),
                vector="encoding",
            ))
            break  # One signal per text, to avoid flooding the result set.

    return signals



# 5. Polyglot / Multi-Vector Detector


def _detect_polyglot(signals: list[InjectionSignal]) -> list[InjectionSignal]:
    """
    Given the signals already fired by the other detectors, emit an
    additional "polyglot_multi_vector" signal when two or more distinct
    attack vectors (e.g. 'sql' and 'prompt') fired on the same input.

    Rationale: combining multiple distinct injection techniques in a single
    payload is treated as an indicator of deliberate evasion attempts, and
    is assigned a higher confidence than any individual vector-specific
    signal.
    """
    vectors = {s.vector for s in signals}
    if len(vectors) >= 2:
        return [InjectionSignal(
            detector="polyglot_multi_vector",
            confidence=0.95,
            evidence=f"Multiple injection vectors detected simultaneously: "
                     + ", ".join(sorted(vectors)),
            vector="polyglot",
        )]
    return []



# Public entry point


def detect_injection(arguments: dict[str, Any]) -> InjectionResult:
    """
    Detect injection-related patterns in tool-call arguments.

    Recursively extracts all string values from `arguments`, applies the
    SQL-injection, prompt-injection, Unicode-obfuscation, and
    encoding-obfuscation detectors to each extracted string, and evaluates
    the aggregated signals against the module's confidence thresholds.

    Parameters
    ----------
    arguments : dict[str, Any]
        Schema-validated tool-call arguments whose string-valued fields
        (including those nested inside dicts/lists, up to the recursion
        depth limit of `_extract_strings`) are inspected.

    Returns
    -------
    InjectionResult
        Aggregated detection result. `.detected` is True if the hard or
        soft threshold was met; `.confidence` holds the corresponding
        score; `.reason` holds a human-readable summary for the audit log;
        `.signals` holds every fired signal, including sub-threshold ones,
        for forensic logging.

    Notes
    -----
    This function performs pattern-based inspection of the argument content
    and does not itself execute the requested tool call.
    """
    all_signals: list[InjectionSignal] = []

    # Flatten all string content from the arguments dict, including values
    # nested inside dicts, lists, and tuples.
    texts = _extract_strings(arguments)

    for text in texts:
        if not text or not text.strip():
            continue
        all_signals.extend(_detect_sql_injection(text))
        all_signals.extend(_detect_prompt_injection(text))
        all_signals.extend(_detect_unicode_obfuscation(text))
        all_signals.extend(_detect_encoding_obfuscation(text))

    # Evaluate cross-vector combinations only after all per-string signals
    # have been collected, so that polyglot detection can consider vectors
    # that fired on different string fields within the same argument set.
    all_signals.extend(_detect_polyglot(all_signals))

    if not all_signals:
        return InjectionResult(detected=False)

    # -- Threshold evaluation -------------------------------------------
    max_confidence = max(s.confidence for s in all_signals)

    # Hard threshold: any single signal at or above HARD_THRESHOLD is
    # sufficient to report a detection on its own.
    hard_fired = [s for s in all_signals if s.confidence >= HARD_THRESHOLD]
    if hard_fired:
        top = max(hard_fired, key=lambda s: s.confidence)
        return InjectionResult(
            detected=True,
            confidence=top.confidence,
            reason=f"[{top.detector}] {top.evidence}",
            signals=all_signals,
        )

    # Soft threshold: no single signal reached HARD_THRESHOLD, but the sum
    # of all fired confidences reaches SOFT_COMBINED_THRESHOLD. This allows
    # several independently inconclusive signals to jointly trigger a
    # detection.
    combined = min(1.0, sum(s.confidence for s in all_signals))
    if combined >= SOFT_COMBINED_THRESHOLD:
        summary = "; ".join(f"[{s.detector}={s.confidence:.2f}]" for s in all_signals)
        return InjectionResult(
            detected=True,
            confidence=combined,
            reason=f"Combined soft signals: {summary}",
            signals=all_signals,
        )

    # Neither threshold was met: report no detection, but retain the
    # sub-threshold signals in the result for audit purposes.
    return InjectionResult(
        detected=False,
        confidence=max_confidence,
        signals=all_signals,     # Low-confidence signals still logged for audit.
    )



# Internal helper


def _extract_strings(obj: Any, _depth: int = 0) -> list[str]:
    """
    Recursively extract all string values from a nested structure composed
    of dicts, lists, and/or tuples.

    Parameters
    ----------
    obj : Any
        The object to extract string values from.
    _depth : int
        Current recursion depth; not intended to be supplied by callers.

    Returns
    -------
    list[str]
        All string values found in `obj`, in traversal order.

    Notes
    -----
    Recursion is depth-limited to 5 levels to bound the cost of processing
    deeply nested argument payloads. Non-string, non-container values are
    ignored.
    """
    if _depth > 5:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        results = []
        for v in obj.values():
            results.extend(_extract_strings(v, _depth + 1))
        return results
    if isinstance(obj, (list, tuple)):
        results = []
        for item in obj:
            results.extend(_extract_strings(item, _depth + 1))
        return results
    return []
