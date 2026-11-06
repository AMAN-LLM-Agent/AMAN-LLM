# src/layer3/web_search_freshness_validator.py
"""

AMAN-LLM
Layer 3 — Output Validator Orchestrator
Component: Web Search Result Freshness Validator


PURPOSE

Estimates whether the text returned by the web-search tool is likely to be
current, using dates (or explicit staleness statements) found in that text.
Where the content is time-sensitive and appears outdated or undated, the
module produces a structured freshness status, a bilingual (Arabic/English)
disclaimer, and machine-readable check codes. It supports the Layer 3
objective of not presenting possibly outdated tool evidence as current fact.

The assessment is heuristic. It relies only on dates that appear in the
result text; it does not verify the true publication or retrieval time of
the underlying source.

ARCHITECTURAL ROLE

Helper of the Output Validator Orchestrator (Layer 3). It is invoked from:

* tool_registry.py     — ``_web_search()`` calls
                         ``validate_web_search_freshness()`` and embeds a
                         metadata header and disclaimer in the tool output
                         (see ``annotate_search_output()``).
* output_validator.py  — Check 5c calls ``validate_web_search_freshness()``
                         on tool evidence and surfaces the resulting codes
                         in ``Layer3Output``. It may also read the header
                         back with ``parse_search_meta()``.

The ``freshness_status`` field is intended to be consumed by downstream
components (Layer 4 — Constitutional Self-Critique, and the pipeline
orchestrator) to decide whether to re-query without parsing ``failed_checks``.

INPUTS

* ``query``       — the user query (used for data-type classification and
                    for the security/political escalation rule).
* ``result_text`` — the raw string returned by ``_web_search()``.
* Optional: ``now``, ``custom_thresholds_h``, ``content_type``,
  ``query_domain`` (see ``validate_web_search_freshness``).

OUTPUTS

A ``FreshnessResult`` dataclass:

* retrieved_at     — most recent date found in the text (UTC), or None.
* data_type        — 'weather' | 'price' | 'news' | 'generic'.
* is_stale         — True when the age of ``retrieved_at`` exceeds the
                     threshold for ``data_type``.
* staleness_h      — age in hours, or None when no usable date was found.
* freshness_status — 'fresh' | 'stale' | 'unknown'.
* should_requote   — True when a fresh search should be attempted before
                     the answer is delivered.
* disclaimer_ar / disclaimer_en — disclaimer text (empty when none applies).
* failed_checks    — check codes (e.g. ``"freshness_stale_weather"``)
                     consumed by ``run_layer3`` / ``Layer3Output``.

IMPLEMENTATION NOTES

Default staleness thresholds (hours; overridable per call):

    weather -> 12      price -> 1      news -> 24      generic -> 72

Processing outline:

1. Optional domain gate: if ``query_domain`` is supplied and is not in
   ``FRESHNESS_GATED_DOMAINS``, the check is skipped and an 'unknown'
   result with no disclaimer is returned.
2. The data type is classified from the query and the first 500 characters
   of the result (``classify_data_type``).
3. The most recent parseable date is extracted (``extract_most_recent_date``);
   future dates are clamped to ``now`` and years before 2000 are discarded
   as corrupt metadata.
4. The age is compared with the per-type threshold.
5. Disclaimers and check codes are produced as follows:
   * Explicit staleness statements in the text (``_STALENESS_ADMISSION_RE``),
     when no date was parsed, are treated as stale regardless of data type.
   * An absent date yields a disclaimer only for freshness-sensitive types
     (``FRESHNESS_SENSITIVE_TYPES``) or for security/political queries
     (code ``freshness_unknown_security``).
   * A date older than the threshold yields ``freshness_stale_<data_type>``.

Design intent recorded by the authors (severity is not assigned in this
module; this module only emits the codes, and the mapping to severity is
performed by the consumer):

* Stale price     -> hard failure in Layer 3 (orthogonal to the existing
                     ``financial_price_sanity`` check).
* Stale weather   -> soft failure; the answer is delivered with a disclaimer.
* Unknown date    -> warning rather than failure.

Only the Python standard library is used (``datetime``, ``re``, ``logging``,
``dataclasses``, ``typing``).

REPRODUCIBILITY

The module has no randomness and no network access. Given identical
``query``, ``result_text`` and ``now``, the output is deterministic; passing
an explicit timezone-aware ``now`` is recommended for tests and audits. The
thresholds and regular expressions defined below are the values used by this
implementation.

DEVELOPMENT NOTES

This module originated from two development traces (IDs b0985a46 and
55328600) in which stale weather and commodity-price data were returned
without a freshness disclaimer. Later revisions, marked ``[FIX ...]`` in
the inline comments, (a) suppressed the "no date" disclaimer for timeless
content, (b) added the structured ``freshness_status`` field, (c) added the
domain gate, (d) added escalation for security/political queries (trace
152750af), and (e) added detection of explicit staleness statements.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# Staleness thresholds

# Maximum acceptable content age, in hours, per data type. Content older than
# the threshold is classified as stale. These defaults can be overridden per
# call through ``custom_thresholds_h``.

_STALE_THRESHOLDS_H: dict[str, float] = {
    "weather":   12.0,
    "price":      1.0,
    "news":      24.0,
    "generic":   72.0,
}


# Freshness-sensitive data types

# Data types for which the absence of any date triggers a disclaimer.
# Other types (e.g. recipe, tutorial, definition, how_to, greeting) are
# treated as timeless and receive no "no date" disclaimer.
# Note: ``classify_data_type()`` returns only 'weather', 'price', 'news' or
# 'generic'; the remaining labels also match pipeline ``content_type`` values
# supplied by the caller.

FRESHNESS_SENSITIVE_TYPES = {"weather", "news", "financial", "current_events", "price", "prices", "sports"}


# Pipeline query-domain gate

# When the pipeline supplies ``query_domain`` to
# ``validate_web_search_freshness()``, the full check (disclaimer generation
# and ``failed_checks`` population) runs only if the domain is in this set.
# Other domains (e.g. recipe, tutorial, definition, greeting, fiqh,
# technical) are treated as timeless and the check is skipped. The gate is
# not applied when ``query_domain`` is None.
#
# Equivalent logic:
#   if domain in {"news", "current_events", "finance", "weather", "sports"}:
#       run_freshness_check()
FRESHNESS_GATED_DOMAINS = {"news", "current_events", "finance", "weather", "sports"}


# [FIX Trace 152750af — Issue 2] Security / political topic escalation

# Rationale: for queries on terrorism, security, geopolitics or
# counter-extremism, the status of events can change quickly, so a missing
# date is not treated as benign. When such a query yields no date, the
# generic 'freshness_unknown_<type>' code is replaced by
# 'freshness_unknown_security'. The caller (output_validator.py, Check 5c
# routing) is expected to map this code to ``require_tool_evidence=True``
# before a Layer 3 PASS can be issued; that routing is outside this module.
#
# Applies even when ``data_type == 'generic'``.

# Keyword pattern (Arabic and English) identifying security / political /
# counter-terrorism queries. It is a lexical heuristic, not a topic classifier.
_SECURITY_POLITICAL_RE = re.compile(
    r"\b(داعش|ISIS|ISIL|إرهاب|إرهابي|تطرف|القاعدة|طالبان|جهاد"
    r"|terrorism|terrorist|extremis[mt]|jihadis[mt]|counter.?terrorism"
    r"|al[- ]qa[ei]da|geopolit|sanctions|حوثي|حزب.?الله|حماس|Hamas|Hezbollah"
    r"|سياسة|سياسي|دبلوماسي|political|security.?policy|national.?security)\b",
    re.IGNORECASE | re.UNICODE,
)


def _is_security_political_query(query: str) -> bool:
    """Return True when *query* matches ``_SECURITY_POLITICAL_RE``
    (security, geopolitics or counter-terrorism vocabulary)."""
    return bool(_SECURITY_POLITICAL_RE.search(query))


# [FIX freshness-admitted-staleness] Explicit staleness statements

# Problem addressed: ``extract_most_recent_date()`` recognises only
# machine-parseable dates (ISO, "June 3 2026", slash formats, etc.). Some
# tool outputs instead state their own staleness in plain language, e.g.
# "تقرير قديم" (old report), "بيانات قديمة جداً" (very old data),
# "منذ سنوات" (for years), "الأسبوع الماضي" (last week), or contain only a
# bare past year ("سنة 2022"). These cases previously fell into the
# "no date found" branch, which is itself gated by
# ``data_type in FRESHNESS_SENSITIVE_TYPES``. That gate is frequently not
# met by time-sensitive queries (sports results, elections, technology
# news), because ``classify_data_type()`` recognises only weather, price and
# news keywords.
#
# Approach: detect such statements directly in the source text. They are
# treated as evidence of staleness independently of ``data_type``,
# ``FRESHNESS_SENSITIVE_TYPES`` and ``FRESHNESS_GATED_DOMAINS``, because
# those gates exist to suppress warnings for undated but plausibly timeless
# content, not for content that indicates it is outdated.
# (The query_domain gate is still evaluated first, at function entry.)

_STALENESS_ADMISSION_RE = re.compile(
    r"(?:نشرة|تقرير|بيانات|معلومات|أرشيف|مصدر|توقعات|مقال)\s+قديم[ةه]?"
    r"|قديم[ةه]?\s*جد"
    r"|من\s+أرشيف"
    r"|منذ\s+(?:عدة\s+)?سنوات?\b"
    r"|الأسبوع\s+الماضي"
    r"|بدون\s+تاريخ"
    r"|\b20(?:0\d|1\d|2[0-5])\b",   # bare year 2000-2025; the upper bound assumes a current year of 2026 or later
    re.UNICODE,
)


def _has_staleness_admission(text: str) -> bool:
    """Return True when *text* matches ``_STALENESS_ADMISSION_RE``."""
    return bool(_STALENESS_ADMISSION_RE.search(text))


# Data-type classifiers (keyword patterns)

# Multilingual (Arabic, English, French) keyword patterns used by
# ``classify_data_type()``. They are lexical heuristics applied to the query
# and to the beginning of the result text.

_WEATHER_RE = re.compile(
    r"\b(weather|forecast|temperature|humidity|rain|cloud|طقس|درجة الحرارة"
    r"|رطوبة|أمطار|غيوم|نشرة جوية|météo|prévisions)\b",
    re.IGNORECASE | re.UNICODE,
)

_PRICE_RE = re.compile(
    r"\b(price|gold|silver|oil|crude|forex|exchange rate|سعر|ذهب|فضة|نفط"
    r"|برنت|دولار|يورو|عملة|بورصة|cours|prix|or|pétrole)\b",
    re.IGNORECASE | re.UNICODE,
)

_NEWS_RE = re.compile(
    r"\b(news|breaking|latest|headline|أخبار|عاجل|آخر الأخبار|nouvelles|actualité)\b",
    re.IGNORECASE | re.UNICODE,
)


# Date extraction


# Date patterns as (regular expression, strptime format or None) pairs,
# ordered from most specific to least specific. A format of None indicates
# that the match is converted by a dedicated parser in
# ``extract_most_recent_date()``. All patterns are applied; the latest
# resulting datetime is retained.
_DATE_PATTERNS: List[Tuple[str, str]] = [
    # ISO-8601 with time and optional timezone: 2026-06-03T14:22:00Z.
    # Any timezone suffix is matched but not used; the value is interpreted as UTC.
    (
        r"\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)"
        r"(?:Z|[+-]\d{2}:?\d{2})?\b",
        "%Y-%m-%dT%H:%M:%S",
    ),
    # ISO date only: 2026-06-03
    (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
    # English month-name format (month first): June 3, 2026 / Jun 3 2026
    (
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
        r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
        r"|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})\b",
        None,   # handled specially
    ),
    # Arabic month-name format (day, month, year): 03 يونيو 2026
    (
        r"\b(\d{1,2})\s+(يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس"
        r"|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+(\d{4})\b",
        None,   # handled specially
    ),
    # Numeric day-first format with '/' or '.' separators:
    # 03/06/2026 or 03.06.2026, read as DD/MM/YYYY (European convention)
    (r"\b(\d{2})[/\.](\d{2})[/\.](\d{4})\b", None),
]

# Gregorian month names used in Arabic text (Levant/Gulf naming) -> month number.
_AR_MONTH_MAP: dict[str, int] = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4,
    "مايو":  5, "يونيو":  6, "يوليو": 7, "أغسطس": 8,
    "سبتمبر": 9, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}

# English month names, keyed by their first three lowercase letters.
_EN_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_en_month(match: re.Match) -> Optional[datetime]:
    """Convert a match of the English month-name pattern (month, day, year)
    to a midnight-UTC datetime; return None if the date is invalid."""
    try:
        month_str, day_str, year_str = match.group(1), match.group(2), match.group(3)
        month = _EN_MONTH_MAP[month_str[:3].lower()]
        return datetime(int(year_str), month, int(day_str), tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_ar_month(match: re.Match) -> Optional[datetime]:
    """Convert a match of the Arabic month-name pattern (day, month, year)
    to a midnight-UTC datetime; return None if the date is invalid."""
    try:
        day_str, month_str, year_str = match.group(1), match.group(2), match.group(3)
        month = _AR_MONTH_MAP.get(month_str)
        if month is None:
            return None
        return datetime(int(year_str), month, int(day_str), tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_slash_dot(match: re.Match) -> Optional[datetime]:
    """Interpret as DD/MM/YYYY (EU day-first convention); return None if invalid."""
    try:
        d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return datetime(y, m, d, tzinfo=timezone.utc)
    except Exception:
        return None


def extract_most_recent_date(text: str) -> Optional[datetime]:
    """
    Scan *text* for date strings and return the most recent datetime found.

    Every pattern in ``_DATE_PATTERNS`` is applied to the whole text and the
    maximum of all successfully parsed values is returned. The result is
    therefore the latest date *mentioned* in the text, which is not
    necessarily its publication date.

    Returns None when no recognisable date is present.
    All returned datetimes are UTC-aware; date-only values are set to midnight UTC.
    """
    candidates: List[datetime] = []

    for pattern_str, fmt in _DATE_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
        for m in pattern.finditer(text):
            dt: Optional[datetime] = None

            # Select the converter from the format tag or, for the pattern
            # entries without a format, from characteristic pattern content.
            if fmt and "T" in fmt:
                # ISO-8601 date with time
                raw = f"{m.group(1)}T{m.group(2)}"
                for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
                    try:
                        dt = datetime.strptime(raw, f).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        pass
            elif fmt == "%Y-%m-%d":
                try:
                    dt = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            elif "Jan" in pattern_str or "jan" in pattern_str.lower():
                dt = _parse_en_month(m)
            elif "يناير" in pattern_str:
                dt = _parse_ar_month(m)
            else:
                dt = _parse_slash_dot(m)

            if dt:
                candidates.append(dt)

    return max(candidates) if candidates else None



# Data-type classifier


def classify_data_type(query: str, result_text: str) -> str:
    """
    Return one of: 'weather', 'price', 'news', 'generic'.

    The query and the first 500 characters of the result are searched with
    the keyword patterns above. Precedence when several match:
    price, then weather, then news; 'generic' if none match.
    """
    combined = f"{query} {result_text[:500]}"
    if _PRICE_RE.search(combined):
        return "price"
    if _WEATHER_RE.search(combined):
        return "weather"
    if _NEWS_RE.search(combined):
        return "news"
    return "generic"



# Output dataclass


@dataclass
class FreshnessResult:
    """
    Result of freshness validation for a single web_search output.

    Fields
    ------
    retrieved_at    : most recent date found in the result (UTC), or None.
                      Despite the name, this is a date extracted from the text,
                      not a measured retrieval time.
    data_type       : 'weather' | 'price' | 'news' | 'generic'
    is_stale        : True when the age of ``retrieved_at`` exceeds the
                      per-type threshold
    staleness_h     : content age in hours (None when no date was found)
    freshness_status: 'fresh' | 'stale' | 'unknown'
    should_requote  : True when a re-query should be attempted before delivery
    disclaimer_ar   : Arabic staleness disclaimer to prepend (empty if fresh/timeless)
    disclaimer_en   : English staleness disclaimer (for logs / multilingual UIs)
    failed_checks   : Layer 3 check codes; consumed by run_layer3()
    """
    retrieved_at:    Optional[datetime]  = None
    data_type:       str                 = "generic"
    is_stale:        bool                = False
    staleness_h:     Optional[float]     = None
    freshness_status: str                = "unknown"  # 'fresh' | 'stale' | 'unknown'
    should_requote:  bool                = False
    disclaimer_ar:   str                 = ""
    disclaimer_en:   str                 = ""
    failed_checks:   List[str]           = field(default_factory=list)



# Disclaimer templates (Arabic)

# Keyed by data type; "{date}" is replaced by the formatted source date.
# The "unknown" template is used when no date is available and takes no
# placeholder.

_DISCLAIMER_TEMPLATES_AR: dict[str, str] = {
    "weather": (
        " تنبيه: بيانات الطقس المعروضة قديمة (تاريخ المصدر: {date})، "
        "مما يعني أنها قد لا تعكس الأحوال الجوية الحالية. "
        "يُرجى التحقق من مصدر حديث للحصول على بيانات دقيقة."
    ),
    "price": (
        " تنبيه: سعر السلعة المعروض قديم (تاريخ المصدر: {date}). "
        "أسعار المعادن والعملات تتغير بشكل مستمر؛ "
        "يُرجى الرجوع إلى منصة بيانات مالية متخصصة للسعر الفعلي."
    ),
    "news": (
        " ملاحظة: الخبر المعروض قد يكون قديماً (تاريخ المصدر: {date}). "
        "يُرجى مراجعة مصدر إخباري حديث للتحقق من آخر المستجدات."
    ),
    "generic": (
        " ملاحظة: قد تكون المعلومات المعروضة غير محدّثة (تاريخ المصدر: {date}). "
        "يُنصح بالتحقق من مصدر حديث."
    ),
    # Used when no date could be found (applied only in the branches that
    # emit an "unknown" disclaimer; see validate_web_search_freshness()).
    "unknown": (
        " ملاحظة: لم يتضمن هذا المصدر تاريخاً واضحاً. "
        "قد تكون المعلومات قديمة؛ يُرجى التحقق من مصدر موثوق وحديث."
    ),
}

# English counterparts of the Arabic templates above (same keys, same
# placeholder), intended for logs and multilingual interfaces.
_DISCLAIMER_TEMPLATES_EN: dict[str, str] = {
    "weather": (
        " Warning: weather data is stale (source date: {date}). "
        "Current conditions may differ — please check a live weather service."
    ),
    "price": (
        " Warning: commodity price is outdated (source date: {date}). "
        "Prices change continuously — please consult a real-time financial data source."
    ),
    "news": (
        " Note: this news item may be outdated (source date: {date}). "
        "Please check a current news outlet for the latest updates."
    ),
    "generic": (
        " Note: this information may not be current (source date: {date}). "
        "Please verify with a recent source."
    ),
    "unknown": (
        " Note: no date was found in this source. "
        "Information may be outdated — please verify with a trusted, current source."
    ),
}


def _format_date(dt: Optional[datetime]) -> str:
    """Format *dt* as 'YYYY-MM-DD HH:MM UTC' for use in disclaimers; return a
    bilingual 'unknown' placeholder when *dt* is None."""
    if dt is None:
        return "غير معروف / unknown"
    return dt.strftime("%Y-%m-%d %H:%M UTC")



# Core validator


def validate_web_search_freshness(
    query: str,
    result_text: str,
    *,
    now: Optional[datetime] = None,
    custom_thresholds_h: Optional[dict[str, float]] = None,
    content_type: Optional[str] = None,
    query_domain: Optional[str] = None,
) -> FreshnessResult:
    """
    Validate the freshness of a web_search result.

    Parameters
    
    query:
        The original user query — used for data-type classification and for
        the security/political escalation rule.
    result_text:
        The raw string returned by ``_web_search()``.
    now:
        Override for the current time (UTC-aware datetime), mainly for unit
        tests and audits. Defaults to ``datetime.now(timezone.utc)``. It must
        be timezone-aware, because it is compared with UTC-aware dates.
    custom_thresholds_h:
        Override per-type staleness thresholds (hours). Keys must be one of
        'weather', 'price', 'news', 'generic'.
    content_type:
        Pipeline content-type label. When the data type is classified as
        'generic' and this label is provided and is not in
        ``FRESHNESS_SENSITIVE_TYPES`` (e.g. recipe, tutorial, definition,
        how_to, greeting, comparison), the "no date" disclaimer is
        suppressed. If None, a 'generic' result is treated as
        freshness-sensitive.
    query_domain:
        Pipeline-level domain classification from ``_classify_query_domain()``.
        When provided, the full freshness check (disclaimers, failed_checks)
        is run only when the domain is in FRESHNESS_GATED_DOMAINS:
            {"news", "current_events", "finance", "weather", "sports"}
        Timeless domains (recipe, tutorial, greeting, fiqh, technical) are
        returned immediately with freshness_status='unknown' and no checks.

    Returns
    
    FreshnessResult
        A result object is returned in all handled cases, including when no
        date is found. No exception handling is performed for invalid
        arguments (for example, a timezone-naive ``now``).
    """
    # ── Domain gate ────────────────────────────────────────────────────────
    # Run the full freshness procedure only for time-sensitive query domains.
    # Timeless domains (recipe, tutorial, definition, greeting, fiqh,
    # technical) skip the check entirely.
    if query_domain is not None and query_domain not in FRESHNESS_GATED_DOMAINS:
        logger.debug(
            "freshness | query_domain=%r not in FRESHNESS_GATED_DOMAINS — "
            "skipping freshness check for timeless domain.",
            query_domain,
        )
        return FreshnessResult(
            data_type="generic",
            freshness_status="unknown",
            # No disclaimer, no failed_checks — timeless content
        )

    now = now or datetime.now(timezone.utc)
    thresholds = {**_STALE_THRESHOLDS_H, **(custom_thresholds_h or {})}

    data_type    = classify_data_type(query, result_text)
    retrieved_at = extract_most_recent_date(result_text)

    # ── Age calculation ────────────────────────────────────────────────────
    staleness_h: Optional[float] = None
    is_stale   = False

    if retrieved_at is not None:
        # Clock-skew / future-date handling: dates later than ``now`` are
        # clamped to ``now`` (age zero).
        if retrieved_at > now:
            retrieved_at = now
        # Corrupt metadata dates (year < 2000, e.g. 1978-12-27 produced by a
        # misconfigured CDN header) must not be interpreted as stale content.
        # The date is discarded so that processing continues in the
        # "no date" branches below.
        if retrieved_at.year < 2000:
            logger.warning(
                "freshness | data_type=%s | corrupt retrieved_at year=%d — "
                "ignoring; treating as freshness unknown.",
                data_type, retrieved_at.year,
            )
            retrieved_at = None

    if retrieved_at is not None:
        age_s       = (now - retrieved_at).total_seconds()
        staleness_h = age_s / 3600.0
        threshold   = thresholds.get(data_type, 72.0)
        is_stale    = staleness_h > threshold

        logger.debug(
            "freshness | data_type=%s | retrieved_at=%s | age_h=%.1f | "
            "threshold_h=%.1f | stale=%s",
            data_type, retrieved_at.isoformat(), staleness_h, threshold, is_stale,
        )
    else:
        logger.warning(
            "freshness | data_type=%s | no date found in result — "
            "freshness unknown.",
            data_type,
        )

    # ── freshness_status ──────────────────────────────────────────────────
    # Derived from the date-based assessment only. The explicit-staleness
    # branch below may override this value.
    if retrieved_at is None:
        freshness_status = "unknown"
    elif is_stale:
        freshness_status = "stale"
    else:
        freshness_status = "fresh"

    # ── Disclaimer generation ──────────────────────────────────────────────
    # A "no date" disclaimer is added only for freshness-sensitive data
    # types, so that timeless content (recipe, tutorial, definition, how_to,
    # greeting, etc.) does not receive spurious disclaimers.
    disclaimer_ar = ""
    disclaimer_en = ""
    failed_checks: List[str] = []

    # Decide whether the content warrants a "no date" disclaimer. The data
    # type from classify_data_type() takes priority; if it is 'generic', the
    # pipeline ``content_type`` (when provided) decides, and a non-sensitive
    # content type suppresses the disclaimer.
    _effective_type_is_sensitive = (
        data_type in FRESHNESS_SENSITIVE_TYPES
        or (
            data_type == "generic"
            and (content_type is None or content_type in FRESHNESS_SENSITIVE_TYPES)
        )
    )

    if retrieved_at is None and _has_staleness_admission(result_text):
        # [FIX freshness-admitted-staleness] The source states its own
        # staleness in plain language although no parseable date exists.
        # It is treated as stale unconditionally, bypassing the sensitivity
        # gate used in the next branch, which handles the case in which
        # there is no signal either way. In this branch ``is_stale`` and
        # ``staleness_h`` keep their initial values (False / None) while
        # ``freshness_status`` is set to 'stale'.
        date_str = "غير معروف / unknown"
        disclaimer_ar = _DISCLAIMER_TEMPLATES_AR.get(
            data_type, _DISCLAIMER_TEMPLATES_AR["generic"]
        ).format(date=date_str)
        disclaimer_en = _DISCLAIMER_TEMPLATES_EN.get(
            data_type, _DISCLAIMER_TEMPLATES_EN["generic"]
        ).format(date="unknown")
        failed_checks.append(f"freshness_stale_{data_type}")
        freshness_status = "stale"
        logger.warning(
            "freshness | data_type=%s | explicit staleness admission found in "
            "source text — treating as stale regardless of sensitivity gate.",
            data_type,
        )
    elif retrieved_at is None:
        # No date at all — warn only for freshness-sensitive types
        if _effective_type_is_sensitive:
            disclaimer_ar = _DISCLAIMER_TEMPLATES_AR["unknown"]
            disclaimer_en = _DISCLAIMER_TEMPLATES_EN["unknown"]
            # [FIX Trace 152750af — Issue 2] For security/political queries,
            # use the code 'freshness_unknown_security' so that
            # output_validator can require tool grounding before a PASS.
            if _is_security_political_query(query):
                check_code = "freshness_unknown_security"
                logger.warning(
                    "freshness | data_type=%s | security/political query + no date — "
                    "escalating to freshness_unknown_security (will require tool grounding "
                    "before L3 PASS).",
                    data_type,
                )
            else:
                check_code = f"freshness_unknown_{data_type}"
            failed_checks.append(check_code)
            logger.warning(
                "freshness | data_type=%s | no date — adding disclaimer "
                "(freshness-sensitive type).",
                data_type,
            )
        else:
            # Non-sensitive type: security/political queries without a date
            # are still escalated, since the situation may change rapidly.
            if _is_security_political_query(query):
                disclaimer_ar = _DISCLAIMER_TEMPLATES_AR["unknown"]
                disclaimer_en = _DISCLAIMER_TEMPLATES_EN["unknown"]
                failed_checks.append("freshness_unknown_security")
                logger.warning(
                    "freshness | data_type=%s | security/political query classified "
                    "as generic + no date — escalating to freshness_unknown_security.",
                    data_type,
                )
            else:
                logger.debug(
                    "freshness | data_type=%s | content_type=%r | no date — "
                    "skipping disclaimer (timeless content type).",
                    data_type, content_type,
                )

    elif is_stale:
        # A date was found and exceeds the per-type threshold.
        date_str      = _format_date(retrieved_at)
        disclaimer_ar = _DISCLAIMER_TEMPLATES_AR.get(
            data_type, _DISCLAIMER_TEMPLATES_AR["generic"]
        ).format(date=date_str)
        disclaimer_en = _DISCLAIMER_TEMPLATES_EN.get(
            data_type, _DISCLAIMER_TEMPLATES_EN["generic"]
        ).format(date=date_str)
        failed_checks.append(f"freshness_stale_{data_type}")

    # Re-query hint: recommend a new search before delivery, only for the
    # time-critical types (weather, price) and only for date-based staleness.
    should_requote = is_stale and data_type in ("weather", "price")

    return FreshnessResult(
        retrieved_at=retrieved_at,
        data_type=data_type,
        is_stale=is_stale,
        staleness_h=round(staleness_h, 2) if staleness_h is not None else None,
        freshness_status=freshness_status,
        should_requote=should_requote,
        disclaimer_ar=disclaimer_ar,
        disclaimer_en=disclaimer_en,
        failed_checks=failed_checks,
    )



# Convenience: annotate raw web_search output with a metadata header


def annotate_search_output(
    result_text: str,
    freshness: FreshnessResult,
    *,
    include_disclaimer: bool = True,
) -> str:
    """
    Prepend a machine-readable ``[web_search_meta]`` header and, when
    available, the Arabic and English disclaimers to *result_text*.

    The header is a single line that Layer 3 / Layer 4 can parse
    (see ``parse_search_meta()``):
        [web_search_meta retrieved_at=2026-06-03T12:00:00Z data_type=weather stale=True age_h=54.3 freshness_status=stale]

    Parameters
    
    result_text:
        Original ``_web_search()`` return string.
    freshness:
        Output of ``validate_web_search_freshness()``.
    include_disclaimer:
        Prepend the Arabic and English disclaimers (when non-empty) if True
        (default). Set to False in test / audit contexts where only the
        header is needed.

    Returns
    -------
    str
        Header line, optional disclaimers, and the original text, joined by
        newlines.
    """
    retrieved_str = (
        freshness.retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        if freshness.retrieved_at
        else "unknown"
    )
    age_str = f"{freshness.staleness_h:.1f}" if freshness.staleness_h is not None else "unknown"

    meta_line = (
        f"[web_search_meta retrieved_at={retrieved_str} "
        f"data_type={freshness.data_type} "
        f"stale={freshness.is_stale} "
        f"age_h={age_str} "
        f"freshness_status={freshness.freshness_status}]"
    )

    parts: List[str] = [meta_line]

    if include_disclaimer and freshness.disclaimer_ar:
        parts.append(freshness.disclaimer_ar)
    if include_disclaimer and freshness.disclaimer_en:
        parts.append(freshness.disclaimer_en)

    parts.append(result_text)
    return "\n".join(parts)



# Metadata header parser (used by Layer 3 Check 5c)

# Pattern matching the header produced by ``annotate_search_output()``. The
# ``freshness_status`` field is optional so that headers written before that
# field was introduced can still be parsed.

_META_RE = re.compile(
    r"\[web_search_meta\s+"
    r"retrieved_at=(?P<retrieved_at>\S+)\s+"
    r"data_type=(?P<data_type>\S+)\s+"
    r"stale=(?P<stale>\S+)\s+"
    r"age_h=(?P<age_h>\S+)"
    r"(?:\s+freshness_status=(?P<freshness_status>\S+))?"
    r"\]"
)


def parse_search_meta(text: str) -> Optional[dict]:
    """
    Extract the ``[web_search_meta ...]`` header injected by
    ``annotate_search_output()``.

    Returns a dict with keys: retrieved_at (str), data_type (str),
    stale (bool), age_h (float|None), freshness_status (str).
    When the header has no ``freshness_status`` field, the value is inferred
    from ``stale`` and ``age_h`` ('stale', 'unknown' or 'fresh').
    Returns None when no header is found.
    """
    m = _META_RE.search(text)
    if not m:
        return None
    stale_raw  = m.group("stale").lower()
    age_raw    = m.group("age_h")
    fs_raw     = m.group("freshness_status")
    return {
        "retrieved_at":    m.group("retrieved_at"),
        "data_type":       m.group("data_type"),
        "stale":           stale_raw == "true",
        "age_h":           float(age_raw) if age_raw != "unknown" else None,
        "freshness_status": fs_raw if fs_raw else (
            "stale" if stale_raw == "true" else
            "unknown" if age_raw == "unknown" else "fresh"
        ),
    }
