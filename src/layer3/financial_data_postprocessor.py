"""

AMAN-LLM
Layer 3 — Output Validator Orchestrator
Financial Price Data Sanity and Jurisdiction Filter
(src/layer3/financial_data_postprocessor.py; internal identifier P1-4)


PURPOSE
    Applies rule-based plausibility checks to web-search output returned for
    price-related queries (precious metals, oil, Bitcoin, selected exchange
    rates, and a small set of everyday consumer goods). The checks are
    intended to reduce the risk that a figure with an implausible magnitude,
    a mismatched currency or unit, or an outdated timestamp is passed on
    without being flagged. The module does not verify prices against an
    authoritative source.

ARCHITECTURAL ROLE
    The module belongs to Layer 3 (Output Validator Orchestrator) and
    post-processes tool output before it is used to compose an answer. It
    only reports its findings, through FinancialPostProcessResult. Acting on
    them (re-querying, correcting a figure, or stating that a price could not
    be verified) is left to the calling Layer 3 component and to Layer 4
    (Constitutional Self-Critique); that calling code is not part of this file.

INPUTS
    tool_output     Text returned by the web_search tool.
    original_query  The user query that triggered the search.
    jurisdiction    Jurisdiction label (default "Algeria"). The jurisdiction
                    filter and the DZD-aware gold check apply to Algeria.

OUTPUTS
    A tuple (processed_text, FinancialPostProcessResult). processed_text is
    the tool output after jurisdiction filtering, with Arabic notices
    prepended when stale data or a probable unit ambiguity is reported. The
    result object carries the pass/fail status, the raised flags, the detected
    prices, and re-query information.

PROCESSING STEPS (see postprocess_financial_data() for details)
    1. Jurisdiction filter   - drop lines quoting only other regional currencies.
    2. Freshness check       - flag data older than DATA_MAX_AGE_HOURS.
    3. Commodity bounds      - compare USD-like prices with COMMODITY_BOUNDS;
                               gold additionally receives a DZD/gram check.
    4. Consumer-goods bounds - compare DZD amounts with STAPLE_GOODS_DZD_BOUNDS.

CALLER CONTRACT (unit mismatch)
    When the result has hard_fail=True and unit_mismatch_detected=True, the
    caller is expected not to accept the price with a disclaimer alone. It
    should either replace the figure with a verified USD/troy-oz value or state
    explicitly that the price could not be verified. This contract is
    documented here; it is not enforced by this module.

IMPLEMENTATION NOTES
    - Rule-based only: regular expressions and fixed numeric bounds. The
      module makes no model calls and no network requests.
    - Arabic and Latin text are supported; Arabic-Indic digits are normalised
      and selected Arabic number words (e.g. "مليون") are interpreted.
    - The bounds, the DZD/USD rate and the DZD/gram range are static,
      approximate constants (mid-2026 estimates). They serve coarse sanity
      checks and are not market data.
    - Detection is heuristic and covers selected phrasings only; it can miss
      valid price statements and can flag acceptable ones.
    - Diagnostics are emitted through the standard logging module. The module
      does not write to the Audit Logger directly.

REPRODUCIBILITY
    Behaviour is determined by the module-level constants and compiled
    patterns below. The freshness check and the refined query depend on the
    current UTC time (datetime.now), so results for identical inputs can
    differ across execution dates. Typical use:
        processed, result = postprocess_financial_data(
            tool_output, original_query, jurisdiction="Algeria")

REVISION NOTES
    b31cf026   Unit-aware gold check. A gold query returned values of roughly
               139-145 that were not USD/troy-oz, apparently DZD/gram figures
               from an Algerian source; the earlier hard failure did not flag
               the unit ambiguity. The check and the unit_mismatch_detected
               flag were added in response.
    financial-arabic-boundary, financial-missing-dollar,
    financial-regex-boundary, financial-staple-goods, financial-dzd-regex,
    financial-word-numbers
               Later corrections, described next to the code they concern.

"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Configuration


# Maximum accepted age of retrieved price data, in hours. Data whose newest
# detectable date is older than this value, or that carries no parsable date,
# is flagged as stale by _is_stale().
DATA_MAX_AGE_HOURS: float = 24.0

# Coarse plausibility bounds per commodity:
#     name -> (lower_bound, upper_bound, unit_label)
# The bounds are static sanity limits intended to reject gross magnitude or
# unit errors (for example a local-currency value reported as USD). They are
# not market-price estimates and are not updated at runtime. unit_label is
# used only in flag and log messages. The "euro" and "dollar" entries hold
# exchange-rate-style ranges.
COMMODITY_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "gold":    (1_800.0, 5_500.0, "USD/oz"),
    "silver":  (    10.0,   300.0, "USD/oz"),
    "oil":     (     5.0,   400.0, "USD/barrel"),
    "bitcoin": (  1_000.0, 1_000_000.0, "USD"),
    "euro":    (     0.5,     2.5, "USD/EUR"),
    "dollar":  (     0.8,     1.3, "USD/USD"),
}


# Unit-detection constants for gold prices (revision b31cf026)
#
# Algerian market sources commonly quote gold in DZD per gram, whereas
# COMMODITY_BOUNDS expects USD per troy ounce. The constants below support a
# heuristic that tests whether an out-of-bounds gold value could be a DZD/gram
# figure and, if so, converts it to USD/troy-oz for comparison with the bounds.
#
# Order-of-magnitude reference for the DZD/gram range (approximate mid-2026
# estimate): 3,300 USD/oz / 31.1035 g/oz * 135 DZD/USD ~= 14,300 DZD/gram.
#
# The constants are approximations intended only for the sanity check. The
# exchange rate for any figure presented to the user is expected to come from a
# separate tool.


# Gold values (read as USD) below this threshold are treated as suspicious:
# they fall below the COMMODITY_BOUNDS range for USD/troy-oz and may be
# local-currency or per-gram figures.
GOLD_LIKELY_DZD_GRAM_THRESHOLD = 500.0   # USD — values below this are suspicious

# Assumed plausible range of the gold price in DZD per gram (rough 2026 bounds
# based on ~3,300 USD/oz and ~135 DZD/USD). A candidate value is converted to
# USD/oz only if it lies inside this range.
GOLD_DZD_GRAM_MIN  =  6_000.0
GOLD_DZD_GRAM_MAX  = 30_000.0

# Approximate conversion from DZD/gram to USD/troy-oz:
#     usd_per_oz = (dzd_per_gram / _DZD_USD_APPROX) * _TROY_OZ_GRAMS
_DZD_USD_APPROX  = 135.0   # approximate DZD per 1 USD (mid-2026 estimate)
_TROY_OZ_GRAMS   = 31.1034768


# Identifiers (ISO codes, Arabic names, Arabic abbreviations) of other regional
# currencies. When the jurisdiction is Algeria, a line that contains a number
# and one of these identifiers but no anchor currency (below) is treated as
# jurisdictionally irrelevant by _filter_irrelevant_jurisdiction_snippets().
_IRRELEVANT_CURRENCIES_DZ: List[str] = [
    "EGP", "جنيه مصري", "ج.م",
    "SAR", "ريال سعودي", "ر.س",
    "AED", "درهم إماراتي", "د.إ",
    "KWD", "دينار كويتي", "د.ك",
    "QAR", "ريال قطري", "ر.ق",
    "BHD", "دينار بحريني", "د.ب",
    "OMR", "ريال عماني", "ر.ع",
    "JOD", "دينار أردني", "د.أ",
    "LBP", "ليرة لبنانية", "ل.ل",
    "MAD", "درهم مغربي", "د.م",
    "TND", "دينار تونسي", "د.ت",
    "LYD", "دينار ليبي", "د.ل",
]

# Anchor currencies (DZD, USD, EUR: ISO codes, Arabic names and symbols). A
# line containing any of these is retained even if it also mentions a currency
# from the list above.
_ANCHOR_CURRENCIES_DZ: List[str] = [
    "DZD", "دج", "دينار جزائري",
    "USD", "دولار", "$", "أمريكي",
    "EUR", "يورو", "€",
]

# Generic price pattern used by _extract_prices(). It matches a number-like
# token (digits, thousands separators, decimal point, spaces) followed by a
# currency code or Arabic currency word, plus up to 30 trailing characters that
# do not end a sentence. Group 1 is the numeric token and group 2 the currency
# token. Matches are reported in FinancialPostProcessResult.detected_prices and
# are not used by the bounds checks.
_PRICE_RE = re.compile(
    r"([\d,،. ]+)\s*"
    r"(USD|EUR|DZD|EGP|SAR|AED|KWD|QAR|BHD|OMR|JOD|LBP|MAD|TND|LYD"
    r"|دولار|يورو|دج|جنيه|ريال|درهم|دينار|ليرة)"
    r"(?:[^؟!.\n]{0,30})?",
    re.IGNORECASE | re.UNICODE,
)

def _bounded_alternation(keywords: List[str]) -> re.Pattern:
    """
    Compile a case-insensitive pattern that matches any of *keywords* as a
    whole token, for both Latin and Arabic text.

    The pattern uses the look-around assertions (?<!\\w) and (?!\\w) instead of
    \\b, and allows an optional attached Arabic definite article ("ال") before
    the keyword (revision "financial-arabic-boundary"). Arabic letters are \\w
    characters, so \\b finds no boundary inside "الدولار" and a \\b-wrapped
    "دولار" would not match it, although this attached-article form is very
    common.

    \\w (Unicode-aware in Python 3) is used instead of an explicit
    \\u0600-\\u06FF range because that range also contains Arabic punctuation
    (e.g. "؟", "،", "؛"). Treating punctuation as part of a word would break
    the trailing boundary for queries that end with an Arabic question mark.

    Parameters
  
    keywords : list of str
        Literal keywords; each is passed through re.escape().

    Returns
   --
    re.Pattern
    """
    parts = "|".join(re.escape(k) for k in keywords)
    return re.compile(
        rf"(?<!\w)(?:ال)?(?:{parts})(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )


# Query-side detectors: commodity name -> compiled keyword pattern (Latin and
# Arabic keywords, matched as whole tokens). _detect_commodity() returns the
# first key, in the insertion order of this mapping, whose pattern matches the
# query, so that order resolves queries mentioning several commodities.
_COMMODITY_QUERY_RE: Dict[str, re.Pattern] = {
    name: _bounded_alternation(keywords)
    for name, keywords in {
        "gold":    ["gold", "ذهب", "أونصة", "جرام ذهب"],
        "silver":  ["silver", "فضة", "argent"],
        "oil":     ["oil", "نفط", "برميل", "brent", "crude", "petrole"],
        "bitcoin": ["bitcoin", "بيتكوين", "btc", "crypto"],
        "euro":    ["euro", "يورو", "eur", "€"],
        # "dollar" is included so that every key of COMMODITY_BOUNDS has a query
        # detector (revision "financial-missing-dollar"); a query such as
        # "كم سعر الدولار؟" is then recognised as a commodity query.
        "dollar":  ["dollar", "دولار", "usd"],
    }.items()
}

# Note on the gold keywords (revision "financial-regex-boundary"): the bare
# French word "or" is intentionally not included. An earlier version matched
# it without token boundaries, so it also matched inside unrelated words (e.g.
# "Word", "before", "story") and could trigger the freshness disclaimer on
# non-financial queries. The remaining gold keywords are retained.


# Consumer-goods plausibility check (revision "financial-staple-goods")
#
# Everyday goods (bread, milk, sugar, gasoline, a car, a phone, a laptop) are
# not forex or metal commodities, so no USD/oz conversion applies. DZD amounts
# extracted from the retrieved text are compared directly with coarse DZD
# plausibility ranges. The check targets grossly implausible figures observed
# in the test and production data (e.g. a car priced at "100 دينار").


# name -> (lower_bound_DZD, upper_bound_DZD). Coarse, static ranges; they are
# not market statistics.
STAPLE_GOODS_DZD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "bread":    (15.0, 100.0),
    "milk":     (40.0, 300.0),
    "sugar":    (80.0, 400.0),
    "gasoline": (30.0, 200.0),
    "car":      (800_000.0, 30_000_000.0),
    "phone":    (3_000.0, 500_000.0),
    "laptop":   (15_000.0, 600_000.0),
}

# Query-side detectors for the consumer goods above (same matching rules as
# _COMMODITY_QUERY_RE).
_STAPLE_GOODS_QUERY_RE: Dict[str, re.Pattern] = {
    name: _bounded_alternation(keywords)
    for name, keywords in {
        "bread":    ["خبز", "رغيف"],
        "milk":     ["حليب"],
        "sugar":    ["سكر"],
        "gasoline": ["بنزين", "gasoline"],
        "car":      ["سيارة"],
        "phone":    ["هاتف", "جوال", "موبايل"],
        "laptop":   ["لابتوب", "حاسوب", "كمبيوتر"],
    }.items()
}


def _detect_staple_good(query: str) -> Optional[str]:
    """Return the first consumer-good key whose pattern matches *query*, or None."""
    for name, pat in _STAPLE_GOODS_QUERY_RE.items():
        if pat.search(query):
            return name
    return None


# Arabic number words used as price multipliers (revision
# "financial-word-numbers"). Prices spelled out in words (e.g. "مليون دولار",
# one million dollars) contain no digits and were previously not extracted.
# The mapping gives the numeric value of the multiplier words that occur in
# price statements; _AR_PRICE_WORD_RE below matches them as whole words.
_AR_PRICE_WORD_MULTIPLIERS: Dict[str, float] = {
    "مليون": 1_000_000.0, "ملايين": 1_000_000.0,
    "مليار": 1_000_000_000.0, "مليارات": 1_000_000_000.0,
    "ألف": 1_000.0, "آلاف": 1_000.0,
}
_AR_PRICE_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _AR_PRICE_WORD_MULTIPLIERS) + r")\b",
    re.UNICODE,
)

# Variant that also accepts "واحد"/"واحدة" (one). These words commonly follow
# the currency word in Arabic price idioms (e.g. "بدينار واحد", "for one
# dinar"), whereas the multiplier words above precede it (revision
# "financial-word-numbers").
_AR_PRICE_WORD_MULTIPLIERS_WITH_ONE: Dict[str, float] = {
    **_AR_PRICE_WORD_MULTIPLIERS,
    "واحد": 1.0, "واحدة": 1.0,
}
_AR_PRICE_WORD_RE_WITH_ONE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _AR_PRICE_WORD_MULTIPLIERS_WITH_ONE) + r")\b",
    re.UNICODE,
)

# Date patterns used by _is_stale(): numeric D/M/Y or M/D/Y with "/" or "-"
# separators (2- or 4-digit year), Y-M-D, "Mon D, YYYY", and "D Month YYYY"
# with English month names. Only the full match text (group 0) is consumed.
_DATE_RE = re.compile(
    r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"
    r"|\b(\d{4})[/-](\d{2})[/-](\d{2})"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"\s*\d{1,2},?\s*\d{4}"
    r"|\b(\d{1,2})\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)

# Textual signals (currency markers and Algeria-related terms) that the
# retrieved text quotes prices in DZD (Algerian dinar). Used by the gold unit
# check to decide whether the context is Algerian.
_DZD_SIGNALS_RE = re.compile(
    r"(دج|DZD|دينار جزائري|دينار\s+جزائري|الجزائر|السوق الجزائري"
    r"|بورصة الجزائر|السعر في الجزائر)",
    re.IGNORECASE | re.UNICODE,
)

# Textual signals that a price is quoted per gram rather than per troy ounce.
# Currently evaluated in postprocess_financial_data() but not used in the
# conversion decision.
_GRAM_SIGNALS_RE = re.compile(
    r"\b(جرام|غرام|gram|gramme|g\b|per\s+gram|للجرام|لكل\s+جرام)\b",
    re.IGNORECASE | re.UNICODE,
)



# Output dataclass


@dataclass
class FinancialPostProcessResult:
    """
    Structured outcome of postprocess_financial_data().

    Attributes
  
    passed : bool
        True when no flag was raised (``flags`` is empty). Soft flags (stale
        data, jurisdiction filtering, a converted unit) also set it to False
        even when ``hard_fail`` is False.
    hard_fail : bool
        True when a price was found outside its plausibility bounds.
    requery_needed : bool
        True when a new search is suggested; see ``refined_query``.
    refined_query : str
        Suggested replacement query (empty when no re-query is needed).
    processed_text : str
        Tool output after jurisdiction filtering, with any notices prepended.
    flags : list of str
        Machine-readable descriptions of each raised condition.
    commodity : str or None
        Detected commodity or consumer-good key, if any.
    detected_prices : list of (str, str)
        (numeric token, currency token) pairs found by _PRICE_RE; reported
        for inspection and not used in the bounds checks.
    stale : bool
        True when the freshness check flagged the data.
    jurisdiction_filtered_snippets : int
        Number of lines removed by the jurisdiction filter.
    unit_mismatch_detected : bool
        True when a hard failure was attributed to a suspected unit mismatch.
    """
    passed:                           bool
    hard_fail:                        bool       = False
    requery_needed:                   bool       = False
    refined_query:                    str        = ""
    processed_text:                   str        = ""
    flags:                            List[str]  = field(default_factory=list)
    commodity:                        Optional[str] = None
    detected_prices:                  List[Tuple[str, str]] = field(default_factory=list)
    stale:                            bool       = False
    jurisdiction_filtered_snippets:   int        = 0
    # True when the hard failure was attributed to a suspected unit mismatch
    # (e.g. a DZD/gram value compared with USD/oz bounds; revision b31cf026).
    # Callers are expected to correct the value or to state that it could not
    # be verified (see the caller contract in the module documentation).
    unit_mismatch_detected:           bool       = False



# Public entry point


def postprocess_financial_data(
    tool_output:   str,
    original_query: str,
    jurisdiction:  str = "Algeria",
) -> Tuple[str, FinancialPostProcessResult]:
    """
    Run the financial-data sanity checks on retrieved tool output.

    Processing steps, in order:
      1. Jurisdiction filter: if *jurisdiction* is Algeria, remove lines that
         contain a number and a currency of another regional jurisdiction but
         no anchor currency (DZD, USD, EUR).
      2. Freshness check: if the query matched a commodity or consumer good,
         flag the data as stale when no date can be parsed or the newest date
         is older than DATA_MAX_AGE_HOURS, and prepend an Arabic warning.
      3. Commodity bounds: compare USD-like prices extracted from
         *tool_output* with COMMODITY_BOUNDS.
      4. Consumer-goods bounds: compare DZD amounts extracted from
         *tool_output* with STAPLE_GOODS_DZD_BOUNDS.

    Unit-aware gold check (step 3, revision b31cf026). For gold, an
    out-of-bounds value below GOLD_LIKELY_DZD_GRAM_THRESHOLD is treated as a
    possible DZD/gram price when the jurisdiction is Algeria or the text
    contains DZD signals.
      - If the value lies in [GOLD_DZD_GRAM_MIN, GOLD_DZD_GRAM_MAX], it is
        converted to USD/troy-oz with the approximate DZD/USD rate. A
        converted value within the bounds raises a soft flag and prepends a
        unit-clarification notice (no hard failure).
      - Otherwise a hard failure is raised with unit_mismatch_detected=True.
    Out-of-bounds values outside this scenario raise a hard failure without
    that flag.

    Parameters
   -----
    tool_output : str
        Text returned by the web_search tool. Empty input, or input starting
        with "TOOL_ERROR:", is returned unchanged with passed=True.
    original_query : str
        User query, used to detect the commodity or consumer good.
    jurisdiction : str
        Jurisdiction label; "Algeria", "Algérie" and "الجزائر" enable the
        jurisdiction filter and the DZD-aware gold check.

    Returns
   --
    (str, FinancialPostProcessResult)
        The processed text and the structured result.

    Notes
   
    Date detection and price extraction (steps 2-4) operate on the original
    *tool_output*, not on the jurisdiction-filtered text.
    """
    if not tool_output or tool_output.startswith("TOOL_ERROR:"):
        return tool_output, FinancialPostProcessResult(
            passed=True, processed_text=tool_output
        )

    flags:           List[str] = []
    processed        = tool_output
    hard_fail        = False
    requery_needed   = False
    refined_query    = ""
    commodity        = _detect_commodity(original_query)
    # Consumer goods (revision "financial-staple-goods") are detected only when
    # no commodity matched. _detect_commodity() recognises only the keys of
    # _COMMODITY_QUERY_RE, so implausible DZD prices for everyday goods (e.g. a
    # car at "100 دينار") would otherwise bypass the sanity checks below.
    staple_good      = _detect_staple_good(original_query) if not commodity else None
    detected_prices  = _extract_prices(tool_output)
    filtered_count   = 0
    unit_mismatch_detected = False

    # ── Step 1: Jurisdiction relevance filter ─────────────────────────────
    # Applied only for Algeria. Removes lines that quote other regional
    # currencies without an anchor currency. The filter changes the returned
    # text only; later steps analyse the original tool_output.
    if jurisdiction.lower() in ("algeria", "algérie", "الجزائر"):
        processed, filtered_count = _filter_irrelevant_jurisdiction_snippets(processed)
        if filtered_count > 0:
            flags.append(f"jurisdiction_filter:{filtered_count}_snippets_removed")
            logger.warning(
                "P1-4 jurisdiction_filter | %d snippet(s) removed",
                filtered_count,
            )

    # ── Step 2: Freshness check ────────────────────────────────────────────
    # Evaluated only for queries recognised as commodity or consumer-good price
    # queries. An Arabic warning is prepended to the text when data is stale.
    stale = _is_stale(tool_output) if (commodity or staple_good) else False
    if stale:
        flags.append("stale_data:>24h")
        disclaimer = (
            "⚠️ تحذير: البيانات السعرية المسترجعة قد تكون قديمة (أكثر من 24 ساعة). "
            "يُرجى التحقق من المصدر الرسمي للسعر الحالي.\n\n"
        )
        processed = disclaimer + processed
        logger.warning("P1-4 freshness_check | stale data detected")

    # ── Step 3: Commodity sanity bounds (unit-aware) ──────────────────────
    # Applies to commodities listed in COMMODITY_BOUNDS. Each price extracted
    # from the original tool output is compared with the bounds; out-of-bounds
    # gold values are first tested for a possible DZD/gram unit mismatch.
    if commodity and commodity in COMMODITY_BOUNDS:
        min_val, max_val, unit = COMMODITY_BOUNDS[commodity]
        usd_prices = _extract_usd_prices(tool_output, commodity)

        for raw_str, usd_val in usd_prices:
            if min_val <= usd_val <= max_val:
                continue  # within bounds — no action needed

            # Out-of-bounds value: test for a possible unit mismatch before
            # treating it as a plain magnitude error (revision b31cf026).
            _unit_mismatch = False
            _converted_usd: Optional[float] = None

            if commodity == "gold" and usd_val < GOLD_LIKELY_DZD_GRAM_THRESHOLD:
                # Gold value below the suspicion threshold: it may be a DZD/gram
                # figure; the context signals below decide whether to test that.
                is_dzd_context = (
                    jurisdiction.lower() in ("algeria", "algérie", "الجزائر")
                    or bool(_DZD_SIGNALS_RE.search(tool_output))
                )
                is_gram_context = bool(_GRAM_SIGNALS_RE.search(tool_output))

                if is_dzd_context:
                    # Conversion is attempted even without an explicit gram
                    # signal, because Algerian gold markets quote in DZD/gram
                    # by default.
                    _unit_mismatch = True
                    # If the value lies in the assumed DZD/gram range, convert
                    # it to USD/troy-oz for comparison with the bounds.
                    if GOLD_DZD_GRAM_MIN <= usd_val <= GOLD_DZD_GRAM_MAX:
                        # Interpret the raw value as DZD/gram (approximate rate).
                        _converted_usd = (usd_val / _DZD_USD_APPROX) * _TROY_OZ_GRAMS
                        logger.info(
                            "P1-4 unit_aware | commodity=gold | raw_val=%.2f "
                            "interpreted as DZD/gram | converted=%.2f USD/oz",
                            usd_val, _converted_usd,
                        )

            if _unit_mismatch:
                if _converted_usd is not None and min_val <= _converted_usd <= max_val:
                    # Converted value lies within the bounds: soft flag only.
                    flags.append(
                        f"unit_mismatch_converted:{commodity}={usd_val:.0f}DZD/gram"
                        f"→{_converted_usd:.0f}USD/oz (within_bounds)"
                    )
                    logger.warning(
                        "P1-4 unit_mismatch | commodity=%s | DZD/gram value %.2f "
                        "converted to %.2f USD/oz — WITHIN bounds; flagging unit "
                        "ambiguity in answer",
                        commodity, usd_val, _converted_usd,
                    )
                    # Prepend an Arabic notice stating that the value appears to
                    # be DZD per gram and giving the approximate USD/oz figure.
                    unit_disclaimer = (
                        f"⚠️ ملاحظة: السعر الوارد ({usd_val:,.0f}) يبدو بالدينار الجزائري "
                        f"للجرام، وليس بالدولار للأونصة. "
                        f"السعر التقريبي المحوّل: {_converted_usd:,.0f} دولار/أونصة.\n\n"
                    )
                    processed = unit_disclaimer + processed
                else:
                    # Conversion not possible, or converted value still outside
                    # the bounds: hard failure with the unit-mismatch flag set.
                    flags.append(
                        f"price_out_of_bounds:{commodity}={usd_val:.0f}{unit}"
                        f"(unit_mismatch_suspected;expected {min_val:.0f}–{max_val:.0f})"
                    )
                    unit_mismatch_detected = True
                    requery_needed  = True
                    refined_query   = _build_refined_query(original_query, commodity)
                    hard_fail       = True
                    logger.error(
                        "P1-4 price_sanity+unit_mismatch | commodity=%s raw='%s' "
                        "value=%.2f likely_unit=DZD/gram converted=%s "
                        "outside bounds [%.0f,%.0f] USD/oz",
                        commodity, raw_str, usd_val,
                        f"{_converted_usd:.2f}" if _converted_usd else "N/A",
                        min_val, max_val,
                    )
            else:
                # Out-of-bounds value with no unit-mismatch hypothesis: hard failure.
                flags.append(
                    f"price_out_of_bounds:{commodity}={usd_val:.0f}{unit}"
                    f"(expected {min_val:.0f}–{max_val:.0f})"
                )
                requery_needed  = True
                refined_query   = _build_refined_query(original_query, commodity)
                hard_fail       = True
                logger.error(
                    "P1-4 price_sanity | commodity=%s raw='%s' value=%.2f "
                    "outside bounds [%.0f, %.0f] %s",
                    commodity, raw_str, usd_val, min_val, max_val, unit,
                )

    # ── Step 4: Consumer-goods DZD sanity ──────────────────────────────────
    # Applies when the query matched a consumer good (and no commodity). Each
    # DZD amount extracted from the original tool output is compared directly
    # with STAPLE_GOODS_DZD_BOUNDS; no currency conversion is involved.
    if staple_good and staple_good in STAPLE_GOODS_DZD_BOUNDS:
        min_v, max_v = STAPLE_GOODS_DZD_BOUNDS[staple_good]
        dzd_vals = _extract_dzd_values(tool_output)
        for raw_str, val in dzd_vals:
            if min_v <= val <= max_v:
                continue
            flags.append(
                f"price_out_of_bounds:{staple_good}={val:.0f}DZD"
                f"(expected {min_v:.0f}–{max_v:.0f})"
            )
            requery_needed = True
            refined_query  = _build_refined_query(original_query, staple_good)
            hard_fail      = True
            logger.error(
                "P1-4 price_sanity | good=%s raw='%s' value=%.2f DZD "
                "outside plausible bounds [%.0f, %.0f]",
                staple_good, raw_str, val, min_v, max_v,
            )
        # Report the consumer good in the result's commodity field.
        commodity = commodity or staple_good

    # passed is True only when no flag, soft or hard, was raised.
    passed = len(flags) == 0

    return processed, FinancialPostProcessResult(
        passed                         = passed,
        hard_fail                      = hard_fail,
        requery_needed                 = requery_needed,
        refined_query                  = refined_query,
        processed_text                 = processed,
        flags                          = flags,
        commodity                      = commodity,
        detected_prices                = detected_prices,
        stale                          = stale,
        jurisdiction_filtered_snippets = filtered_count,
        unit_mismatch_detected         = unit_mismatch_detected,
    )



# Internal helpers


def _detect_commodity(query: str) -> Optional[str]:
    """Return the first commodity key whose query pattern matches *query*, or None."""
    for name, pat in _COMMODITY_QUERY_RE.items():
        if pat.search(query):
            return name
    return None


def _extract_prices(text: str) -> List[Tuple[str, str]]:
    """
    Return (numeric_token, currency_token) pairs found by _PRICE_RE in *text*.

    The values are raw strings (not normalised). They are used only for
    reporting in FinancialPostProcessResult.detected_prices and do not enter
    the bounds checks.
    """
    results = []
    for m in _PRICE_RE.finditer(text):
        val = m.group(1) or ""
        cur = m.group(2) or ""
        if val.strip():
            results.append((val.strip(), cur.strip()))
    return results


def _normalise_number(s: str) -> float:
    """
    Parse a numeric string into a float; return NaN on failure.

    Arabic-Indic digits are mapped to ASCII digits and the thousands
    separators "," and "،" are removed. If the string is not numeric, a bare
    Arabic multiplier word (e.g. "مليون" -> 1,000,000) is accepted as a
    fallback. The fallback was added in revision "financial-word-numbers":
    amounts stated only in words (e.g. "مليون دولار") previously produced NaN
    and were dropped before the bounds check.
    """
    return _normalise_number_with(s, _AR_PRICE_WORD_RE)


def _normalise_number_with(s: str, word_re: re.Pattern) -> float:
    """
    Implementation of _normalise_number() with a selectable number-word pattern.

    *word_re* is either _AR_PRICE_WORD_RE (multiplier words) or
    _AR_PRICE_WORD_RE_WITH_ONE (multiplier words plus "واحد"/"واحدة"); the
    matching lookup table is chosen by identity of *word_re*. Returns NaN when
    neither a numeric nor a word form can be parsed.
    """
    arabic_indic = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    s = s.translate(arabic_indic)
    s = s.replace("،", "").replace(",", "")
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    word_m = word_re.search(s)
    if word_m:
        table = (
            _AR_PRICE_WORD_MULTIPLIERS_WITH_ONE
            if word_re is _AR_PRICE_WORD_RE_WITH_ONE
            else _AR_PRICE_WORD_MULTIPLIERS
        )
        return table.get(word_m.group(0), float("nan"))
    return float("nan")


def _extract_usd_prices(text: str, commodity: str) -> List[Tuple[str, float]]:
    """
    Return (raw_string, value) pairs for price candidates found in *text*.

    Three extraction passes are performed; number words (e.g. "مليون") are
    accepted in place of digits:
      1. a number followed by a USD marker ("USD", "دولار", "$", "أمريكي");
      2. a USD marker followed by a number;
      3. a number followed by a DZD marker ("دج", "DZD", "دينار", "دنانير",
         optionally followed by "جزائري(ة)").

    Pass 3 was added in revision b31cf026 so that DZD/local-currency figures
    in Algerian sources are also examined. Values from all passes are returned
    in one list without a currency tag, and the list is deduplicated by
    (raw_string, value). The *commodity* argument is currently not used.
    """
    results = []
    usd_pattern = re.compile(
        r"([\d٠-٩,،. ]{2,12}|مليون|ملايين|مليار|مليارات|ألف|آلاف)"
        r"\s*(?:USD|دولار|\$|أمريكي)",
        re.IGNORECASE | re.UNICODE,
    )
    usd_pattern_rev = re.compile(
        r"(?:USD|دولار|\$|أمريكي)\s*([\d٠-٩,،. ]{2,12}|مليون|ملايين|مليار|مليارات|ألف|آلاف)",
        re.IGNORECASE | re.UNICODE,
    )
    for pat in (usd_pattern, usd_pattern_rev):
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            val = _normalise_number(raw)
            if not (val != val):
                results.append((raw, val))

    # Pass 3 (revision b31cf026): numbers followed by a DZD marker, e.g.
    # "سعر الذهب: 14500 دج/جرام", which would otherwise be compared with USD
    # bounds without any unit context. Revision "financial-dzd-regex": the
    # earlier pattern required the literal phrase "دينار جزائري"; the plain
    # "دينار", the plural "دنانير" and an optional "جزائري(ة)" qualifier are now
    # accepted, as are word-based amounts (e.g. "مليون دينار").
    dzd_price_pattern = re.compile(
        r"([\d٠-٩,،. ]{1,12}|مليون|ملايين|مليار|مليارات|ألف|آلاف)\s*"
        r"(?:دج|DZD|دينار(?:\s*جزائري[ةه]?)?|دنانير(?:\s*جزائري[ةه]?)?)",
        re.IGNORECASE | re.UNICODE,
    )
    for m in dzd_price_pattern.finditer(text):
        raw = m.group(1).strip()
        val = _normalise_number(raw)
        if not (val != val) and val >= 0:
            # DZD-denominated value: appended unchanged. No currency tag is
            # attached and no conversion is performed in this function.
            results.append((raw, val))

    # Deduplicate by (raw_str, val) to avoid double-counting
    seen: set = set()
    deduped = []
    for pair in results:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)
    return deduped


def _extract_dzd_values(text: str) -> List[Tuple[str, float]]:
    """
    Return (raw_string, value) pairs for DZD-denominated amounts in *text*.

    Standalone counterpart of the DZD pass in _extract_usd_prices(), used for
    the consumer-goods check, where no commodity bounds-conversion path exists
    and the DZD figure is compared directly with STAPLE_GOODS_DZD_BOUNDS.
    Two orders are handled: number before currency, and currency before
    number (see the reverse-order pattern below). Results are deduplicated.
    """
    dzd_price_pattern = re.compile(
        r"([\d٠-٩,،. ]{1,12}|مليون|ملايين|مليار|مليارات|ألف|آلاف)\s*"
        r"(?:دج|DZD|دينار(?:\s*جزائري[ةه]?)?|دنانير(?:\s*جزائري[ةه]?)?)",
        re.IGNORECASE | re.UNICODE,
    )
    # Reverse-order idiom (revision "financial-word-numbers"): e.g.
    # "بدينار واحد" ("for one dinar") states the number after the currency word.
    dzd_price_pattern_rev = re.compile(
        r"(?:دج|DZD|دينار(?:\s*جزائري[ةه]?)?|دنانير(?:\s*جزائري[ةه]?)?)\s*"
        rf"({_AR_PRICE_WORD_RE_WITH_ONE.pattern}|[\d٠-٩,،. ]{{1,12}})",
        re.IGNORECASE | re.UNICODE,
    )
    results: List[Tuple[str, float]] = []
    for pat, word_re in (
        (dzd_price_pattern, _AR_PRICE_WORD_RE),
        (dzd_price_pattern_rev, _AR_PRICE_WORD_RE_WITH_ONE),
    ):
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            val = _normalise_number_with(raw, word_re)
            if not (val != val) and val >= 0:
                results.append((raw, val))
    seen: set = set()
    deduped = []
    for pair in results:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)
    return deduped


def _is_stale(text: str) -> bool:
    """
    Return True when the newest date found in *text* is older than DATA_MAX_AGE_HOURS.

    Dates are located with _DATE_RE and parsed as UTC midnight using one of
    four numeric formats. Boundary behaviour:
      - no parsable date found           -> True (treated as stale);
      - newest parsed year before 2000   -> False (treated as a corrupt
        timestamp and ignored);
      - otherwise                        -> age in hours > DATA_MAX_AGE_HOURS.
    The result depends on the current UTC time (datetime.now).
    """
    now = datetime.now(timezone.utc)
    dates_found = []
    for m in _DATE_RE.finditer(text):
        raw = m.group(0)
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                normalised = re.sub(r"[/\- ]+", "-", raw.strip()[:10])
                dt = datetime.strptime(normalised, fmt).replace(tzinfo=timezone.utc)
                dates_found.append(dt)
                break
            except ValueError:
                continue
    if not dates_found:
        logger.debug("P1-4 freshness | no date found — treating as stale")
        return True
    newest = max(dates_found)
    if newest.year < 2000:
        logger.warning(
            "P1-4 freshness | corrupt retrieved_at year=%d — ignoring",
            newest.year,
        )
        return False
    age_hours = (now - newest).total_seconds() / 3600
    return age_hours > DATA_MAX_AGE_HOURS


def _filter_irrelevant_jurisdiction_snippets(text: str) -> Tuple[str, int]:
    """
    Remove lines that quote only other regional currencies.

    A line is removed when it contains a number of at least two digits and an
    identifier from _IRRELEVANT_CURRENCIES_DZ, and contains no identifier from
    _ANCHOR_CURRENCIES_DZ. Matching is case-insensitive substring matching.

    Returns the filtered text and the number of removed lines.
    """
    lines = text.split("\n")
    kept: List[str] = []
    removed = 0
    anchor_re = re.compile(
        "|".join(re.escape(c) for c in _ANCHOR_CURRENCIES_DZ),
        re.IGNORECASE | re.UNICODE,
    )
    irrelevant_re = re.compile(
        "|".join(re.escape(c) for c in _IRRELEVANT_CURRENCIES_DZ),
        re.IGNORECASE | re.UNICODE,
    )
    price_present_re = re.compile(r"[\d٠-٩]{2,}", re.UNICODE)
    for line in lines:
        has_price      = bool(price_present_re.search(line))
        has_anchor     = bool(anchor_re.search(line))
        has_irrelevant = bool(irrelevant_re.search(line))
        if has_price and has_irrelevant and not has_anchor:
            removed += 1
            logger.debug("P1-4 jurisdiction_filter | removed line: %r", line[:80])
            continue
        kept.append(line)
    return "\n".join(kept), removed


def _build_refined_query(original_query: str, commodity: str) -> str:
    """
    Build a replacement search query for a re-query after a failed price check.

    The query combines a commodity-specific description, the current UTC year
    and "USD". Keys without a dedicated label fall back to "<key> price USD".
    The *original_query* argument is currently not used.
    """
    year = datetime.now(timezone.utc).year
    commodity_label = {
        "gold":    "gold price per troy ounce",
        "silver":  "silver price per troy ounce",
        "oil":     "crude oil price per barrel",
        "bitcoin": "Bitcoin price USD",
        "euro":    "EUR/USD exchange rate",
    }.get(commodity, f"{commodity} price USD")
    return f"{commodity_label} live {year} USD"
