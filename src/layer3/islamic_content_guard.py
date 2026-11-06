# src/layer3/islamic_content_guard.py
"""

AMAN-LLM
Layer 3 — Output Validator Orchestrator
Sub-module: Islamic Content Guard


PURPOSE

Domain-specific output check for religious (Islamic) content. The module
examines a candidate LLM answer and detects three classes of problems:

  1. Quranic citations that overlap a reference ayah but deviate from its
     canonical wording (treated as fabricated or altered citations).
  2. Selected structural errors in Hajj rules (e.g. a condition or a sunnah
     act stated as a rukn) detected through regular-expression patterns.
  3. Selected well-known counts stated incorrectly (the pillars of Islam and
     the daily obligatory prayers).

It also reports, as a warning only, when a fiqh query is answered without
any grounded evidence passages.

ARCHITECTURAL ROLE

The module is invoked by the Output Validator Orchestrator (Layer 3) through
``check_islamic_content()``, after the standard Layer 3 checks have run.
Its outcome is mapped by ``output_validator.run_layer3`` onto the Layer 3
decision as follows:

  * ``IslamicFabricationError`` raised  -> l3_passed=False, block=True,
    block_layer='layer3'. The answer is not forwarded to Layer 4
    (Constitutional Self-Critique) for rewriting, because a rewrite loop
    without a verified Quran corpus cannot be relied upon to restore the
    correct text.
  * ``IslamicContentResult.passed == False`` -> reported to Layer 3 through
    ``failed_checks``.
  * ``IslamicContentResult.islamic_no_evidence == True`` -> advisory signal
    that the caller may use to set ``require_tool_evidence=True`` on the
    next invocation.

The module has no dependency on a retrieval (RAG) tool. It relies on a
compact in-process reference corpus (``_CANONICAL_AYAT``) and rule tables
(``_HAJJ_FALSE_ARKAAN``, ``_KNOWN_ISLAMIC_COUNTS``). This is an interim
design: when a full Islamic retrieval tool becomes available, these tables
can be replaced by live retrieval while keeping the interface unchanged.

INPUTS

raw_output          Candidate LLM answer to be validated.
original_query      User query that produced the answer.
evidence_passages   Optional grounded passages from tool results.
content_type        Optional domain label from the query classifier.

OUTPUTS

``IslamicContentResult`` (see the class documentation), or
``IslamicFabricationError`` when a quoted ayah is judged fabricated.

Outcome categories:
  * Exception (IslamicFabricationError) — a quoted text overlaps a canonical
    ayah but its similarity is below ``AYAH_SIMILARITY_THRESHOLD``.
  * Failed check (``passed=False``) — a Hajj structural pattern or a
    known-count error was detected.
  * Warning only — an unverifiable ayah (no overlap with the corpus) or a
    fiqh query without evidence passages.
  * Skipped (``skipped=True``) — no religious-domain signal was found.

IMPLEMENTATION NOTES
--
Domain pre-check. The guard runs only when the query is classified as
religious content, so that non-religious queries (technical, culinary,
greetings, etc.) are returned immediately with ``skipped=True``. Detection
has two tiers:
  1. ``content_type`` supplied by the query classifier (agent core or
     pipeline orchestrator) is compared with ``RELIGIOUS_DOMAINS``.
  2. If it is absent or generic, the query is scanned for Arabic religious
     keywords (``_RELIGIOUS_KEYWORDS``).

Fabricated-ayah policy. When a quoted text matches a canonical ayah with a
similarity below ``AYAH_SIMILARITY_THRESHOLD``, the guard raises
``IslamicFabricationError`` rather than only logging a warning.

REPRODUCIBILITY
---------------
Behaviour is deterministic: it depends only on the input strings, the
constants defined in this file (thresholds, keyword lists, regular
expressions, reference corpus) and the standard-library ``difflib``
implementation of ``SequenceMatcher``. No model calls, network access or
random components are involved. The reference corpus and rule tables are
limited in coverage, so the guard should be interpreted as a targeted
check, not as a complete verification of Islamic content.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Custom exception — raised on fabricated Quran citations


class IslamicFabricationError(ValueError):
    """
    Raised when a quoted Quranic ayah overlaps a canonical ayah but its
    similarity to that ayah is below ``AYAH_SIMILARITY_THRESHOLD``.

    Interface contract: the caller (``output_validator.run_layer3``) is
    expected to catch this exception and set l3_passed=False, block=True,
    block_layer='layer3'. The answer is not to be forwarded to Layer 4
    (Constitutional Self-Critique) for rewriting.
    """



# Thresholds


# Minimum character-level similarity between a quoted ayah and the canonical
# text; values below this threshold cause IslamicFabricationError to be
# raised.
#
# Rationale: after diacritics and letter variants are normalised (_norm), a
# faithful quotation is expected to be nearly character-exact (the original
# development note reports scores of approximately 0.95 or higher). The
# threshold was raised from an earlier value of 0.72, which admitted
# single-word alterations that change the meaning while the remaining text
# is identical (e.g. "ولن يولد" for "ولم يولد", or "يوم الحساب" for
# "يوم الدين"), because fuzzy ratios treat such strings as highly similar.
# The value is empirically chosen, not derived analytically.
AYAH_SIMILARITY_THRESHOLD = 0.93

# Minimum token containment (fraction of one token set found in the other,
# in either direction) for a canonical ayah to be considered a candidate
# match. It acts as a fast pre-filter before the character-level comparison.
#
# Rationale: the earlier value of 0.35 was calibrated for a symmetric
# Jaccard metric. With a containment (recall) metric, and with every
# overlapping ayah now examined, a low value allowed very short ayat (e.g.
# the two-word "الله الصمد") to match on a single frequent word such as
# "الله", producing spurious matches against unrelated ayat. A value of 0.6
# requires most of the words of the ayah to be present.
AYAH_TOKEN_OVERLAP_MIN = 0.6


# Domain routing


#: content_type values (assigned by the query classifier or the pipeline
#: orchestrator) that activate the Islamic guard.
RELIGIOUS_DOMAINS: Set[str] = {
    "fiqh", "theology", "quran", "hadith", "islamic_law",
    "islamic", "religion", "religious",
}

#: Arabic keyword fallback. If any keyword occurs in the query (plain
#: substring match, without word-boundary constraints), the guard is
#: activated regardless of content_type.
_RELIGIOUS_KEYWORDS: List[str] = [
    "إسلام", "قرآن", "فقه", "حلال", "حرام", "ربا", "صلاة", "صلوات",
    "زكاة", "حج", "عمرة", "حديث", "شريعة", "فتوى", "أركان",
    "آية", "آيات", "سورة",
]
_RELIGIOUS_KW_RE = re.compile(
    "|".join(re.escape(kw) for kw in _RELIGIOUS_KEYWORDS),
    re.UNICODE,
)


def _is_religious_domain(content_type: Optional[str], query: str) -> bool:
    """
    Return True when the query belongs to a religious domain.

    Two-tier decision:
      1. The ``content_type`` label from the pipeline classifier is compared
         (case-insensitively) with ``RELIGIOUS_DOMAINS``.
      2. Otherwise, the query is scanned for Arabic religious keywords. This
         covers the cases in which ``content_type`` is None or a generic
         label such as 'factual'.
    """
    if content_type and content_type.lower() in RELIGIOUS_DOMAINS:
        return True
    return bool(_RELIGIOUS_KW_RE.search(query))



# Canonical Quran corpus (compact set of frequently cited ayat)
#
# Stated source: Cairo Mushaf (Hafs ʿan ʿĀsim). The text is stored without
# tashkeel (diacritics), so that diacritic variation in LLM output does not
# produce false mismatches. Keys use the "surah:ayah" format, which matches
# the way LLMs typically cite verses.
#
# Coverage is limited (28 entries in the present version) and some entries
# are excerpts rather than complete ayat (e.g. 2:196, 4:11 and 5:3). The
# table is intended to be extended, or replaced by loading a complete Quran
# text file at import time.


_CANONICAL_AYAT: dict[str, str] = {
    # Al-Baqara
    "2:255":  "الله لا اله الا هو الحي القيوم لا تاخذه سنة ولا نوم له ما في السماوات وما في الارض من ذا الذي يشفع عنده الا باذنه يعلم ما بين ايديهم وما خلفهم ولا يحيطون بشيء من علمه الا بما شاء وسع كرسيه السماوات والارض ولا يوده حفظهما وهو العلي العظيم",
    "2:275":  "الذين ياكلون الربا لا يقومون الا كما يقوم الذي يتخبطه الشيطان من المس ذلك بانهم قالوا انما البيع مثل الربا واحل الله البيع وحرم الربا فمن جاءه موعظة من ربه فانتهى فله ما سلف وامره الى الله ومن عاد فاولئك اصحاب النار هم فيها خالدون",
    "2:286":  "لا يكلف الله نفسا الا وسعها لها ما كسبت وعليها ما اكتسبت ربنا لا تواخذنا ان نسينا او اخطانا ربنا ولا تحمل علينا اصرا كما حملته على الذين من قبلنا ربنا ولا تحملنا ما لا طاقة لنا به واعف عنا واغفر لنا وارحمنا انت مولانا فانصرنا على القوم الكافرين",
    "2:183":  "يا ايها الذين امنوا كتب عليكم الصيام كما كتب على الذين من قبلكم لعلكم تتقون",
    "2:185":  "شهر رمضان الذي انزل فيه القران هدى للناس وبينات من الهدى والفرقان فمن شهد منكم الشهر فليصمه ومن كان مريضا او على سفر فعدة من ايام اخر يريد الله بكم اليسر ولا يريد بكم العسر ولتكملوا العدة ولتكبروا الله على ما هداكم ولعلكم تشكرون",
    "2:196":  "واتموا الحج والعمرة لله فان احصرتم فما استيسر من الهدي ولا تحلقوا رءوسكم حتى يبلغ الهدي محله",
    # Al-Imran
    "3:97":   "فيه ايات بينات مقام ابراهيم ومن دخله كان امنا ولله على الناس حج البيت من استطاع اليه سبيلا ومن كفر فان الله غني عن العالمين",
    "3:185":  "كل نفس ذائقة الموت وانما توفون اجوركم يوم القيامة فمن زحزح عن النار وادخل الجنة فقد فاز وما الحياة الدنيا الا متاع الغرور",
    # An-Nisa
    "4:11":   "يوصيكم الله في اولادكم للذكر مثل حظ الانثيين فان كن نساء فوق اثنتين فلهن ثلثا ما ترك وان كانت واحدة فلها النصف",
    "4:29":   "يا ايها الذين امنوا لا تاكلوا اموالكم بينكم بالباطل الا ان تكون تجارة عن تراض منكم ولا تقتلوا انفسكم ان الله كان بكم رحيما",
    # Al-Maidah
    "5:3":    "حرمت عليكم الميتة والدم ولحم الخنزير وما اهل لغير الله به والمنخنقة والموقوذة والمتردية والنطيحة وما اكل السبع الا ما ذكيتم وما ذبح على النصب وان تستقسموا بالازلام ذلكم فسق",
    "5:90":   "يا ايها الذين امنوا انما الخمر والميسر والانصاب والازلام رجس من عمل الشيطان فاجتنبوه لعلكم تفلحون",
    # Al-Isra
    "17:23":  "وقضى ربك الا تعبدوا الا اياه وبالوالدين احسانا اما يبلغن عندك الكبر احدهما او كلاهما فلا تقل لهما اف ولا تنهرهما وقل لهما قولا كريما",
    # Al-Baqara (additional entry)
    "2:177":  "ليس البر ان تولوا وجوهكم قبل المشرق والمغرب ولكن البر من امن بالله واليوم الاخر والملائكة والكتاب والنبيين واتى المال على حبه ذوي القربى واليتامى والمساكين وابن السبيل والسائلين وفي الرقاب واقام الصلاة واتى الزكاة والموفون بعهدهم اذا عاهدوا والصابرين في الباساء والضراء وحين الباس اولئك الذين صدقوا واولئك هم المتقون",
    # Al-Hujurat
    "49:13":  "يا ايها الناس انا خلقناكم من ذكر وانثى وجعلناكم شعوبا وقبائل لتعارفوا ان اكرمكم عند الله اتقاكم ان الله عليم خبير",
    # Al-Fatiha
    "1:1":    "بسم الله الرحمن الرحيم",
    "1:2":    "الحمد لله رب العالمين",
    "1:3":    "الرحمن الرحيم",
    "1:4":    "مالك يوم الدين",
    "1:5":    "اياك نعبد واياك نستعين",
    "1:6":    "اهدنا الصراط المستقيم",
    "1:7":    "صراط الذين انعمت عليهم غير المغضوب عليهم ولا الضالين",
    # Al-Ikhlas
    "112:1":  "قل هو الله احد",
    "112:2":  "الله الصمد",
    "112:3":  "لم يلد ولم يولد",
    "112:4":  "ولم يكن له كفوا احد",
    # Al-Falaq / An-Nas (first ayah of each surah only)
    "113:1":  "قل اعوذ برب الفلق",
    "114:1":  "قل اعوذ برب الناس",
}


# Hajj reference rules
#
# Arkan (pillars) of Hajj as encoded by this module, following the position
# attributed to the majority (Shafi'i, Hanbali, Maliki) schools:
#   1. Ihram (النية مع الإحرام — the intention is part of ihram and is not a
#      stand-alone rukn)
#   2. Wuquf at Arafah (الوقوف بعرفة)
#   3. Tawaf al-Ifadah (طواف الإفاضة / طواف الزيارة)
#   4. Sa'y (السعي بين الصفا والمروة)
#
# Error types targeted by the patterns below (as observed in LLM outputs):
#   * النية listed as a stand-alone rukn (it is a shart, i.e. a condition)
#   * Pebbles described as collected FROM Jamrat al-Aqaba (they are
#     collected at Muzdalifa)
#   * The talbiyah described as a rukn (it is a sunnah)
#
# Scope note: the rule set encodes a single scholarly position and does not
# attempt to adjudicate legitimate juristic disagreement between schools.


# Patterns whose match is reported as a hajj_structural_error (hard-fail
# item). They are applied to the raw, un-normalised answer text.
_HAJJ_FALSE_ARKAAN = [
    # النية as a stand-alone pillar
    re.compile(r"(ركن|أركان).{0,40}(النية)\b", re.UNICODE),
    re.compile(r"\bالنية\b.{0,20}(ركن|أركان)", re.UNICODE),
    # collecting pebbles FROM the jamarat (incorrect direction); the pattern
    # matches a collection verb followed within 30 characters by a pebble or
    # jamrah term
    re.compile(
        r"(يجمع|تجمع|يأخذ|تأخذ|جمع|أخذ).{0,30}(الحصى|الجمرات|جمرة العقبة)",
        re.UNICODE,
    ),
    # talbiyah as a rukn
    re.compile(r"(ركن|أركان).{0,40}(التلبية)\b", re.UNICODE),
    re.compile(r"\bالتلبية\b.{0,20}(ركن|أركان)", re.UNICODE),
]

# Landmarks whose presence is consistent with a correct Hajj description.
# Informational only: not referenced by the current checks and not used for
# blocking.
_HAJJ_CORRECT_SIGNALS = [
    re.compile(r"الوقوف\s*(بـ|في|ب)?\s*عرفة", re.UNICODE),
    re.compile(r"طواف\s*(الإفاضة|الزيارة)", re.UNICODE),
    re.compile(r"السعي\s*بين\s*الصفا\s*والمروة", re.UNICODE),
    re.compile(r"مزدلفة", re.UNICODE),
]


# Islamic intent patterns


# Broad lexical pattern used by _is_islamic_fiqh_query() on the concatenation
# of query and answer. A match only influences the advisory
# ``islamic_no_evidence`` warning; it does not trigger a failure by itself.
_ISLAMIC_FIQH_PATTERNS = re.compile(
    r"(فقه|شريعة|حلال|حرام|ربا|زكاة|صلاة|حج|عمرة|نكاح|طلاق|ميراث|قرآن|آية|آيات"
    r"|حديث|سنة|إسلام|مسلم|الله|رسول|نبي|فتوى|مذهب|إجماع|اجتهاد"
    r"|صوم|جهاد|وضوء|تيمم|غسل|نجاسة|إحرام|تلبية|مشعر|عرفة|منى|مزدلفة"
    r"|ربا|معاملات|عقود|بيع|قرض|رهن|وقف|وصية|إرث)",
    re.IGNORECASE | re.UNICODE,
)

# Quran citation reference pattern (surah/ayah numbers), e.g.
# "سورة البقرة 275", "البقرة:275" or "2:275". Defined for reference; not
# referenced by the current checks.
_QURAN_CITE_RE = re.compile(
    r"(?:"
    # Arabic style: سورة البقرة 275 / البقرة:275 / 2:275
    r"(?:سورة\s+)?[\u0600-\u06FF\w]+[\s:،,]*(\d{1,3})[\s:،,]*(\d{1,3})"
    r"|"
    # Numeric only: 2:275 or (2:275)
    r"[(]?(\d{1,3})[:\-](\d{1,3})[)]?"
    r")",
    re.UNICODE,
)

# Captures a quoted Arabic block of at least 20 characters (lazy match)
# enclosed by quotation marks; group 1 is the candidate citation text.
_QUOTED_AYAH_RE = re.compile(
    r"(?:[\"«»'])\s*([\u0600-\u06FF\s،؛.!?]{20,}?)\s*(?:[\"»'])",
    re.UNICODE,
)

# Citation-intent detector applied to the query. Answers frequently recite an
# ayah or surah in plain text without quotation marks (the evaluation corpus
# does not use quotation marks). The intent to cite scripture is therefore
# detected from the query itself, so that the whole answer can be treated as
# a citation candidate when no quoted block is found.
_AYAH_CITATION_INTENT_RE = re.compile(
    r"(اذكر|اقرأ|اكتب|قل)\s*.{0,10}(آية|الآية|سورة|السورة)"
    r"|(ما\s+هي|ماهي)\s*.{0,10}(سورة|السورة|آية|الآية)",
    re.UNICODE,
)



# Result dataclass


@dataclass
class IslamicContentResult:
    """
    Outcome of ``check_islamic_content()``, consumed by Layer 3.

    Fields
    ------
    passed
        False when at least one hard-fail check code is present.
    skipped
        True when the domain pre-check fast-pathed the query (no religious
        signal); in that case ``passed`` is True.
    failed_checks
        Hard-fail check codes of the form ``"<code>:<count>"``, e.g.
        ``islamic_hajj_error:1``.
    warnings
        Non-blocking findings (unverifiable ayat, fiqh query without
        evidence).
    islamic_no_evidence
        True when the query is classified as fiqh and no evidence passages
        were supplied. Advisory signal for the caller.
    fabricated_citations, hajj_errors
        Itemised findings recorded for the Audit Logger. ``hajj_errors`` also
        contains known-count errors (see ``_validate_islamic_known_counts``).
    """

    passed: bool
    skipped: bool = False          # True when the domain pre-check fast-pathed
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # True when the query is fiqh-related and no evidence passages were provided
    islamic_no_evidence: bool = False
    # Itemised findings for the audit log
    fabricated_citations: List[str] = field(default_factory=list)
    hajj_errors: List[str] = field(default_factory=list)



# Arabic normalization (mirrors claim_verifier to avoid import cycle)


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")


def _norm(text: str) -> str:
    """
    Normalise Arabic text for comparison.

    Applies NFC normalisation, removes diacritics, and maps variant letters
    to a canonical form (أ/إ/آ/ٱ -> ا, ة -> ه, ى -> ي, ؤ -> و, ئ -> ي).
    The mapping mirrors the one in ``claim_verifier``; it is duplicated here
    to avoid a circular import.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    return text.strip()



# Similarity helpers


def _token_overlap(a: str, b: str) -> float:
    """
    Jaccard overlap on whitespace tokens (symmetric).

    Not referenced by the current implementation; it has been superseded by
    ``_token_containment`` for the reasons given there.
    """
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _seq_similarity(a: str, b: str) -> float:
    """Character-level ``SequenceMatcher`` ratio in the range [0, 1]."""
    return SequenceMatcher(None, a, b).ratio()


def _token_containment(small_tokens: List[str], big_tokens: List[str]) -> float:
    """
    Fraction of the vocabulary of *small_tokens* that is present in
    *big_tokens*.

    This is a recall-style measure. Unlike symmetric Jaccard overlap, it
    remains high when *big_tokens* (the candidate) is much longer than
    *small_tokens* (e.g. a single short canonical ayah), as happens when
    several ayat are concatenated without quotation marks or punctuation,
    which is common in LLM answers.
    """
    small = set(small_tokens)
    big = set(big_tokens)
    if not small:
        return 0.0
    return len(small & big) / len(small)


def _best_canonical_match(quoted: str) -> Tuple[float, str]:
    """
    Return ``(similarity, matched_key)`` for the WORST-matching canonical
    ayah that meaningfully overlaps with *quoted*.

    Procedure
   
    1. Normalise *quoted* and each canonical ayah (``_norm``).
    2. Pre-filter: a canonical ayah is a candidate only if the token
       containment between it and *quoted*, evaluated in both directions
       (maximum of the two), is at least ``AYAH_TOKEN_OVERLAP_MIN``.
    3. For each candidate, compute the best character-level similarity:
       the whole-string ``SequenceMatcher`` ratio when *quoted* has no more
       tokens than the ayah, otherwise the maximum ratio over all
       token windows of the ayah's length slid across *quoted*.
    4. Return the candidate with the LOWEST such similarity, i.e. the ayah
       that is present but most distorted. If there is no candidate,
       return ``(0.0, "")``.

    Rationale for the design
   
    * Containment instead of symmetric Jaccard: when a candidate
      concatenates several ayat (e.g. all of Al-Ikhlas as one unpunctuated
      block), the union vocabulary grows beyond any single ayah and dilutes
      both the Jaccard pre-filter and the whole-string ratio, so that an
      altered ayah could fall below the filters.
    * Worst match instead of best match: returning only the highest score
      would allow a faithfully reproduced fragment (e.g. 112:1) to mask a
      corrupted one elsewhere in the same answer (e.g. an altered 112:3).
    * Bidirectional containment: a genuine partial quotation of a long
      ayah (e.g. the first part of Ayat al-Kursi) has low containment of
      the canonical text in the candidate merely because the candidate is
      shorter, so both directions are checked.
    """
    norm_quoted   = _norm(quoted)
    quoted_tokens = norm_quoted.split()
    worst_sim = None
    worst_key = ""

    for key, canonical in _CANONICAL_AYAT.items():
        norm_can  = _norm(canonical)
        can_tokens = norm_can.split()
        n = len(can_tokens)
        if n == 0:
            continue

        # Token-level pre-filter (maximum of both containment directions);
        # see the bidirectional-containment rationale in the docstring.
        overlap = max(
            _token_containment(can_tokens, quoted_tokens),
            _token_containment(quoted_tokens, can_tokens),
        )
        if overlap < AYAH_TOKEN_OVERLAP_MIN:
            continue

        # Character-level similarity: whole string for short candidates,
        # best sliding token window for longer ones.
        if len(quoted_tokens) <= n:
            window_sim = _seq_similarity(norm_quoted, norm_can)
        else:
            window_sim = 0.0
            for start in range(0, len(quoted_tokens) - n + 1):
                window = " ".join(quoted_tokens[start:start + n])
                s = _seq_similarity(window, norm_can)
                if s > window_sim:
                    window_sim = s

        # Keep the lowest similarity across candidates (worst match).
        if worst_sim is None or window_sim < worst_sim:
            worst_sim = window_sim
            worst_key = key

    if worst_sim is None:
        return 0.0, ""
    return worst_sim, worst_key



# Sub-checks


def _validate_quran_citations(text: str, query: str = "") -> Tuple[List[str], List[str]]:
    """
    Locate quoted Quranic text in *text* and validate it against the
    canonical corpus.

    Candidate extraction
   
    Quoted Arabic blocks are extracted with ``_QUOTED_AYAH_RE``. If none is
    found and *query* signals explicit citation intent (e.g. "اذكر آية
    الكرسي", "ما هي سورة الفاتحة؟"), the whole stripped answer (at least
    15 characters) is treated as one implicit citation candidate, because
    recitations of scripture are not always enclosed in quotation marks.
    Candidates shorter than 20 characters are ignored.

    Decision per candidate (see ``_best_canonical_match``)
    
    * similarity == 0.0 (no canonical overlap): the citation is unverifiable;
      a warning is recorded and processing continues.
    * 0.0 < similarity < ``AYAH_SIMILARITY_THRESHOLD``: the text overlaps a
      known ayah but deviates from it; ``IslamicFabricationError`` is
      raised. The exception propagates through ``check_islamic_content`` to
      ``output_validator.run_layer3``, which must map it to l3_passed=False,
      block=True.
    * similarity >= threshold: accepted.

    Returns
    
    (hard_fails, warnings)
        hard_fails — always empty in the current implementation, because the
                     fabricated-citation path raises instead of returning.
                     The list is retained for interface compatibility.
        warnings   — unverifiable ayat (no overlap with the corpus).

    Raises
    
    IslamicFabricationError
        See the decision rule above.
    """
    hard_fails: List[str] = []
    warnings:   List[str] = []

    quoted_blocks = _QUOTED_AYAH_RE.findall(text)
    if not quoted_blocks:
        if _AYAH_CITATION_INTENT_RE.search(query):
            candidate = text.strip()
            if len(candidate) >= 15:
                quoted_blocks = [candidate]
        if not quoted_blocks:
            return hard_fails, warnings

    for quoted in quoted_blocks:
        quoted_stripped = quoted.strip()
        if len(quoted_stripped) < 20:
            continue

        sim, best_key = _best_canonical_match(quoted_stripped)

        if sim == 0.0:
            # No canonical ayah overlaps the candidate: it cannot be verified
            # against the corpus (warning only).
            warnings.append(f"unverifiable_ayah: '{quoted_stripped[:60]}…'")
            logger.warning(
                "islamic_guard: quoted ayah has no canonical overlap — "
                "possible fabrication: '%s'", quoted_stripped[:80]
            )
            continue

        if sim < AYAH_SIMILARITY_THRESHOLD:
            # The candidate overlaps a known ayah but deviates from its
            # canonical wording: treated as fabricated and escalated by
            # exception (hard block) rather than by a soft warning.
            msg = (
                f"Fabricated Quranic citation detected "
                f"(best_match={best_key}, sim={sim:.2f}). "
                f"Cannot serve answer without verified source. "
                f"Quoted: '{quoted_stripped[:80]}…'"
            )
            logger.error(
                "islamic_guard: FABRICATED AYAH — hard block. "
                "best_canonical_match=%s sim=%.2f quoted='%s'",
                best_key, sim, quoted_stripped[:80],
            )
            raise IslamicFabricationError(msg)

    return hard_fails, warnings



# Well-known static Islamic counts
#
# Statements such as "أركان الإسلام أربعة فقط" (only four pillars of Islam)
# or "الصلوات المفروضة أربع صلوات" (four obligatory prayers) are factual
# errors: the correct counts are five in both cases. They are neither Quran
# citations nor Hajj structural errors, so neither of the two checks above
# covers them. Because these facts are unambiguous and do not vary over
# time, a small explicit lookup table is used.


# Arabic number words (1–10) mapped to integers. Keys are written in the
# normalised form produced by _norm() (ة -> ه), since they are matched
# against normalised text.
_AR_NUM_WORD_MAP: Dict[str, int] = {
    "واحد": 1, "واحده": 1, "اثنان": 2, "اثنين": 2, "ثلاثه": 3, "ثلاث": 3,
    "اربعه": 4, "اربع": 4, "خمسه": 5, "خمس": 5, "سته": 6, "ست": 6,
    "سبعه": 7, "سبع": 7, "ثمانيه": 8, "ثمان": 8, "تسعه": 9, "تسع": 9,
    "عشره": 10, "عشر": 10,
}
# Alternation of the keys above, ordered by decreasing length so that longer
# forms take precedence over their prefixes.
_AR_NUM_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted((re.escape(k) for k in _AR_NUM_WORD_MAP), key=len, reverse=True)) + r")\b",
    re.UNICODE,
)


def _stated_count_near(text: str, anchor_end: int, window: int = 20) -> Optional[int]:
    """
    Return the count stated near an anchor phrase, or None if none is found.

    The inspected span covers *window* characters on each side of
    *anchor_end*. A decimal number in the span takes precedence over an
    Arabic number word; if neither is present, None is returned. The input
    is expected to be normalised with ``_norm``.
    """
    span = text[max(0, anchor_end - window): anchor_end + window]
    digit_m = re.search(r"\d+", span)
    if digit_m:
        return int(digit_m.group(0))
    word_m = _AR_NUM_WORD_RE.search(span)
    if word_m:
        return _AR_NUM_WORD_MAP.get(word_m.group(0))
    return None


# Each entry: (anchor pattern, correct count, display label). The patterns
# are written for the normalised form of the text (_norm maps أ/إ/آ/ٱ -> ا
# and ة -> ه) because _validate_islamic_known_counts() applies them to
# normalised text; they are not written in standard orthography.
_KNOWN_ISLAMIC_COUNTS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"اركان\s*الاسلام", re.UNICODE), 5, "أركان الإسلام"),
    (re.compile(r"الصلوات\s*المفروضه|الصلاه\s*المفروضه", re.UNICODE), 5, "الصلوات المفروضة"),
]


def _validate_islamic_known_counts(text: str) -> List[str]:
    """
    Return error strings for well-known Islamic counts (pillars of Islam,
    daily obligatory prayers) that are stated incorrectly.

    For each entry of ``_KNOWN_ISLAMIC_COUNTS`` the first occurrence of the
    anchor phrase in the normalised text is located, the count stated near
    it is extracted, and an ``islamic_fact_error`` is reported if it differs
    from the expected value.
    """
    errors: List[str] = []
    norm = _norm(text)
    for anchor_re, correct, label in _KNOWN_ISLAMIC_COUNTS:
        m = anchor_re.search(norm)
        if not m:
            continue
        stated = _stated_count_near(norm, m.end())
        if stated is not None and stated != correct:
            errors.append(f"islamic_fact_error: {label}={stated} (expected {correct})")
            logger.error(
                "islamic_guard: known-count error — %s stated=%s expected=%s",
                label, stated, correct,
            )
    return errors


def _validate_hajj_content(text: str) -> List[str]:
    """
    Return the Hajj structural errors detected in *text* (hard-fail items).

    Each pattern of ``_HAJJ_FALSE_ARKAAN`` is searched in the raw text, and
    its first match is reported as ``hajj_structural_error``.
    """
    errors: List[str] = []
    for pat in _HAJJ_FALSE_ARKAAN:
        m = pat.search(text)
        if m:
            errors.append(f"hajj_structural_error: '{m.group(0)}'")
            logger.error(
                "islamic_guard: Hajj structural error detected: '%s'", m.group(0)
            )
    return errors


def _is_islamic_fiqh_query(query: str, answer: str) -> bool:
    """Return True when the query or the answer contains fiqh-related lexical signals."""
    combined = query + " " + answer
    return bool(_ISLAMIC_FIQH_PATTERNS.search(combined))



# Public entry point


def check_islamic_content(
    raw_output: str,
    original_query: str,
    evidence_passages: Optional[List[str]] = None,
    *,
    content_type: Optional[str] = None,
) -> IslamicContentResult:
    """
    Run the Islamic content checks on *raw_output*.

    Invoked by ``run_layer3()`` (Output Validator Orchestrator, Layer 3)
    after the standard checks. A domain pre-check restricts execution to
    queries classified as religious; other queries are fast-pathed and
    returned as ``IslamicContentResult(skipped=True)``.

    Processing order
    
    1. Empty output -> ``passed=True`` (nothing to validate).
    2. Domain pre-check (``_is_religious_domain``) -> skipped result.
    3. Quran citation integrity (may raise ``IslamicFabricationError``).
    4. Hajj structural errors, followed by the known-count errors, which are
       reported under the same ``islamic_hajj_error`` check code so that
       they surface under an already-recognised code.
    5. Advisory warning when the query is fiqh-related and no evidence
       passages were provided.

    Parameters
    
    raw_output:
        The LLM answer to validate.
    original_query:
        The user's original query.
    evidence_passages:
        Grounded passages from tool results. When None or empty and the
        query is classified as fiqh-related, ``islamic_no_evidence=True`` is
        set on the result.
    content_type:
        Query domain label from the pipeline classifier (e.g. 'fiqh',
        'technical', 'recipe'). When it belongs to ``RELIGIOUS_DOMAINS`` the
        guard is activated; otherwise the keyword fallback is used.

    Returns
 
    IslamicContentResult
        ``skipped=True`` when the domain pre-check excluded the query.
        ``passed=False`` on hard fails (Hajj structural errors, known-count
        errors). Fabricated Quran citations do not produce
        ``passed=False``: they raise ``IslamicFabricationError`` instead, so
        callers must catch it. ``islamic_no_evidence=True`` is a warning and
        not a hard fail; the caller may use it to set
        ``require_tool_evidence=True`` for the next invocation.

    Raises
    
    IslamicFabricationError
        When a quoted Quran text overlaps a canonical ayah (passing the
        token-overlap pre-filter) but its similarity is below
        ``AYAH_SIMILARITY_THRESHOLD``. ``output_validator.run_layer3`` must
        catch it.
    """
    if not raw_output:
        return IslamicContentResult(passed=True)

    # ── Domain pre-check ─────────────────────────────────────────────────
    # The guard is activated only for religious-domain queries. This avoids
    # false activation on non-religious queries, e.g. the Arabic request
    # "شرح لي كيف يشتغل transformer architecture".
    if not _is_religious_domain(content_type, original_query):
        logger.debug(
            "islamic_guard: content_type=%r query has no religious-domain signal "
            "— skipping guard entirely.",
            content_type,
        )
        return IslamicContentResult(passed=True, skipped=True)

    hard_fails:  List[str] = []
    warnings:    List[str] = []
    is_fiqh      = _is_islamic_fiqh_query(original_query, raw_output)

    # ── Check 1: Quran citation integrity ─────────────────────────────────
    # _validate_quran_citations raises IslamicFabricationError for fabricated
    # ayat; the exception propagates out of this function.
    cite_fails, cite_warns = _validate_quran_citations(raw_output, original_query)
    hard_fails.extend(cite_fails)
    warnings.extend(cite_warns)

    # ── Check 2: Hajj structural errors ───────────────────────────────────
    hajj_errors = _validate_hajj_content(raw_output)

    # ── Check 2b: other well-known static Islamic counts ──────────────────
    # Folded into the same hajj_errors bucket (instead of introducing a new
    # failed_checks prefix) so that these errors surface under the existing,
    # already-recognised "islamic_hajj_error" check code.
    hajj_errors.extend(_validate_islamic_known_counts(raw_output))
    hard_fails.extend(hajj_errors)

    # ── Warning: fiqh query without grounded evidence ─────────────────────
    islamic_no_evidence = False
    if is_fiqh and not evidence_passages:
        islamic_no_evidence = True
        warnings.append(
            "islamic_fiqh_query_without_tool_evidence — "
            "require_tool_evidence should be True for this query type"
        )
        logger.warning(
            "islamic_guard: Islamic fiqh query answered without tool evidence. "
            "Set require_tool_evidence=True and provide a fiqh RAG corpus."
        )

    # Build the check codes reported to Layer 3 ("<code>:<count>").
    failed_check_codes = []
    if cite_fails:
        failed_check_codes.append(f"islamic_fabricated_citation:{len(cite_fails)}")
    if hajj_errors:
        failed_check_codes.append(f"islamic_hajj_error:{len(hajj_errors)}")

    passed = len(failed_check_codes) == 0

    return IslamicContentResult(
        passed=passed,
        skipped=False,
        failed_checks=failed_check_codes,
        warnings=warnings,
        islamic_no_evidence=islamic_no_evidence,
        fabricated_citations=cite_fails,
        hajj_errors=hajj_errors,
    )
