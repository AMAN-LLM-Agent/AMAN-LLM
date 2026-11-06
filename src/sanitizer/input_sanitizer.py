# src/sanitizer/input_sanitizer.py
"""
Input sanitizer for a multilingual AI safety pipeline.

Design contract (NON-NEGOTIABLE):
  - NEVER blocks, raises, or rejects any input.
  - NEVER silently discards content without logging.
  - All transformations preserve semantic intent and are traceable.
  - Truncation is smart (head + tail), not a hard cut.
  - Every output carries a structured risk assessment.
  - Forensic evidence (original text, all signals) is always preserved.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

# CONSTANT

_MAX_LEN = 4096

_SMART_TRUNC_HEAD = 2048
_SMART_TRUNC_TAIL = 512
_SMART_TRUNC_SEP  = "\n… [truncated] …\n"

# SCRIPT DETECTIO

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_SCRIPT_RE  = re.compile(r"[a-zA-Z]")
_CJK_SCRIPT_RE    = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF]")

# INVISIBLE / DANGEROUS CHARACTER STRIPPIN
#
# RTL override detection always runs on `original` BEFORE stripping,
# so evidence is preserved in the risk report.
#
# Ranges covered:
#   U+00AD          Soft hyphen
#   U+034F          Combining grapheme joiner
#   U+200B–U+200F   Zero-width space / ZWNJ / ZWJ / LRM / RLM
#   U+2028–U+2029   Line/paragraph separator
#   U+202A–U+202E   Directional formatting (LRE, RLE, PDF, LRO, RLO)
#   U+2060–U+2064   Word joiner and related
#   U+206A–U+206F   Deprecated formatting characters
#   U+FE00–U+FE0F   Variation selectors 1–16
#   U+FEFF          BOM / zero-width no-break space
#   U+E0000–U+E007F Unicode tag characters
#   U+E0100–U+E01EF Variation selectors supplement
#
_INVISIBLE_RE = re.compile(
    r"[\u00AD\u034F"
    r"\u200B-\u200F"
    r"\u2028\u2029"
    r"\u202A-\u202E"
    r"\u2060-\u2064"
    r"\u206A-\u206F"
    r"\uFE00-\uFE0F"
    r"\uFEFF"
    r"\U000E0000-\U000E007F"
    r"\U000E0100-\U000E01EF"
    r"]"
)

_AR_DIACRITICS_RE = re.compile(
    r"[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]"
)

# ── Copy-paste Unicode whitespace variants (P3-4) ──────────────
# These survive NFKC normalization and are common copy-paste artifacts
# (e.g. U+00A0 from web/PDF, U+202F narrow no-break space, U+3000 ideographic
# space). They are NOT in _INVISIBLE_RE (which targets zero-width and bidi
# control chars) — they are visible-width spaces that displace character
# positions and cause false obfuscation_detected signals when the deobfuscator
# produces a different string solely because of these boundary characters.
#
# Ranges / codepoints covered:
#   U+00A0   NO-BREAK SPACE          (most common copy-paste artifact)
#   U+1680   OGHAM SPACE MARK
#   U+2000–U+200A  General punctuation spaces (en quad … hair space)
#   U+202F   NARROW NO-BREAK SPACE
#   U+205F   MEDIUM MATHEMATICAL SPACE
#   U+3000   IDEOGRAPHIC SPACE
#
# Note: U+200B–U+200F (zero-width spaces / directional marks) are already
# handled by _INVISIBLE_RE above; they are excluded here to avoid double-
# stripping and conflating the two distinct normalization steps.
_UNICODE_SPACE_VARIANTS_RE = re.compile(
    r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]"
)

_RTL_OVERRIDE_RE = re.compile(r"[\u202E\u200F\u202B]")

_ENCLOSED_LETTER_RE = re.compile(
    r"[\U0001F1E0-\U0001F1FF"
    r"\U0001F170-\U0001F171"
    r"\U0001F17E-\U0001F17F"
    r"\U0001F191-\U0001F19A"
    r"\U0001F1E6-\U0001F1FF"
    r"\U0001F150-\U0001F169"
    r"]"
)


# WHITESPACE ARTIFACT STRIPPING  (P3-4

def strip_whitespace_artifacts(text: str) -> Tuple[str, bool]:
    """
    Replace Unicode space variants (U+00A0, en/em-space, narrow NBSP, etc.)
    with regular ASCII spaces, then strip leading/trailing ASCII whitespace.

    Returns (cleaned_text, artifact_was_present).

    This step runs AFTER NFKC normalization (step 4) and BEFORE the invisible-
    char strip (step 5).  It targets visible-width Unicode spaces that survive
    NFKC and are overwhelmingly copy-paste artifacts (e.g. U+00A0 from web/PDF
    copy, U+202F from mobile keyboards).  They are distinct from the zero-width
    bidi/control chars handled by _INVISIBLE_RE:

      • _INVISIBLE_RE  — zero-width / directional control chars (security risk)
      • strip_whitespace_artifacts — visible-width non-ASCII spaces (UX noise)

    Separating the two allows the logger and metrics to distinguish genuine
    obfuscation attempts from copy-paste normalization (P3-4 recommendation).
    """
    replaced = _UNICODE_SPACE_VARIANTS_RE.sub(" ", text)
    cleaned  = replaced.strip()
    return cleaned, cleaned != text

# NORMALIZATION PATTERN

_REPEATED_LETTERS_RE = re.compile(r"([A-Za-z\u0600-\u06FF])\1{2,}")
_REPEATED_PUNCT_RE   = re.compile(r"([^\w\s])\1{3,}")

# ── Spaced / punctuation-attack deobfuscation ─────────────────
# Covers ALL common single-char punctuation used to evade detection.
_SPACED_ATTACK_SEP = r"""[ \t\-_./:;,~|@#%^&*!?<>=+`'"()\[\]{}\\]"""
_SPACED_ATTACK_RE = re.compile(
    r"(?:(?:^|(?<=\s)|(?<![A-Za-z\u0600-\u06FF\d])))"
    r"[A-Za-z\u0600-\u06FF]"
    r"(?:" + _SPACED_ATTACK_SEP + r"+[A-Za-z\u0600-\u06FF]){3,}",
    re.MULTILINE,
)

_ARABIZI_DIGIT_MARKERS_RE = re.compile(
    r"\b[a-zA-Z]+[2375698][a-zA-Z]+\b",
    re.IGNORECASE,
)

# FIX v3: match digit-at-start tokens (e.g. "3tini", "7abit", "9al")
# as well as mid-word and end-of-word Arabizi digits.
_ARABIZI_DIGIT_ANY_RE = re.compile(
    r"\b(?:[2375698][a-zA-Z]+|[a-zA-Z]+[2375698][a-zA-Z]*)\b",
    re.IGNORECASE,
)


# LEET-SPEAK DEOBFUSCATIO

_LEET_MAP: dict[int, str] = {
    ord("0"): "o",
    ord("1"): "i",
    ord("3"): "e",
    ord("4"): "a",
    ord("5"): "s",
    ord("6"): "g",
    ord("@"): "a",
    ord("¥"): "y",
    ord("€"): "e",
    ord("$"): "s",
    ord("!"): "i",
    ord("|"): "l",
}

_LEET_TARGET_WORDS = frozenset({
    "1gn0re", "1gnore", "ign0re", "ignor3",
    "byp4ss", "byp@ss", "b¥pass", "bypa55", "bypas5",
    "syst3m", "sy5tem", "syst€m",
    "pr0mpt", "prom9t",
    "h4ck", "hack1ng", "hack3r",
    "m4lware", "malw4re",
    "sh0w", "5how",
    "unf1ltered", "unfilt3red",
    "prомpt",
    "r3v3al", "r3veal", "reve4l",
    "3xploit", "expl0it",
    "0verride", "overr1de",
})

LEET_SPEAK_ALLOWLIST: frozenset[str] = frozenset({
    "api", "url", "http", "https", "sql", "css", "xml", "json",
    "pdf", "cpu", "gpu", "ram", "sdk", "ide", "cli", "gui",
    "ftp", "ssh", "tcp", "udp", "ip", "dns", "ssl", "tls",
    "jwt", "oauth", "html", "csv", "yaml", "toml", "uuid",
    "crud", "rest", "rpc", "orm", "npm", "pip", "apt", "yum",
    "gcc", "llvm", "jvm", "jdk", "jre", "php", "asp", "jsp",
})

_LEET_CANDIDATE_RE = re.compile(
    r"[a-zA-Z0-9¥€$@!|]{3,}",
    re.IGNORECASE,
)

# Module-level regex for leet-speak attack keyword detection.
# Defined here (not inline in _assess_risk) so _detect_leet_speak()
# can reference it without recompiling on every call.
# Words that match _leet_attack_re purely because of their normal spelling
# (no digit/symbol substitution).  These should never fire leet_speak_attack_keyword
# — they are handled by _INJECTION_RE / code_injection signals instead.
_LEET_PLAIN_ENGLISH_SKIP: frozenset[str] = frozenset({
    # [s5]y[s5]t[e3]m  matches "system" — real word, not obfuscated
    "system",
    # [s5]h[0o]w       matches "show"   — real word, not obfuscated
    "show",
    # pr[0o]mpt        matches "prompt" — real word, not obfuscated
    "prompt",
    # ign[0o]r[e3]     matches "ignore" — real word, not obfuscated
    "ignore",
})

_leet_attack_re = re.compile(
    r"\b(?:"
    r"1gn[0o]r[e3]|ign[0o]r[e3]|"
    r"byp[4a][s5][s5]|b[¥y]p[4a][s5][s5]|"
    r"[s5]y[s5]t[e3]m|"
    r"pr[0o]mpt|"
    r"[s5]h[0o]w|"
    r"unf[i1]lt[e3]r[e3]d"
    r")\b",
    re.IGNORECASE,
)

# ── Code / shell injection pattern (P2-6 fix) ─────────────────────────────
# Matches Python / shell injection constructs that should be classified as
# code_injection rather than leet_speak_attack_keyword.
_CODE_INJECTION_RE = re.compile(
    r"(?:"
    # Python builtins used for code execution
    r"\beval\s*\(|"
    r"\bexec\s*\(|"
    # Import-based execution: import("os"), __import__("os")
    r"\b(?:__import__|import)\s*\(['\"]|"
    # os / subprocess execution calls
    r"\bos\s*\.\s*system\s*\(|"
    r"\bos\s*\.\s*popen\s*\(|"
    r"\bsubprocess\s*\.\s*(?:call|run|Popen|check_output)\s*\(|"
    # base64 decode used to smuggle payloads
    r"\bbase64\s*\.\s*b64decode\s*\(|"
    # Shell command patterns
    r"\bsh\s+-c\s+['\"]|"
    r"\bbash\s+-c\s+['\"]|"
    # Chained shell execution via ; or &&
    r"(?:ls|cat|rm|wget|curl|chmod|chown|id|whoami)\s*(?:;|&&|\|\|)"
    r")",
    re.IGNORECASE,
)


_LEET_CHAR_RE = re.compile(r"[0-9¥€$@!|]")

def _detect_leet_speak(text: str) -> list[str]:
    signals = []
    tokens = text.split()
    for token in tokens:
        cleaned = re.sub(r'[^a-z0-9]', '', token.lower())
        # Skip known-safe technical acronyms (API, URL, SQL, …)
        if cleaned in LEET_SPEAK_ALLOWLIST:
            continue
        # A leet-speak substitution REQUIRES at least one digit or special
        # character replacing a letter.  Tokens with no such characters are
        # plain English words and must not fire this signal — they are either
        # benign (show, system, prompt) or will be caught by _CODE_INJECTION_RE.
        if not _LEET_CHAR_RE.search(token):
            continue
        # Belt-and-suspenders: skip any token whose stripped form is a known
        # plain English keyword that has no leet variant in _LEET_TARGET_WORDS.
        if token.lower() in _LEET_PLAIN_ENGLISH_SKIP:
            continue
        if _leet_attack_re.search(token):
            signals.append('leet_speak_attack_keyword')
    return signals

def _apply_leet_deobfuscation(text: str) -> str:
    """
    Replace leet-speak tokens with their plaintext equivalents.
    Only tokens that are known attack-keyword candidates are transformed,
    preventing false positives on ordinary technical text like "3gp", "mp3".
    """
    def _maybe_deleet(m: re.Match) -> str:
        token = m.group()
        if token.lower() in _LEET_TARGET_WORDS:
            return token.translate(_LEET_MAP)
        return token

    return _LEET_CANDIDATE_RE.sub(_maybe_deleet, text)

# ALGERIAN DIALECT TOKEN

_ALGERIAN_WORDS_RE = re.compile(
    r"\b(?:wach|wesh|wa9ila|rani|rak|raki|khoya|kho|sahbi|sa7bi|"
    r"bezzaf|bzf|3lah|3lech|ndir|nheb|n7eb|ta3|dork|drouk|"
    r"labas|mlih|mliha|kifah|kifeh|chouf|chof|"
    # FIX v3: "3tini" must appear explicitly because digit-prefixed
    # tokens at word-start can trip up \b in some re engines.
    r"3tini|"
    r"rana|nta|nti|hna|mazal|walou|inchallah|brk|"
    r"cv|svp|saha|ya3tik|"
    r"n9ra|t9der|ta3tini|7abit|nrou7|ya3ni|w9t|f9t|"
    r"ma3ndich|ma3ndi|baraka|winou|win|mta3|"
    r"nchallah|yallah|yala|hadra|kima|"
    r"3andi|3andek|3andha|3andna|3andkom|"
    r"n7eb|n7ib|t7eb|t7ib|"
    r"roh|rouh|nroh|nrouh)\b",
    re.IGNORECASE,
)

_ALGERIAN_AR_RE = re.compile(
    r"(?:واش|راك|راكي|بزاف|خويا|صحبي|درك|برك|"
    r"واشراك|نتا|نتي|حنا|علاه|كيفاش|مليح|مليحة|"
    r"شوف|عطيني|والو|مازال|"
    # Additional unambiguous Darja markers (P3-3 fix):
    #   ضكا = "now / at this moment" (Algerian Darja, not MSA)
    #   شكون = "who?" (Algerian Darja interrogative, not MSA مَن)
    r"ضكا|شكون)",
)

# FRENCH / ENGLISH DETECTIO

_FRENCH_RE = re.compile(
    r"\b(?:bonjour|merci|pourquoi|comment|parce|"
    r"je|tu|vous|nous|avec|sans|peux|veux|"
    r"montre|donne|explique|aide|syst[eè]me|"
    r"contourner|d[eé]sactiver|"
    r"les|des|une|est|que|qui|pas|"
    r"tout|toute|toutes|tous|depuis|aussi|mais|donc|"
    r"sur|pour|dans|cette|cet|ces|"
    r"oubliez?|oublier|ignorer|ignorez?|"
    r"nouvelles?|depuis|toujours|jamais|"
    r"tr[eè]s|bien|ici|eux|elle|elles|leur|leurs|"
    r"alors|encore|avant|apr[eè]s|quand|comme|"
    r"non|oui|voil[aà]|voici|peut|être|avoir)\b",
    re.IGNORECASE,
)

_ENGLISH_RE = re.compile(
    r"\b(?:the|and|with|from|hello|please|"
    r"help|show|tell|make|give|system|"
    r"instructions|ignore|bypass|developer|"
    r"how|what|why|can|could|would|"
    r"i|my|your|his|her|its|we|they|you|it|he|she|"
    r"is|are|was|were|be|been|being|"
    r"have|has|had|do|does|did|"
    r"get|got|let|put|use|used|using|"
    r"want|need|know|think|see|look|say|said|"
    r"this|that|these|those|not|no|yes|okay|ok|"
    r"a|an|in|on|at|to|of|for|by|or|but|if|so)\b",
    re.IGNORECASE,
)

# PROMPT INJECTION / ATTACK SIGNAL PATTERN

_INJECTION_RE = re.compile(
    r"\b(?:"
    # ── English injection phrases ──────────────────────────────
    r"ignore\s+(?:all\s+)?(?:previous|above|prior|your)\s+instructions?|"
    r"disregard\s+(?:all\s+)?(?:previous|above|prior|your)\s+instructions?|"
    r"forget\s+(?:all\s+)?(?:previous|above|prior|your)\s+instructions?|"
    r"you\s+are\s+now|act\s+as|pretend\s+(?:you\s+are|to\s+be)|"
    r"new\s+instructions?|override\s+(?:your\s+)?instructions?|"
    r"system\s*prompt|developer\s*mode|jailbreak|"
    r"bypass\s+(?:your\s+)?(?:filters?|restrictions?|rules?|guidelines?|safety)|"
    r"ignore\s+(?:your\s+)?(?:filters?|restrictions?|rules?|guidelines?|safety|rules?)|"
    # Prompt-extraction phrases
    r"reveal\s+(?:hidden\s+)?(?:prompt|instructions?|secrets?|stored)|"
    r"show\s+(?:me\s+)?(?:the\s+)?(?:hidden|system|developer|internal)\s+(?:prompt|instructions?|rules?)|"
    r"print\s+(?:the\s+)?(?:system\s+prompt|internal|api\s+keys?)|"
    r"dump\s+(?:your\s+)?(?:initialization|database)|"
    r"output\s+confidential\s+instructions?|"
    r"display\s+(?:internal|moderation|user)\s+(?:config(?:uration)?|rules?|sessions?)|"
    r"(?:what\s+are\s+your|tell\s+me\s+your)\s+(?:hidden\s+)?rules?|"
    r"show\s+(?:memory|environment)\s+(?:contents?|variables?)|"
    r"export\s+hidden\s+data|"
    r"retrieve\s+confidential\s+info|"
    r"leak\s+private\s+information|"
    r"show\s+authentication\s+tokens?|"
    # System-access / execution phrases
    r"execute\s+(?:shell|remote)\s+(?:command|script)|"
    r"run\s+terminal\s+access|"
    r"open\s+filesystem|"
    r"delete\s+logs?|"
    r"modify\s+permissions?|"
    r"install\s+malware|"
    r"download\s+payload|"
    # Modern jailbreak phrases
    r"from\s+now\s+on\s+you\s+(?:will|must|should)|"
    r"your\s+new\s+persona\s+is|"
    r"enter\s+(?:dan|jailbreak|developer|god|maintenance|debug)\s*mode|"
    r"pretend\s+(?:this\s+is\s+)?(?:a\s+)?(?:game|story|simulation|test)|"
    r"hypothetically\s+speaking|"
    r"for\s+(?:educational|research|fictional)\s+purposes|"
    r"in\s+this\s+(?:story|scenario|roleplay|game)|"
    r"as\s+a\s+(?:fictional|hypothetical)|"
    r"act\s+(?:as\s+an?\s+)?(?:uncensored|unrestricted|administrator|evil)|"
    r"simulate\s+unrestricted|"
    r"disable\s+(?:safety|ethical|content)\s+(?:filters?|constraints?|limits?)|"
    r"respond\s+without\s+(?:ethical|restrictions?)|"
    r"answer\s+without\s+restrictions?|"
    r"pretend\s+(?:safety|the\s+policy)\s+(?:does\s+not\s+exist|changed)|"
    # ── French injection phrases ───────────────────────────────
    r"oubliez?\s+(?:toutes?\s+)?les\s+instructions?|"
    r"ignorez?\s+(?:toutes?\s+)?les\s+instructions?|"
    r"nouvelles?\s+instructions?|"
    r"contourne(?:r|z)?\s+(?:les\s+)?(?:filtres?|restrictions?)|"
    r"donne\s+moi\s+les\s+instructions\s+cach[eé]es?|"
    r"r[eé]v[eè]le\s+le\s+prompt|"
    r"instructions\s+cach[eé]es?|"
    # ── Arabic injection phrases ───────────────────────────────
    r"تجاهل\s+(?:جميع\s+)?التعليمات|"
    r"تجاهل\s+الأوامر|"
    r"أنت\s+الآن|"
    r"تعليمات\s+جديدة|"
    r"تجاوز\s+(?:الفلاتر|القيود)|"
    r"اعرض\s+القواعد|"
    r"اعطني\s+(?:ال)?system\s*prompt"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


_SYSPROMPT_RE = re.compile(
    r"\bsystem[\s_\-]*prompt\b",
    re.IGNORECASE,
)

_HOMOGLYPH_CHARS_RE = re.compile(r"[аеорсхіѕαορνуγгոոե]")

# HOMOGLYPH MA

def _build_homoglyph_map() -> dict[int, str]:
    """
    Curated map of confusable characters → ASCII equivalents.
    """
    m: dict[int, str] = {}

    # ── Cyrillic lookalikes ────────────────────────────────────
    cyrillic: list[tuple[str, str]] = [
        ("а", "a"), ("е", "e"), ("о", "o"), ("р", "p"),
        ("с", "c"), ("х", "x"), ("і", "i"), ("ѕ", "s"),
        ("ј", "j"), ("ԁ", "d"), ("ԛ", "q"), ("ԝ", "w"),
        ("у", "u"),
        ("\u0433", "g"),   # Cyrillic SMALL LETTER GHE (г)
        ("н", "n"), ("т", "t"), ("в", "v"), ("м", "m"),
        ("б", "b"), ("з", "z"), ("к", "k"), ("л", "l"),
        ("п", "n"), ("ф", "f"), ("ц", "c"), ("ч", "ch"),
        ("ш", "sh"), ("щ", "shch"), ("ы", "y"), ("э", "e"),
        ("ю", "yu"), ("я", "ya"),
    ]
    for src, dst in cyrillic:
        m[ord(src)] = dst

    # ── Greek lookalikes ───────────────────────────────────────
    greek: list[tuple[str, str]] = [
        ("α", "a"), ("ο", "o"), ("ρ", "p"), ("ν", "v"),
        ("ε", "e"), ("κ", "k"), ("τ", "t"), ("υ", "u"),
        ("η", "n"), ("ω", "o"), ("β", "b"), ("γ", "g"),
        ("δ", "d"), ("ζ", "z"), ("λ", "l"), ("μ", "m"),
        ("ξ", "x"), ("π", "p"), ("σ", "s"), ("φ", "f"),
        ("χ", "x"),
    ]
    for src, dst in greek:
        m[ord(src)] = dst

    # ── IPA / Phonetic small-caps ──────────────────────────────
    ipa: list[tuple[str, str]] = [
        ("ɪ", "i"), ("ɢ", "g"), ("ɴ", "n"), ("ᴏ", "o"),
        ("ʀ", "r"), ("ᴇ", "e"), ("ᴀ", "a"), ("ʙ", "b"),
        ("ᴄ", "c"), ("ᴅ", "d"), ("ꜰ", "f"), ("ʜ", "h"),
        ("ᴊ", "j"), ("ᴋ", "k"), ("ʟ", "l"), ("ᴍ", "m"),
        ("ᴘ", "p"), ("ǫ", "q"), ("ꜱ", "s"), ("ᴛ", "t"),
        ("ᴜ", "u"), ("ᴠ", "v"), ("ᴡ", "w"), ("ʏ", "y"),
        ("ᴢ", "z"),
    ]
    for src, dst in ipa:
        m[ord(src)] = dst

    # ── Armenian lookalikes ────────────────────────────────────
    # FIX v3: Use explicit codepoints to avoid collision with Cyrillic г.
    armenian: list[tuple[str, str]] = [
        ("\u0563", "r"),   # Armenian SMALL LETTER GIM (գ → looks like г)
        ("\u0578", "n"),   # Armenian SMALL LETTER VO  (ո)
        ("\u0565", "e"),   # Armenian SMALL LETTER ECH  (ե)
        ("\u0574", "m"),   # Armenian SMALL LETTER MEN  (մ)
        ("\u0561", "a"),   # Armenian SMALL LETTER AYB  (ա)
        ("\u056B", "i"),   # Armenian SMALL LETTER INI  (ի)
    ]
    for src, dst in armenian:
        m[ord(src)] = dst

    # ── Circled/enclosed digits ────────────────────────────────
    for i, ch in enumerate("①②③④⑤⑥⑦⑧⑨", start=1):
        m[ord(ch)] = str(i)
    m[ord("⓪")] = "0"

    # ── Typographic dashes → ASCII hyphen ─────────────────────
    for ch in "–—‐‑":
        m[ord(ch)] = "-"

    # ── Typographic quotes → ASCII ─────────────────────────────
    m[ord("\u2018")] = "'"
    m[ord("\u2019")] = "'"
    m[ord("\u201C")] = '"'
    m[ord("\u201D")] = '"'

    # ── Fraction / division slash ──────────────────────────────
    m[ord("\u2044")] = "/"
    m[ord("\u2215")] = "/"

    return m


_HOMOGLYPH_MAP: dict[int, str] = _build_homoglyph_map()

# ARABIZI NORMALIZATIO

_ARABIZI_CHAR_MAP: dict[str, str] = {
    "2": "ء",
    "3": "ع",
    "5": "خ",
    "6": "ط",
    "7": "ح",
    "8": "غ",
    "9": "ق",
}
_ARABIZI_DIGITS: frozenset[str] = frozenset(_ARABIZI_CHAR_MAP.keys())

_ARABIZI_WORD_MAP: dict[str, str] = {
    # Greetings / status
    "wach":      "واش",     "wesh":       "واش",
    "rak":       "راك",     "raki":       "راكي",
    "rani":      "راني",    "rana":       "رانا",
    "labas":     "لاباس",
    # People / address
    "kho":       "خو",      "khoya":      "خويا",
    "sahbi":     "صاحبي",   "sa7bi":      "صاحبي",
    # Verbs / actions
    "ndir":      "ندير",
    "nheb":      "نحب",     "n7eb":       "نحب",     "n7ib": "نحب",
    "t7eb":      "تحب",     "t7ib":       "تحب",
    "3tini":     "اعطيني",
    "chouf":     "شوف",     "chof":       "شوف",
    "roh":       "روح",     "rouh":       "روح",
    "nroh":      "نروح",    "nrouh":      "نروح",
    "n9ra":      "نقرأ",    "n9rah":      "نقرأه",
    "t9der":     "تقدر",    "n9der":      "نقدر",    "y9der": "يقدر",
    "t9dri":     "تقدري",
    "ta3tini":   "تعطيني",  "ta3tik":     "تعطيك",
    "7abit":     "حابيت",   "7abba":      "حبّ",
    "7abs":      "حبس",
    # Descriptors
    "bezzaf":    "بزاف",    "bzf":        "بزاف",
    "mlih":      "مليح",    "mliha":      "مليحة",
    "walou":     "والو",    "mazal":      "مازال",
    # Grammar / function words
    "nta":       "نتا",     "nti":        "نتي",
    "hna":       "حنا",     "ta3":        "تاع",
    "3lah":      "علاه",    "3lech":      "علاش",
    "ya3tik":    "يعطيك",   "ya3ni":      "يعني",
    "3andi":     "عندي",    "3andek":     "عندك",
    "3andha":    "عندها",   "3andna":     "عندنا",
    "3andkom":   "عندكم",
    "mta3":      "متاع",
    # Misc common tokens
    "wa9ila":    "واقيلا",  "inchallah":  "إنشالله",
    "nchallah":  "إنشالله", "yallah":     "يلاه",
    "yala":      "يلا",     "saha":       "صحة",
    "kifah":     "كيفاه",   "kifeh":      "كيفاه",
    "drouk":     "درك",     "dork":       "درك",
    "baraka":    "بركة",    "winou":      "وينو",
    "win":       "وين",     "hadra":      "هضرة",
    "kima":      "كيما",    "w9t":        "وقت",
    "f9t":       "فات",
}


def _token_has_arabizi_digits(token: str) -> bool:
    return any(ch in _ARABIZI_DIGITS for ch in token)


def _token_is_pure_latin(token: str) -> bool:
    return (
        bool(_LATIN_SCRIPT_RE.search(token))
        and not bool(_ARABIC_SCRIPT_RE.search(token))
    )


def _convert_arabizi_token(token: str) -> str:
    """
    Convert a single Arabizi token to Arabic.

    Strategy:
      1. Exact word-map lookup (highest confidence).
      2. Char-level: accept if ≥ 60% of letters were converted to Arabic.
         This handles tokens like "n9ra" where '9'→'ق' but 'n','r','a'
         remain — still a useful signal even if not perfectly converted.
    """
    lower = token.lower()

    # 1. Exact known-word lookup
    if lower in _ARABIZI_WORD_MAP:
        return _ARABIZI_WORD_MAP[lower]

    # 2. Char-level: only on pure-Latin tokens containing Arabizi digits
    if not _token_is_pure_latin(token) or not _token_has_arabizi_digits(token):
        return token

    converted = lower
    for src, dst in _ARABIZI_CHAR_MAP.items():
        converted = converted.replace(src, dst)

    arabic_chars = sum(1 for c in converted if "\u0600" <= c <= "\u06FF")
    total_letters = sum(1 for c in converted if c.isalpha() or "\u0600" <= c <= "\u06FF")

    if total_letters == 0:
        return token

    if arabic_chars / total_letters >= 0.60:
        return converted

    return token


def normalize_arabizi(text: str) -> Optional[str]:
    """
    Normalize Arabizi text token by token.

    Returns the normalized string if any Arabizi tokens were found,
    or None if no conversion was performed.
    """
    tokens = text.split()
    any_converted = False
    converted_tokens: list[str] = []

    for token in tokens:
        converted = _convert_arabizi_token(token)
        if converted != token:
            any_converted = True
        converted_tokens.append(converted)

    return " ".join(converted_tokens) if any_converted else None

# DEOBFUSCATIO

def _normalize_homoglyphs(text: str) -> str:
    """Apply the curated homoglyph → ASCII translation table."""
    return text.translate(_HOMOGLYPH_MAP)


def _collapse_spaced_attack(text: str) -> str:
    """
    Collapse spaced/punctuation-separated single-letter sequences.
    """
    def _join(m: re.Match) -> str:
        # Collapse separators inside the matched spaced-letter run.
        collapsed = re.sub(
            r"[ \t\-_./:;,~|@#%^&*!?<>=+`'\"()\[\]{}\\]", "", m.group()
        )
        # Ensure a space follows the collapsed token so surrounding words
        # remain separated (idempotent if there is already a space).
        return collapsed + " "

    result = _SPACED_ATTACK_RE.sub(_join, text)
    # Clean up any double spaces introduced by the above.
    return re.sub(r" {2,}", " ", result).strip()


def _collapse_repeated_chars(text: str) -> str:
    """
    Collapse repeated characters:
      - Letters: 3+ → 1  (e.g. "soooo" → "so", "hellooooo" → "helo")
      - Punctuation: 4+ → 2  (e.g. "!!!!" → "!!", "..." preserved)

    NOTE: 2-char runs like "ss" in "bypass", "ll" in "hello" are intentionally
    preserved — only 3+ repetitions are collapsed.
    """
    text = _REPEATED_LETTERS_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1\1", text)
    return text


def deobfuscate(text: str) -> str:
    """
    Apply all deobfuscation passes in order.

    Pipeline :
      1. Homoglyph substitution (Cyrillic/Greek/IPA/Armenian → ASCII)
      2. Leet-speak deobfuscation (scoped to known attack keywords)
      3. Spaced/punctuation-attack collapse (with space reinjection)
      4. Repeated-character collapse
    """
    text = _normalize_homoglyphs(text)
    text = _apply_leet_deobfuscation(text)
    text = _collapse_spaced_attack(text)
    text = _collapse_repeated_chars(text)
    return text

# SCRIPT PROFIL

def _compute_script_profile(text: str) -> Dict[str, float]:
    """
    Return the fraction of (non-whitespace) characters belonging
    to each major script.

    Keys: "arabic", "latin", "cjk", "other"
    Values: float in [0.0, 1.0] summing to 1.0 (or all 0.0 for empty).
    """
    chars = [c for c in text if not c.isspace()]
    total = len(chars)
    if total == 0:
        return {"arabic": 0.0, "latin": 0.0, "cjk": 0.0, "other": 0.0}

    arabic = sum(1 for c in chars if "\u0600" <= c <= "\u06FF")
    latin  = sum(1 for c in chars if c.isascii() and c.isalpha())
    cjk    = sum(
        1 for c in chars
        if "\u4E00" <= c <= "\u9FFF" or "\u3040" <= c <= "\u30FF"
    )
    other  = total - arabic - latin - cjk

    return {
        "arabic": round(arabic / total, 3),
        "latin":  round(latin  / total, 3),
        "cjk":    round(cjk    / total, 3),
        "other":  round(other  / total, 3),
    }

# DIALECT DETECTIO

def detect_dialect(text: str) -> Tuple[str, float]:
    """
    Detect dialect / language and return a (dialect, confidence) tuple.

    Returned dialect labels:
      "Algerian_dialect"  — Algerian Arabic, Arabizi, or mixed Arabic+Latin
      "French"            — French or Francarabe
      "English"           — English or Latin-script unknown
      "MSA"               — Modern Standard Arabic (by elimination)
      "Unknown"           — emoji/digit/symbol-only input

    """
    lower = text.lower()

    has_arabic = bool(_ARABIC_SCRIPT_RE.search(text))
    has_latin  = bool(_LATIN_SCRIPT_RE.search(lower))

    alg_score = 0
    fr_score  = 0
    en_score  = 0

    # ── Algerian signals ─────────────────────────────────────
    alg_latin_hits = len(_ALGERIAN_WORDS_RE.findall(lower))
    alg_ar_hits    = len(_ALGERIAN_AR_RE.findall(text))
    arabizi_hits   = len(_ARABIZI_DIGIT_ANY_RE.findall(lower))

    if alg_latin_hits:
        alg_score += alg_latin_hits * 2
    if alg_ar_hits:
        alg_score += alg_ar_hits * 2
    if arabizi_hits:
        alg_score += arabizi_hits * 3

    # Mixed script is a strong Algerian indicator
    if has_arabic and has_latin:
        alg_score += 3

    # ── French / English: count ALL matching tokens ──────────
    fr_score += len(_FRENCH_RE.findall(lower))
    en_score += len(_ENGLISH_RE.findall(lower))

    # ── Decision with confidence ──────────────────────────────
    total_score = max(alg_score + fr_score + en_score, 1)

    if alg_score > 0 and alg_score >= max(fr_score, en_score):
        conf = min(alg_score / total_score + 0.1, 1.0)
        return "Algerian_dialect", round(conf, 2)

    if fr_score > 0 and fr_score >= en_score:
        conf = min(fr_score / total_score + 0.05, 1.0)
        return "French", round(conf, 2)

    if en_score > 0:
        conf = min(en_score / total_score + 0.05, 1.0)
        return "English", round(conf, 2)

    if has_arabic:
        if _ALGERIAN_AR_RE.search(text):
            return "Algerian_dialect", 0.65
        
        if len(text.strip()) < 30:
            return "MSA", 0.50
        return "MSA", 0.70

    if has_latin:
        return "English", 0.35

    return "Unknown", 0.0

# SMART TRUNCATIO

def _smart_truncate(text: str, max_len: int) -> Tuple[str, bool]:
    """
    Preserve head + tail rather than hard-cutting at max_len.
    Breaks on whitespace boundaries where possible.

    Must be called AFTER NFKC normalization for stable character positions.
    Returns (result, was_truncated).
    """
    if len(text) <= max_len:
        return text, False

    head_end = _SMART_TRUNC_HEAD
    while head_end > 0 and head_end < len(text) and text[head_end] not in " \n\t":
        head_end -= 1

    tail_start = len(text) - _SMART_TRUNC_TAIL
    while tail_start < len(text) and text[tail_start] not in " \n\t":
        tail_start += 1

    head = text[:head_end].rstrip()
    tail = text[tail_start:].lstrip() if tail_start < len(text) else ""
    result = head + _SMART_TRUNC_SEP + tail if tail else head + _SMART_TRUNC_SEP
    return result, True

# RISK ASSESSMEN

RISK_LOW    = "low"
RISK_MEDIUM = "medium"
RISK_HIGH   = "high"


@dataclass
class RiskReport:
    level:   str
    score:   int
    signals: List[str] = field(default_factory=list)


def _assess_risk(
    original:          str,
    sanitized:         str,
    deobfuscated:      str,
    obfuscation_found: bool,
    script_profile:    Optional[Dict[str, float]] = None,
) -> RiskReport:
    """
    Score input for adversarial risk and return a RiskReport.

    Detection only — never modifies text, never blocks input.

    Scoring table (v3 + P2-6 fixes)
    ──────────────────────────────────────────────────────────────
    Signal                                                Points
    ──────────────────────────────────────────────────────────────
    RTL override character                                  +40
    Prompt injection phrase (visible in sanitized)          +30
    Injection found only after deobfuscation                +35
    Standalone "system prompt" phrase (NEW)                 +30
    Enclosed/regional-indicator emoji spelling words        +25
    Code / shell injection construct (P2-6)                 +35
    Obfuscation detected (homoglyphs / spaced attack)       +15
    Residual spaced-attack pattern in sanitized text        +15
    Leet-speak obfuscation of attack keywords               +20
    High punctuation density in short query (P2-6)          +15
    Homoglyph lookalike characters in original              +10
    Currency-char substitution (¥, €, $) trick              +10
    Arabizi digit markers (ambiguous; low weight)            +5
    ──────────────────────────────────────────────────────────────
    Thresholds: score ≥ 40 → HIGH | ≥ 15 → MEDIUM | else → LOW
    """
    score   = 0
    signals: List[str] = []

    # ── Highest severity: layout-level attacks ────────────────
    if _RTL_OVERRIDE_RE.search(original):
        score += 40
        signals.append("rtl_override_character")

    # ── Emoji-encoded text ────────────────────────────────────
    enclosed_hits = _ENCLOSED_LETTER_RE.findall(original)
    if enclosed_hits:
        score += 25
        signals.append(f"enclosed_emoji_letters({len(enclosed_hits)})")

    # ── Injection in sanitized (plain) text ───────────────────
    if _INJECTION_RE.search(sanitized):
        score += 30
        signals.append("prompt_injection_phrase")

    # ── Injection only visible AFTER deobfuscation ────────────
    if _INJECTION_RE.search(deobfuscated) and not _INJECTION_RE.search(sanitized):
        score += 35
        signals.append("prompt_injection_after_deobfuscation")

    
    _sysprompt_in_sanitized    = bool(_SYSPROMPT_RE.search(sanitized))
    _sysprompt_in_deobfuscated = bool(_SYSPROMPT_RE.search(deobfuscated))
    if _sysprompt_in_sanitized or _sysprompt_in_deobfuscated:
        # Only add if not already counted via _INJECTION_RE to avoid
        # double-counting when the full injection phrase was matched.
        _already_counted = "prompt_injection_phrase" in signals or \
                           "prompt_injection_after_deobfuscation" in signals
        if not _already_counted:
            score += 30
            signals.append("standalone_system_prompt_phrase")

    # ── General obfuscation ───────────────────────────────────
    if obfuscation_found:
        score += 15
        signals.append("obfuscation_detected")

    # ── Residual spaced-attack not fully collapsed ─────────────
    if _SPACED_ATTACK_RE.search(sanitized):
        score += 15
        signals.append("residual_spaced_attack_pattern")

    # ── Code / shell injection constructs (P2-6) ─────────────
    # Checked before leet-speak so that constructs like os.system() are
    # attributed to the correct signal class even when they also happen to
    # match _leet_attack_re (which would otherwise mis-label them as
    # leet_speak_attack_keyword).
    _code_inj_match = _CODE_INJECTION_RE.search(original) or \
                      _CODE_INJECTION_RE.search(deobfuscated)
    if _code_inj_match:
        score += 35
        signals.append("code_injection")

    # ── High punctuation density in short query (P2-6) ───────
    # When the "other" script fraction (punctuation, brackets, operators)
    # exceeds 0.30 in a query under 120 chars, it is a reliable indicator
    # of embedded code or shell payloads (e.g. import("os").system("ls")).
    _sp = script_profile or {}
    _other_frac = _sp.get("other", 0.0)
    if _other_frac > 0.30 and len(original.strip()) < 120:
        score += 15
        signals.append(f"high_punctuation_density(other={_other_frac:.2f})")

    # ── Leet-speak attack keyword obfuscation ─────────────────
    
    # LEET_SPEAK_ALLOWLIST (api, url, sql, gpu …) are skipped.
    if _detect_leet_speak(original):
        score += 20
        signals.append("leet_speak_attack_keyword")

    # ── Currency-character substitution ──────────────────────
    _currency_sub_re = re.compile(r"[¥€$]{1}")
    if _currency_sub_re.search(original):
        score += 10
        signals.append("currency_char_substitution")

    # ── Homoglyph lookalike chars in original ─────────────────
    homoglyph_hits = _HOMOGLYPH_CHARS_RE.findall(original)
    if homoglyph_hits:
        score += 10
        signals.append(f"homoglyph_chars({len(homoglyph_hits)})")

    # ── Arabizi digit markers ─────────────────────────────────
    if _ARABIZI_DIGIT_MARKERS_RE.search(sanitized):
        score += 5
        signals.append("arabizi_digit_markers")

    level = RISK_HIGH if score >= 40 else RISK_MEDIUM if score >= 15 else RISK_LOW
    return RiskReport(level=level, score=score, signals=signals)

# OUTPUT CONTRAC

@dataclass
class SanitizerOutput:
    """
    Complete output of a sanitize_input() call.

    Fields:
      original                      — raw input, never mutated (forensic evidence)
      sanitized                     — after Unicode normalization + invisible-char strip
      deobfuscated                  — after all deobfuscation passes
      dialect                       — detected language/dialect label
      dialect_confidence            — float in [0.0, 1.0] for dialect decision
      script_profile                — fraction of chars per script {"arabic": …, …}
      arabizi_normalized            — Arabizi→Arabic conversion result, or None
      truncated                     — whether input was truncated
      encoding_valid                — whether input is valid UTF-8
      char_count_original           — character count of raw input
      char_count_sanitized          — character count after sanitization
      risk                          — structured RiskReport
      processing_time_ms            — wall-clock time of sanitize_input() in ms
      trace_id                      — caller-provided trace identifier
      user_id                       — caller-provided user identifier
      whitespace_artifact_stripped  — True if Unicode space variants (U+00A0,
                                      narrow NBSP, etc.) were found and removed
                                      before obfuscation detection. Separates
                                      copy-paste UX noise from real obfuscation
                                      signals in metrics (P3-4 fix).
    """
    original:                    str
    sanitized:                   str
    deobfuscated:                str
    dialect:                     str
    dialect_confidence:          float
    script_profile:              Dict[str, float]
    arabizi_normalized:          Optional[str]
    truncated:                   bool
    encoding_valid:              bool
    char_count_original:         int
    char_count_sanitized:        int
    risk:                        RiskReport
    processing_time_ms:          float
    trace_id:                    Optional[str] = None
    user_id:                     Optional[str] = None
    whitespace_artifact_stripped: bool         = False

# MAIN ENTRY POIN

def sanitize_input(
    text: str,
    *,
    trace_id: Optional[str] = None,
    user_id:  Optional[str] = None,
) -> SanitizerOutput:
    """
    Sanitize and analyse a raw user input string.

    Pipeline (v3, in order):
      1.  Type coercion          — any non-str input is str()-cast, never raises
      2.  Encoding check         — informational only, never blocks
      3.  Char count (original)  — recorded before any mutation
      4.  NFKC normalization     — must happen BEFORE truncation for safe slicing
      4.5 Whitespace artifact    — replace Unicode space variants (U+00A0, narrow
          strip (P3-4)             NBSP, etc.) with ASCII space then strip edges.
                                   Runs AFTER NFKC (so fullwidth spaces are already
                                   collapsed) and BEFORE invisible-char strip (step 5)
                                   so the two removal steps are independently logged.
                                   Sets whitespace_artifact_stripped flag on output.
      5.  Invisible char strip   — removes zero-width, bidi, tag chars
                                   NOTE: strip BEFORE repeated-char collapse so
                                   "bypass​safety" → "bypasssafety" not "bypasafety"
      6.  Arabic diacritic strip — normalization for Arabic text
      7.  Smart truncation       — head+tail preserve, not hard cut
      8.  Deobfuscation          — homoglyphs → leet → spaced-attack → repeated
      9.  Dialect detection      — with confidence score
                                   NOTE: deobfuscated text is .strip()-ped
                                   before classification (not before storage)
                                   so leading/trailing whitespace cannot
                                   suppress Darja marker matching on short
                                   queries (P3-3 fix).
      10. Arabizi normalization  — token-level comparison, None if no Arabizi found
      11. Script profile         — character-level script distribution
      12. Risk assessment        — additive scoring, no blocking
      13. Structured logging     — all signals emitted to logger
    """
    _t_start = time.monotonic()

    # ── 1. Type coercion ─────────────────────────────────────
    if not isinstance(text, str):
        text = str(text)

    original = text

    # ── 2. Encoding check (informational only) ───────────────
    try:
        text.encode("utf-8").decode("utf-8")
        encoding_valid = True
    except Exception:
        encoding_valid = False

    # ── 3. Original char count ───────────────────────────────
    char_count_original = len(original)

    # ── 4. NFKC normalization ────────────────────────────────
    # Handles fullwidth chars (ｉｇｎｏｒｅ → ignore), bold Unicode, etc.
    # Must happen BEFORE truncation so slice positions are stable.
    text = unicodedata.normalize("NFKC", text)

    # ── 4.5 Whitespace artifact strip (P3-4) ─────────────────
    # Replace Unicode space variants (U+00A0 non-breaking space, narrow NBSP,
    # en/em-space, ideographic space, etc.) with plain ASCII spaces, then
    # strip leading/trailing ASCII whitespace.
    #
    # WHY HERE: After NFKC (so fullwidth spaces are already resolved) but
    # BEFORE the invisible-char strip (step 5) so the two distinct removal
    # steps are independently observable in logs and metrics.  This lets
    # operators distinguish:
    #   • whitespace_artifact_stripped=True  → copy-paste UX noise (U+00A0 etc.)
    #   • obfuscation_found=True             → genuine adversarial obfuscation
    # Previously these were conflated: a trailing U+00A0 caused
    # obfuscation_detected=True on pass 1 then False on pass 2 after a 1-char
    # strip, inflating obfuscation metrics (~11% of runs in affected traces).
    text, whitespace_artifact_stripped = strip_whitespace_artifacts(text)

    # ── 5. Invisible / control char strip ────────────────────
    # Must happen BEFORE repeated-char collapse.
    # "bypass" + ZWJ + "safety" → after strip = "bypasssafety"
    # → repeated-char collapse sees "ss" (2 chars, not 3+) → keeps "ss"
    # → correct result "bypasssafety".
    text = _INVISIBLE_RE.sub("", text)

    # ── 6. Arabic diacritic strip ────────────────────────────
    text = _AR_DIACRITICS_RE.sub("", text)

    sanitized = text

    # ── 7. Smart truncation ──────────────────────────────────
    text, truncated = _smart_truncate(text, _MAX_LEN)
    sanitized = text

    # ── 8. Deobfuscation ─────────────────────────────────────
    deobfuscated      = deobfuscate(sanitized)
    obfuscation_found = sanitized != deobfuscated

    # ── 9. Dialect detection ─────────────────────────────────
    
    dialect, dialect_confidence = detect_dialect(deobfuscated.strip())

    # ── 10. Arabizi normalization ────────────────────────────
    arabizi_normalized: Optional[str] = normalize_arabizi(deobfuscated)

    # ── 11. Script profile ───────────────────────────────────
    script_profile = _compute_script_profile(deobfuscated)

    # ── 12. Risk assessment ──────────────────────────────────
    risk = _assess_risk(original, sanitized, deobfuscated, obfuscation_found, script_profile)

    # ── 13. Logging ──────────────────────────────────────────
    processing_time_ms = round((time.monotonic() - _t_start) * 1000, 3)

    logger.debug(
        "[SANITIZER] trace={} | user={} | dialect={} (conf={}) | "
        "risk={} (score={}, signals={}) | script={} | "
        "truncated={} | encoding_valid={} | "
        "obfuscation={} | ws_artifact_stripped={} | "
        "chars_in={} chars_out={} | time_ms={}",
        trace_id,
        user_id,
        dialect,
        dialect_confidence,
        risk.level,
        risk.score,
        risk.signals,
        script_profile,
        truncated,
        encoding_valid,
        obfuscation_found,
        whitespace_artifact_stripped,
        char_count_original,
        len(sanitized),
        processing_time_ms,
    )

    return SanitizerOutput(
        original                     = original,
        sanitized                    = sanitized,
        deobfuscated                 = deobfuscated,
        dialect                      = dialect,
        dialect_confidence           = dialect_confidence,
        script_profile               = script_profile,
        arabizi_normalized           = arabizi_normalized,
        truncated                    = truncated,
        encoding_valid               = encoding_valid,
        char_count_original          = char_count_original,
        char_count_sanitized         = len(sanitized),
        risk                         = risk,
        processing_time_ms           = processing_time_ms,
        trace_id                     = trace_id,
        user_id                      = user_id,
        whitespace_artifact_stripped = whitespace_artifact_stripped,
    )
