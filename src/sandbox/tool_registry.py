# =============================================================
# src/sandbox/tool_registry.py
# Tool Registry used by  AMAN-LLM Agent Core
#




# Imports — guarded for Colab compatibility

import math

# [P2-4] Freshness validator
try:
    from src.layer3.web_search_freshness_validator import (
        validate_web_search_freshness,
        annotate_search_output,
        FreshnessResult,
    )
    _FRESHNESS_AVAILABLE = True
except ImportError:
    _FRESHNESS_AVAILABLE = False
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "web_search_freshness_validator not found — "
        "freshness validation disabled (P2-4)."
    )

# LangChain Tool import — location changed in langchain >= 0.2
try:
    from langchain_core.tools import Tool          # langchain >= 0.2
except ImportError:
    try:
        from langchain.tools import Tool           # langchain < 0.2
    except ImportError:
        Tool = None                                # LangChain not installed at all

# Docker executor — only available when Docker daemon is present (not in Colab)
try:
    from .docker_executor import execute_in_sandbox
    _SANDBOX_AVAILABLE = True
except Exception:
    _SANDBOX_AVAILABLE = False
    def execute_in_sandbox(tool_type: str, payload: dict) -> str:
        return (
            f"[STUB {tool_type}] Docker sandbox not available in this environment.\n"
            f"Input: {payload}"
        )



# Tool 1 — Web Search


def _web_search_raw(query: str) -> str:
    """
    Bare Tavily web search — returns the raw result string with no freshness
    annotation.  Used by unit tests and by _web_search() internally.
    """
    try:
        from tavily import TavilyClient
        import os
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        resp = client.search(query, max_results=5)
        results = resp.get("results", [])
        if not results:
            return f"[web_search] No results for: {query}"
        lines = [f"[web_search] Results for: {query}"]
        for r in results:
            lines.append(
                f"  • {r.get('title', '').strip()}: "
                f"{r.get('content', '').strip()[:200]}  "
                f"({r.get('url', '')})"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"[web_search] Error: {e}"


def _web_search(query: str) -> str:
    """
    Tavily web search with P2-4 freshness validation.

    Wraps _web_search_raw() with:
      1. A [web_search_meta ...] header embedded in the returned string so
         that Layer 3 Check 5c can parse staleness without re-running the
         validator.
      2. An Arabic + English staleness disclaimer prepended when the content
         is older than the per-type threshold (weather: 12 h, price: 1 h,
         news: 24 h, generic: 72 h).
      3. A ``should_requote`` hint logged at WARNING level so operators can
         see when a re-query was warranted.

    Falls back to _web_search_raw() output unchanged when the freshness
    module is unavailable (_FRESHNESS_AVAILABLE=False).
    """
    import logging
    _log = logging.getLogger(__name__)

    raw_result = _web_search_raw(query)

    if not _FRESHNESS_AVAILABLE:
        return raw_result

    try:
        freshness: FreshnessResult = validate_web_search_freshness(
            query=query,
            result_text=raw_result,
        )

        if freshness.failed_checks:
            _log.warning(
                "[P2-4] web_search freshness check: query=%r | data_type=%s | "
                "stale=%s | age_h=%s | checks=%s | should_requote=%s",
                query[:80],
                freshness.data_type,
                freshness.is_stale,
                freshness.staleness_h,
                freshness.failed_checks,
                freshness.should_requote,
            )

        return annotate_search_output(raw_result, freshness, include_disclaimer=True)

    except Exception as exc:
        _log.error(
            "[P2-4] freshness validation error (returning raw result): %s", exc
        )
        return raw_result



# Tool 2 — Calculator (safe arithmetic evaluator)
# Mirrors the notebook's _calculator exactly so both paths
# produce identical output for the same input.

def _calculator(expression: str) -> str:
    """
    Safe arithmetic evaluator.

    Normalises Arabic-Indic digits (٠١٢..٩) and Arabic math symbols
    (× ÷) before evaluating, so inputs like '١٤٤ × ٢٥' work correctly.
    Guards against complex results (e.g. sqrt(-1)).

    No sandbox needed — pure Python, no I/O, no imports.
    """
    # Normalise Arabic-Indic digits → ASCII
    arabic_indic = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    expression = expression.translate(arabic_indic)

    # Normalise Arabic math symbols
    expression = (
        expression
        .replace("×", "*")
        .replace("÷", "/")
        .replace("،", ",")
        .strip()
    )

    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "TOOL_ERROR: تعبير غير مسموح به"

    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        if isinstance(result, complex):
            return "TOOL_ERROR: النتيجة عدد مركّب (مثلاً جذر عدد سالب)"
        return f"النتيجة: {result}"
    except Exception as e:
        return f"TOOL_ERROR: {str(e)}"



# Tool 3 — Python Executor (sandboxed Python)

def _python_executor(code: str) -> str:
    if not _SANDBOX_AVAILABLE:
        return "[python_executor] Docker sandbox not available in Colab. Code execution skipped."
    return execute_in_sandbox("code", {"script": code})



# Tool 4 — Vector Search (semantic document retrieval)

def _vector_search(query: str) -> str:
    if not _SANDBOX_AVAILABLE:
        return f"[vector_search] No vector DB configured. Query was: {query}"
    return execute_in_sandbox("vector_search", {"query": query})



# Tool 5 — Database Query (read-only)


def _database_query(query: str) -> str:
    if not _SANDBOX_AVAILABLE:
        return f"[database_query] No database configured. Query was: {query}"
    return execute_in_sandbox("database", {"query": query})



# Public API — PATH 1: LangChain AgentExecutor (pipeline.py)

def get_registered_tools():
    """
    Return the 5 LangChain Tool objects for the agent executor.

    Tool names match the notebook's TOOL_REGISTRY exactly so that
    intent → tool routing is consistent across both integration paths.
    """
    if Tool is None:
        raise ImportError(
            "langchain or langchain-core is not installed. "
            "Run: pip install langchain-core"
        )
    return [
        Tool(
            name="web_search",
            func=_web_search,
            description=(
                "Search the web for current information, recent news, "
                "or up-to-date facts. Input: a search query string."
            ),
        ),
        Tool(
            name="calculator",
            func=_calculator,
            description=(
                "Evaluate arithmetic expressions, including Arabic-Indic "
                "digits and Arabic math symbols (× ÷). "
                "Input: a mathematical expression as a string."
            ),
        ),
        Tool(
            name="python_executor",
            func=_python_executor,
            description=(
                "Execute Python code in a sandboxed Docker environment. "
                "Use for programming tasks, data processing, or algorithms. "
                "Input: valid Python code as a string."
            ),
        ),
        Tool(
            name="vector_search",
            func=_vector_search,
            description=(
                "Perform semantic search over the internal knowledge base. "
                "Use for document retrieval or context lookup. "
                "Input: a natural-language query string."
            ),
        ),
        Tool(
            name="database_query",
            func=_database_query,
            description=(
                "Query the structured database for historical records "
                "or statistical data (read-only, approved schemas only). "
                "Input: a natural-language or SQL query string."
            ),
        ),
    ]
