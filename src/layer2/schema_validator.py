
# AMAN-LLM
# Layer 2 — Policy Engine / Tool Argument Schema Validation

#
# ## PURPOSE
#
# This module implements the schema-validation stage of the Layer 2 policy
# engine described in the AMAN-LLM architecture. It validates the *shape and
# content* of every PlannedToolCall produced by the LLM Agent Core before the
# call reaches the Layer 2 policy rule engine. It performs two distinct
# checks:
#
#   1. STRUCTURAL VALIDATION — Pydantic verifies that required fields are
#      present, that field types are correct, and that values fall within
#      declared bounds. A structurally invalid call is rejected immediately
#      with a schema_violation outcome, before any policy rule (regex-based
#      or otherwise) is evaluated.
#
#   2. SEMANTIC PRE-CHECKS — field-level and model-level validators enforce
#      domain constraints that plain Pydantic types cannot express on their
#      own (e.g. URL scheme allowlisting, path character restrictions, query
#      length limits, script size limits).
#
# ## ARCHITECTURAL ROLE
#
# This module sits between the LLM Agent Core (which proposes tool calls)
# and the Layer 2 policy rule engine (which applies allow/deny rules such as
# PR-02, PR-03, PR-04, PR-09, PR-10; these policy rules are implemented
# elsewhere and are referenced here only by identifier). The design
# separates two concerns:
#
#   • Structure — "is this call well-formed?" (this module)
#   • Policy    — "is this well-formed call allowed?" (the rule engine)
#
# Schema validators in this module are therefore ADDITIVE to, and not a
# substitute for, the policy rule engine: several checks performed here
# (e.g. SQL statement-type restriction, path traversal exclusion, dangerous
# token filtering) are deliberately duplicated as independent policy rules
# downstream, so that the two layers provide defence-in-depth rather than a
# single point of enforcement.
#
# ## INPUTS
#
# A tool name (str) and the corresponding raw argument dictionary, as
# produced by the LLM Agent Core's planned tool call.
#
# ## OUTPUTS
#
# A ValidationResult indicating whether the call is structurally valid and,
# if not, a list of human-readable errors together with a structured error
# map intended for consumption by the Audit Logger.
#
# ## DESIGN PRINCIPLES
#
#   • One Pydantic model per tool — each model is the single source of truth
#     for what constitutes a well-formed call to that tool.
#   • All models set model_config = ConfigDict(extra="forbid"), so that
#     arguments not declared in the schema are rejected. This reduces the
#     risk of argument smuggling via undeclared keys.
#   • validate_tool_call() is the single public entry point consumed by
#     policy_engine.run_layer2().
#
# ## RETURN CONTRACT
#
#   ValidationResult.valid = True  → the call is structurally sound; the
#                                    Layer 2 rule engine evaluates it next.
#   ValidationResult.valid = False → the call is rejected immediately with a
#                                    schema_violation outcome.
#     .errors  list[str] — human-readable, field-level error messages.
#     .details dict      — structured field → error mapping, intended for
#                           the Audit Logger.
#
# ## IMPLEMENTATION NOTES
#
# This module performs schema-level checks only; it does not execute tool
# calls and does not itself decide whether a structurally valid call is
# policy-compliant. Detection and enforcement logic beyond structural and
# stated semantic constraints (e.g. broader prompt-injection detection) is
# out of scope for this module.
#
# ## REPRODUCIBILITY
#
# This module has no external runtime dependencies beyond Pydantic and the
# Python standard library `re` module. Tool-specific numeric and length
# constants are defined as module-level constants below; their values are
# configurable at the source level. Not specified in this module: Python
# version and Pydantic version requirements (see repository-level
# dependency manifest).


from __future__ import annotations

import re
from typing import Any, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Return type


@dataclass
class ValidationResult:
    """
    Structural-validation outcome returned by validate_tool_call() to the
    Layer 2 policy engine.

    Attributes
    ----------
    valid : bool
        True if the argument payload satisfies its Pydantic schema and all
        associated field-/model-level validators; False otherwise.
    errors : list[str]
        Human-readable, field-level validation error messages. Empty when
        valid is True.
    details : dict[str, Any]
        Structured field → error mapping, intended for the Audit Logger.
        Empty when valid is True.
    """
    valid:   bool
    errors:  list[str]          = field(default_factory=list)
    details: dict[str, Any]     = field(default_factory=dict)



# Shared constants

#
# The byte-size limits below bound argument payloads per tool. They are
# enforced independently of, and in addition to, any length constraint
# implemented by policy rule PR-06 in the Layer 2 rule engine, providing a
# second enforcement point for the same class of constraint (defence in
# depth). The specific numeric values are configurable; their calibration
# procedure, if any, is not specified in this module.

_MAX_QUERY_BYTES   = 8_192    # database_query: maximum query string size (UTF-8 bytes)
_MAX_SCRIPT_BYTES  = 16_384   # code_interpreter / python_executor: maximum script size (UTF-8 bytes)
_MAX_PATH_BYTES    = 1_024    # file_read: maximum path length (UTF-8 bytes)
_MAX_TEXT_BYTES    = 32_768   # translation / web_search / vector_search: maximum text size (UTF-8 bytes)
_MAX_URL_BYTES     = 2_048    # web_fetch: maximum URL length (UTF-8 bytes)

# URL schemes accepted for web-access tools. Restricting to http/https
# excludes schemes such as file:// or data://, which could otherwise be used
# to reference local resources or embed arbitrary payloads in place of a
# genuine remote fetch.
_ALLOWED_URL_SCHEMES = {"http", "https"}

# Characters excluded from file paths at the schema level: ASCII control
# characters, DEL, and shell metacharacters (; | & ` $). This check is
# independent of, and narrower in scope than, the path-traversal and
# path-manipulation policy rule PR-02 enforced downstream in the rule
# engine; it addresses malformed or shell-significant characters rather
# than traversal sequences such as "../".
_BANNED_PATH_CHARS_RE = re.compile(r'[\x00-\x1f\x7f;|&`$]')

# Matches queries beginning with a read-only SQL statement keyword. SELECT
# and WITH (common table expressions) are standard data-read forms.
# SHOW/DESCRIBE/DESC are also syntactically read-only but are treated as
# reconnaissance and blocked as a policy matter by rule PR-03 in the rule
# engine; this schema-level check only constrains the query to begin with
# one of these five keywords and does not itself distinguish between them.
_SQL_SELECT_RE = re.compile(r'^\s*(SELECT|WITH|SHOW|DESCRIBE|DESC)\b', re.IGNORECASE)

# Matches a Windows drive-letter absolute path (e.g. "C:\"), used together
# with a leading "/" check to determine whether a file_read path is
# absolute. Windows-style paths that pass this check are still considered
# structurally well-formed by this module; rejecting them (if required) is
# a policy decision made by rule PR-02 in the rule engine, not by schema
# validation.
_WINDOWS_ABS_PATH_RE = re.compile(r'^[A-Za-z]:\\')



# Per-tool argument schemas
#
# Each class below is a Pydantic model defining the structural and semantic
# schema for one tool's arguments. All models are registered in
# _TOOL_SCHEMA_MAP and are consumed exclusively through validate_tool_call().


class WebSearchArgs(BaseModel):
    """
    Argument schema for the web_search tool.

    Represents a read-only web search request. The schema does not permit
    URL fetching or the inclusion of authentication material; those are out
    of scope for this tool.
    """
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The search query string.",
    )
    max_results: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return (1–20).",
    )
    language: Optional[str] = Field(
        default="ar",
        pattern=r'^[a-z]{2}(-[A-Z]{2})?$',
        description="BCP-47 language tag, e.g. 'ar', 'en', 'fr'.",
    )

    @field_validator("query")
    @classmethod
    def query_byte_limit(cls, v: str) -> str:
        """Reject queries whose UTF-8 encoding exceeds _MAX_TEXT_BYTES."""
        if len(v.encode("utf-8")) > _MAX_TEXT_BYTES:
            raise ValueError(
                f"query exceeds {_MAX_TEXT_BYTES} bytes — "
                "use a shorter search string."
            )
        return v


class DatabaseQueryArgs(BaseModel):
    """
    Argument schema for the database_query tool.

    Only SELECT and WITH (common table expression) statements are
    structurally valid, restricting the tool to read-only queries at the
    schema level. Write-statement keywords are rejected here and are also
    checked independently by policy rule PR-03 in the rule engine, so that
    the restriction to read-only queries does not depend on a single
    enforcement point.
    """
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=7,       # Shortest structurally valid query: "SELECT 1"
        description="SQL query string — must begin with SELECT or WITH.",
    )
    database: Optional[str] = Field(
        default=None,
        pattern=r'^[a-zA-Z0-9_\-]{1,64}$',
        description="Target database/schema name (alphanumeric + _ -).",
    )
    timeout_seconds: Optional[int] = Field(
        default=10,
        ge=1,
        le=30,
        description="Query execution timeout (1–30 seconds).",
    )

    @field_validator("query")
    @classmethod
    def must_be_select(cls, v: str) -> str:
        """Reject queries that do not begin with a read-only statement keyword."""
        if not _SQL_SELECT_RE.match(v):
            raise ValueError(
                "Only SELECT / WITH (CTE) queries are permitted. "
                "Write operations (INSERT, UPDATE, DELETE, DROP, …) are blocked."
            )
        return v

    @field_validator("query")
    @classmethod
    def query_byte_limit(cls, v: str) -> str:
        """Reject queries whose UTF-8 encoding exceeds _MAX_QUERY_BYTES."""
        if len(v.encode("utf-8")) > _MAX_QUERY_BYTES:
            raise ValueError(
                f"query exceeds {_MAX_QUERY_BYTES} bytes — "
                "break into smaller, scoped queries."
            )
        return v


class CodeInterpreterArgs(BaseModel):
    """
    Argument schema for the code_interpreter tool (retained for backward
    compatibility; see PythonExecutorArgs for the current tool registration).

    Accepts Python source code only. The allow_network field, if present,
    is constrained to False at the schema level, reflecting the sandbox's
    lack of outbound network access. This schema does not itself inspect
    the script body for OS-command patterns; that inspection is performed
    independently by policy rule PR-04 in the rule engine.
    """
    model_config = ConfigDict(extra="forbid")

    script: str = Field(
        ...,
        min_length=1,
        description="Python source code to execute in the sandbox.",
    )
    timeout_seconds: Optional[int] = Field(
        default=15,
        ge=1,
        le=60,
        description="Execution timeout (1–60 seconds).",
    )
    allow_network: Optional[bool] = Field(
        default=False,
        description="Must be False — network is always disabled in sandbox.",
    )

    @field_validator("script")
    @classmethod
    def script_byte_limit(cls, v: str) -> str:
        """Reject scripts whose UTF-8 encoding exceeds _MAX_SCRIPT_BYTES."""
        if len(v.encode("utf-8")) > _MAX_SCRIPT_BYTES:
            raise ValueError(
                f"script exceeds {_MAX_SCRIPT_BYTES} bytes — "
                "split into smaller, focused scripts."
            )
        return v

    @field_validator("allow_network")
    @classmethod
    def network_must_be_disabled(cls, v: bool) -> bool:
        """Reject any explicit request for network access in the sandbox."""
        if v is True:
            raise ValueError(
                "allow_network must be False — "
                "the code sandbox has no outbound network access."
            )
        return v


class FileReadArgs(BaseModel):
    """
    Argument schema for the file_read tool.

    The path field must be a non-empty, absolute path free of the
    characters excluded by _BANNED_PATH_CHARS_RE. This schema does not
    itself verify that the path lies within an approved working directory
    or check for path-traversal sequences; those checks, together with
    detection of sensitive file-path patterns, are performed independently
    by policy rule PR-10 in the rule engine.
    """
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        min_length=1,
        description="Absolute path to the file to read.",
    )
    encoding: Optional[str] = Field(
        default="utf-8",
        pattern=r'^[a-zA-Z0-9\-_]{1,20}$',
        description="File encoding (default utf-8).",
    )
    max_bytes: Optional[int] = Field(
        default=65_536,
        ge=1,
        le=1_048_576,        # 1 MB hard cap
        description="Maximum bytes to read (1 B – 1 MB).",
    )

    @field_validator("path")
    @classmethod
    def path_byte_limit(cls, v: str) -> str:
        """Reject paths whose UTF-8 encoding exceeds _MAX_PATH_BYTES."""
        if len(v.encode("utf-8")) > _MAX_PATH_BYTES:
            raise ValueError(f"path exceeds {_MAX_PATH_BYTES} bytes.")
        return v

    @field_validator("path")
    @classmethod
    def no_banned_chars(cls, v: str) -> str:
        """Reject paths containing a character matched by _BANNED_PATH_CHARS_RE."""
        match = _BANNED_PATH_CHARS_RE.search(v)
        if match:
            raise ValueError(
                f"Path contains disallowed character "
                f"(0x{ord(match.group()):02X}) at position {match.start()}."
            )
        return v

    @field_validator("path")
    @classmethod
    def must_be_absolute(cls, v: str) -> str:
        """Reject relative paths; only Unix- or Windows-style absolute paths pass."""
        if not (v.startswith("/") or _WINDOWS_ABS_PATH_RE.match(v)):
            raise ValueError(
                "path must be absolute (start with '/' or a drive letter, "
                "e.g. 'C:\\') — relative paths are not permitted."
            )
        return v


class TranslationArgs(BaseModel):
    """
    Argument schema for the translation tool.

    Supports translation between Arabic dialects and Modern Standard Arabic
    (MSA), as well as other BCP-47-tagged languages, depending on the
    capabilities of the underlying translation backend (not specified in
    this module).
    """
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description="Text to translate.",
    )
    source_language: Optional[str] = Field(
        default=None,
        pattern=r'^[a-z]{2}(-[A-Z]{2})?$',
        description="BCP-47 source language tag (auto-detected if omitted).",
    )
    target_language: str = Field(
        default="ar",
        pattern=r'^[a-z]{2}(-[A-Z]{2})?$',
        description="BCP-47 target language tag.",
    )

    @field_validator("text")
    @classmethod
    def text_byte_limit(cls, v: str) -> str:
        """Reject text whose UTF-8 encoding exceeds _MAX_TEXT_BYTES."""
        if len(v.encode("utf-8")) > _MAX_TEXT_BYTES:
            raise ValueError(
                f"text exceeds {_MAX_TEXT_BYTES} bytes — "
                "split into smaller segments for translation."
            )
        return v

    @model_validator(mode="after")
    def source_target_differ(self) -> "TranslationArgs":
        """
        Reject requests where source_language and target_language are equal.

        When source_language is explicitly provided and matches
        target_language, the requested translation is a no-op; such calls
        are rejected as structurally invalid.
        """
        if (
            self.source_language is not None
            and self.source_language == self.target_language
        ):
            raise ValueError(
                "source_language and target_language must differ — "
                "translating a language to itself is a no-op."
            )
        return self


class WebFetchArgs(BaseModel):
    """
    Argument schema shared by the web_fetch, external_request, and browser
    tools (see _TOOL_SCHEMA_MAP).

    Restricts the URL scheme to http/https and excludes selected
    authentication-related headers from the request. This schema does not
    itself resolve or classify the target host; detection of requests
    directed at internal network targets (server-side request forgery, or
    SSRF) is performed independently by policy rule PR-09 in the rule
    engine.
    """
    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        ...,
        min_length=7,
        description="Fully-qualified URL to fetch (http/https only).",
    )
    method: Optional[str] = Field(
        default="GET",
        pattern=r'^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)$',
        description="HTTP method (default GET).",
    )
    headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional HTTP headers — must not include Authorization.",
    )
    body: Optional[str] = Field(
        default=None,
        max_length=16_384,
        description="Optional request body (POST/PUT/PATCH only).",
    )
    timeout_seconds: Optional[int] = Field(
        default=10,
        ge=1,
        le=30,
        description="Request timeout (1–30 seconds).",
    )

    @field_validator("url")
    @classmethod
    def url_byte_limit(cls, v: str) -> str:
        """Reject URLs whose UTF-8 encoding exceeds _MAX_URL_BYTES."""
        if len(v.encode("utf-8")) > _MAX_URL_BYTES:
            raise ValueError(f"url exceeds {_MAX_URL_BYTES} bytes.")
        return v

    @field_validator("url")
    @classmethod
    def allowed_scheme_only(cls, v: str) -> str:
        """Reject URLs whose scheme is not in _ALLOWED_URL_SCHEMES (http, https)."""
        scheme = v.split("://")[0].lower() if "://" in v else ""
        if scheme not in _ALLOWED_URL_SCHEMES:
            raise ValueError(
                f"URL scheme '{scheme}' is not permitted — "
                f"only {sorted(_ALLOWED_URL_SCHEMES)} are allowed."
            )
        return v

    @field_validator("headers")
    @classmethod
    def no_auth_header(cls, v: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
        """
        Reject requests carrying selected authentication-related headers
        (Authorization, X-Api-Key, X-Auth-Token), case-insensitively, so
        that credential material is not passed through outbound tool calls.
        """
        if v is None:
            return v
        for key in v:
            if key.lower() in {"authorization", "x-api-key", "x-auth-token"}:
                raise ValueError(
                    f"Header '{key}' is not permitted — "
                    "authentication headers must not be passed through tool calls."
                )
        return v



class CalculatorArgs(BaseModel):
    """
    Argument schema for the calculator tool.

    The agent routes arithmetic expressions to this schema as a plain
    string, via LangChain's Tool(func=_calculator). Layer 2 receives the
    call in the form {"expression": "<math string>"}; the agent layer is
    responsible for wrapping the raw expression string in this key before
    invoking run_layer2().

    Arabic-Indic digit normalisation and × / ÷ symbol handling are
    performed by tool_registry._calculator(), not by this schema; this
    schema validates only the argument envelope (presence, length, and
    absence of disallowed tokens).
    """
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Arithmetic expression to evaluate, e.g. '1547 * 83 / 7' "
            "or '١٤٤ × ٢٥'.  Supports +  -  *  /  **  ()  %  and "
            "numeric literals only."
        ),
    )
    precision: Optional[int] = Field(
        default=None,
        ge=0,
        le=15,
        description="Decimal places to round the result to (0–15). Omit for full precision.",
    )

    @field_validator("expression")
    @classmethod
    def expression_byte_limit(cls, v: str) -> str:
        """Reject expressions whose UTF-8 encoding exceeds 500 bytes."""
        if len(v.encode("utf-8")) > 500:
            raise ValueError(
                "expression exceeds 500 bytes — simplify or split the calculation."
            )
        return v

    @field_validator("expression")
    @classmethod
    def no_dangerous_tokens(cls, v: str) -> str:
        """
        Reject expressions containing Python execution primitives (e.g.
        import, exec, eval, open, os., sys., subprocess, dunder attribute
        access) or shell metacharacters.

        tool_registry._calculator() independently restricts input to an
        allowed character set; this schema-level check provides an
        additional enforcement point that rejects the call before it
        reaches injection-detection logic further down the pipeline,
        yielding a schema_violation audit entry rather than a
        detection-stage entry for this class of input.
        """
        _DANGEROUS_EXPR_RE = re.compile(
            r"(__|import\b|exec\b|eval\b|open\b|os\.|sys\.|subprocess"
            r"|[;&|`$\\])",
            re.IGNORECASE,
        )
        m = _DANGEROUS_EXPR_RE.search(v)
        if m:
            raise ValueError(
                f"expression contains disallowed token '{m.group()}' — "
                "only arithmetic operators and numeric literals are permitted."
            )
        return v
    
class PythonExecutorArgs(BaseModel):
    """
    Argument schema for the python_executor tool (Docker-sandboxed Python
    execution). This tool name replaces code_interpreter in tool_registry
    and the notebook TOOL_REGISTRY; CodeInterpreterArgs is retained above
    for backward compatibility with callers still using the earlier name.

    Accepts the argument under either 'script' (canonical) or 'code'
    (legacy alias), so that callers using either key are not rejected on a
    key-name mismatch alone. Internally, the canonical field is always
    'script'; a 'code' value is normalised to 'script' before the remaining
    validators run (see normalise_code_alias below).
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    script: str = Field(
        ...,
        min_length=1,
        alias="script",
        description="Python source code to execute in the Docker sandbox.",
    )
    # 'code' is a legacy argument key: earlier pipeline versions sent
    # {"code": "..."} rather than {"script": "..."}. It is accepted here
    # and normalised to 'script' by normalise_code_alias so that Layer 2
    # does not reject a call solely because of the older key name.
    # Candidate for removal once all callers have migrated to 'script'.
    code: Optional[str] = Field(
        default=None,
        description="[DEPRECATED] Use 'script' instead. Accepted for backward compat.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalise_code_alias(cls, values: dict) -> dict:
        """
        Normalise the legacy 'code' key to the canonical 'script' key.

        If the caller supplied 'code' but not 'script', 'code' is promoted
        to 'script'. If both are present, the legacy 'code' key is dropped
        and the 'script' value is kept. Runs before field-level validation,
        so downstream validators always observe the canonical 'script'
        field.
        """
        if "code" in values and "script" not in values:
            values["script"] = values.pop("code")
        elif "code" in values:
            values.pop("code")   # both present: drop legacy key, keep script
        return values
    timeout_seconds: Optional[int] = Field(
        default=15,
        ge=1,
        le=60,
        description="Execution timeout (1–60 seconds).",
    )
    allow_network: Optional[bool] = Field(
        default=False,
        description="Must be False — sandbox has no outbound network access.",
    )

    @field_validator("script")
    @classmethod
    def script_byte_limit(cls, v: str) -> str:
        """Reject scripts whose UTF-8 encoding exceeds _MAX_SCRIPT_BYTES."""
        if len(v.encode("utf-8")) > _MAX_SCRIPT_BYTES:
            raise ValueError(
                f"script exceeds {_MAX_SCRIPT_BYTES} bytes — "
                "split into smaller, focused scripts."
            )
        return v

    @field_validator("allow_network")
    @classmethod
    def network_must_be_disabled(cls, v: bool) -> bool:
        """Reject any explicit request for network access in the sandbox."""
        if v is True:
            raise ValueError(
                "allow_network must be False — "
                "the Docker sandbox has no outbound network access."
            )
        return v


class VectorSearchArgs(BaseModel):
    """
    Argument schema for the vector_search tool (semantic document
    retrieval).

    Accepts a natural-language query string; the retrieval backend
    (e.g. Chroma, FAISS, or Pinecone) is selected and configured elsewhere
    and is not determined by this schema.
    """
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural-language query for semantic document retrieval.",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of nearest-neighbour results to return (1–50).",
    )
    collection: Optional[str] = Field(
        default=None,
        pattern=r'^[a-zA-Z0-9_\-]{1,64}$',
        description="Target vector collection/index name (alphanumeric + _ -).",
    )

    @field_validator("query")
    @classmethod
    def query_byte_limit(cls, v: str) -> str:
        """Reject queries whose UTF-8 encoding exceeds _MAX_TEXT_BYTES."""
        if len(v.encode("utf-8")) > _MAX_TEXT_BYTES:
            raise ValueError(
                f"query exceeds {_MAX_TEXT_BYTES} bytes — use a shorter query."
            )
        return v

# Tool registry — maps tool_name to its argument model class.
#
# validate_tool_call() consults this mapping to select the schema against
# which a given tool call is validated. Aliased entries (e.g.
# external_request and browser mapping to WebFetchArgs) indicate multiple
# tool names sharing one argument schema.


_TOOL_SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "web_search":        WebSearchArgs,
    "database_query":    DatabaseQueryArgs,
    "code_interpreter":  CodeInterpreterArgs,   # Retained for backward compatibility.
    "python_executor":   PythonExecutorArgs,    # Current name, matches tool_registry.
    "file_read":         FileReadArgs,
    "translation":       TranslationArgs,
    "web_fetch":         WebFetchArgs,
    "external_request":  WebFetchArgs,          # Alias of web_fetch.
    "browser":           WebFetchArgs,          # Alias of web_fetch.
    "calculator":        CalculatorArgs,
    "vector_search":     VectorSearchArgs,
}



# Public entry point


def validate_tool_call(tool_name: str, arguments: dict[str, Any]) -> ValidationResult:
    """
    Validate a planned tool call against its registered Pydantic schema.

    This is the single public entry point of the schema-validation stage
    and is consumed by policy_engine.run_layer2(). It performs no policy
    evaluation itself: a call that passes this check is structurally
    well-formed and is then evaluated by the Layer 2 rule engine.

    Parameters
    ----------
    tool_name : str
        Name of the tool the LLM Agent Core proposes to invoke.
    arguments : dict[str, Any]
        Raw argument payload from the agent's planned tool call.

    Returns
    -------
    ValidationResult
        valid=True  — the call is structurally sound; the rule engine
        evaluates it next.
        valid=False — the call is rejected immediately with a
        schema_violation outcome, together with human-readable errors and
        an Audit-Logger-oriented error map.

    Notes
    -----
    An unrecognised tool_name is treated as invalid: schema validation
    requires an explicit, registered schema and does not fall back to an
    unrestricted default.
    """
    schema_class = _TOOL_SCHEMA_MAP.get(tool_name)

    # ── Unknown tool ──────────────────────────────────────────────────────
    # A tool name absent from the registry has no corresponding schema, and
    # is therefore rejected rather than validated against a default or
    # permissive schema.
    if schema_class is None:
        return ValidationResult(
            valid=False,
            errors=[
                f"Unknown tool '{tool_name}'. "
                f"Registered tools: {sorted(_TOOL_SCHEMA_MAP.keys())}."
            ],
            details={"tool_name": tool_name, "registered": sorted(_TOOL_SCHEMA_MAP.keys())},
        )

    # ── Pydantic validation ───────────────────────────────────────────────
    try:
        schema_class.model_validate(arguments)
        return ValidationResult(valid=True)

    except Exception as exc:          # pydantic.ValidationError
        # Flatten Pydantic's structured ValidationError into a list of
        # human-readable strings (for display/logging) and a field → message
        # map (for the Audit Logger). Non-Pydantic exceptions are also
        # captured defensively and reported under a generic "error" key.
        errors: list[str] = []
        details: dict[str, Any] = {}

        if hasattr(exc, "errors"):
            for err in exc.errors():
                loc   = " → ".join(str(p) for p in err["loc"]) if err["loc"] else "root"
                msg   = err["msg"]
                human = f"[{loc}] {msg}"
                errors.append(human)
                details[loc] = msg
        else:
            errors.append(str(exc))
            details["error"] = str(exc)

        return ValidationResult(valid=False, errors=errors, details=details)



# Convenience helper used by tests and the policy engine


def get_schema_for_tool(tool_name: str) -> Optional[type[BaseModel]]:
    """
    Return the Pydantic model class registered for a tool name.

    Parameters
    ----------
    tool_name : str
        Name of the tool to look up in _TOOL_SCHEMA_MAP.

    Returns
    -------
    Optional[type[BaseModel]]
        The registered argument schema class, or None if tool_name is not
        registered.
    """
    return _TOOL_SCHEMA_MAP.get(tool_name)