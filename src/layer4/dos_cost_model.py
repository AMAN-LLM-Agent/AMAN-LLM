"""
dos_cost_model.py

Analytical (non-network) cost/call-count model for the Layer-4 DoS example
in Section 7.3. Every constant below is measured directly from
`reflective_reasoner.py` (char counts of the actual template strings /
_MAX_CHARS / max_tokens_* defaults / max_cycles / max_high_sev_retries) --
nothing here is invented. Pricing is Cohere's own published rate for
command-r-plus-08-2024 (cohere.com/pricing, confirmed current as of this
writing: $2.50 / 1M input tokens, $10.00 / 1M output tokens).

This script answers: "what is the worst-case number of Cohere calls, tokens,
and dollars a single Layer-4 invocation can be forced to consume, using only
mechanisms that already exist in the control flow?" It does NOT tell you the
wall-clock latency of a real call -- for that you need
`dos_live_measurement_harness.py` run against a live Cohere key, or the
published third-party throughput benchmarks cited in the .md writeup.

Run: python3 dos_cost_model.py
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants pulled directly from reflective_reasoner.py
# ---------------------------------------------------------------------------
MAX_CHARS = 6_000            # _MAX_CHARS -- ceiling _safe_truncate() enforces
                              # on original_query AND proposed_answer in every
                              # one of the four prompt builders.
CHARS_PER_TOKEN = 4.0        # standard English/Arabic BPE approximation used
                              # throughout Cohere's own pricing calculators.

# Static (non-attacker-controlled) scaffolding text length of each prompt
# template, measured with len() on the literal strings between the {} slots
# in _CRITIQUE_PROMPT_TEMPLATE / _REWRITE_PROMPT_TEMPLATE /
# _VERIFICATION_PROMPT_TEMPLATE / _CRITIQUE_REWRITE_COMBINED_TEMPLATE.
TEMPLATE_OVERHEAD_CHARS = {
    "critique":     1750,
    "rewrite":      1250,
    "verification":  420,
    "combined":     1150,
}

# CONSTITUTIONAL_RULES_COMPACT_AR -- the rules block actually injected into
# every critique/combined call in the normal (failed_checks non-empty) path.
# The uncompacted 2,768-char CONSTITUTIONAL_RULES_AR is only used as a
# fallback when failed_checks is empty, which is not the typical L4 trigger
# path and is therefore excluded from this worst-case model.
COMPACT_RULES_CHARS = 691

# max_tokens_* defaults from Layer4Config -- hard ceilings on Cohere's
# billed output per call type.
OUT_TOKEN_CAP = {
    "critique": 512,
    "rewrite": 384,
    "verification": 192,
    "combined": 768,
}

# Cohere command-r-plus-08-2024 pricing, cohere.com/pricing (checked 2026-07).
PRICE_PER_INPUT_TOKEN = 2.50 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 10.00 / 1_000_000


@dataclass
class CallCost:
    label: str
    kind: str
    input_chars: int
    input_tokens: float
    output_tokens_cap: int

    @property
    def dollar_cost(self) -> float:
        return (
            self.input_tokens * PRICE_PER_INPUT_TOKEN
            + self.output_tokens_cap * PRICE_PER_OUTPUT_TOKEN
        )


def worst_case_call_chain(query_chars: int = MAX_CHARS,
                           answer_chars: int = MAX_CHARS):
    """
    Reconstructs the maximum number of Cohere calls a single Layer-4
    invocation can be forced into WITHOUT relying on network retries
    (i.e. assuming every call succeeds on the first try).

    Two paths exist in _run_critique_loop / _high_severity_rewrite_loop:

      (a) High-severity path: 1 critique call, then up to
          max_high_sev_retries=2 attempts of (rewrite + verification),
          then the function returns unconditionally (accept or BLOCK).
          Max calls = 1 + 2*(1+1) = 5.

      (b) Medium-severity, non-converging path: runs the full
          max_cycles=3 loop. Cycle 1 = critique + rewrite + async
          verification (3 calls). Cycles 2-3 each = 1 combined call +
          1 verification call (2 calls each). The final verification is
          drained after the loop. Max calls = 3 + 2 + 2 = 7.

    (b) is strictly worse for an attacker, because every call in the
    7-call chain re-embeds the full original_query and proposed_answer,
    whereas the 5-call high-severity path is capped by construction.
    This function returns path (b), the actual worst case.
    """
    chain = [
        ("critique",     "cycle 1 - critique-only call"),
        ("rewrite",      "cycle 1 - rewrite call"),
        ("verification", "cycle 1 - verification (queued async)"),
        ("combined",     "cycle 2 - combined critique+rewrite call"),
        ("verification", "cycle 2 - verification (queued async)"),
        ("combined",     "cycle 3 - combined critique+rewrite call"),
        ("verification", "cycle 3 - verification (drained at loop exit)"),
    ]

    calls = []
    for kind, label in chain:
        rules_chars = COMPACT_RULES_CHARS if kind in ("critique", "combined") else 0
        input_chars = TEMPLATE_OVERHEAD_CHARS[kind] + query_chars + answer_chars + rules_chars
        calls.append(CallCost(
            label=label,
            kind=kind,
            input_chars=input_chars,
            input_tokens=input_chars / CHARS_PER_TOKEN,
            output_tokens_cap=OUT_TOKEN_CAP[kind],
        ))
    return calls


def main():
    calls = worst_case_call_chain()

    print(f"{'call':45s} {'in_chars':>9s} {'in_tok~':>8s} {'out_cap':>8s} {'cost($)':>10s}")
    total_cost = 0.0
    total_in_tok = 0.0
    total_out_tok = 0
    for c in calls:
        total_cost += c.dollar_cost
        total_in_tok += c.input_tokens
        total_out_tok += c.output_tokens_cap
        print(f"{c.label:45s} {c.input_chars:9d} {c.input_tokens:8.0f} "
              f"{c.output_tokens_cap:8d} {c.dollar_cost:10.5f}")

    print("-" * 84)
    print(f"{'TOTAL':45s} {'':9s} {total_in_tok:8.0f} {total_out_tok:8d} {total_cost:10.5f}")
    print()
    print(f"Cohere calls fired by ONE malicious request: {len(calls)}")
    print(f"Cost to the defender per malicious request:   ${total_cost:.5f}")
    print()

    for n, label in [(500, "500 concurrent copies"), (10_000, "10,000 copies / hour"),
                      (360_000, "100 req/s sustained for 1 hour")]:
        print(f"  {label:35s} -> ${total_cost * n:,.2f}")

    print()
    print("Attacker-side cost per request: one HTTP POST carrying a "
          f"~{MAX_CHARS/1000:.0f} KB text body -- bandwidth and compute both "
          "negligible (<1 ms of attacker CPU time, no auth/compute required "
          "beyond reaching the public-facing endpoint in front of Layer 4).")


if __name__ == "__main__":
    main()
