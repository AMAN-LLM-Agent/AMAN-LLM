"""
run_layer2_effectiveness.py
============================
Executes every Case in BENCHMARK_CASES / SAFE_BENCHMARK_CASES
(test_layer_2_final.py) against the real Layer 2 pipeline
(schema_validator -> injection_detector -> policy_engine YAML rules)
via policy_engine.run_layer2(), and produces a full effectiveness report:

  - overall accuracy / precision / recall / F1
  - per-category breakdown (grouped by expected_rule / case-name prefix)
  - which gate caught each true positive (schema / injection / rule)
  - every mismatch, with expected vs. actual decision + rule
  - latency stats

Usage:
    python3 run_layer2_effectiveness.py [--verbose] [--json report.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "layer2"))

import policy_engine as pe
from tests.test_layer_2_final import BENCHMARK_CASES, SAFE_BENCHMARK_CASES, Case

# Cases in the suite don't set session_role except for the PR-05 (role
# violation) category, which deliberately uses a restricted role to trigger
# the rule under test. Every other case needs a role with access to every
# tool ("admin") so that the rule actually being exercised gets the chance
# to fire, instead of being masked by an unrelated PR-05 role block.
DEFAULT_ROLE = "admin"

_counter = itertools.count()


def case_to_benchmark(case: Case) -> pe.BenchmarkCase:
    ctx_id = f"ctx-{case.name}-{next(_counter)}"
    call = pe.PlannedToolCall(
        tool_name=case.tool,
        arguments=case.params,
        context_id=ctx_id,
        session_role=case.session_role or DEFAULT_ROLE,
    )
    return pe.BenchmarkCase(
        label=case.name,
        call=call,
        expected_block=case.expected_block,
        expected_rule=case.expected_rule,
    )


def category_of(name: str) -> str:
    """Derive a grouping key from the case name, e.g. 'pr07_en_001' -> 'PR-07'."""
    parts = name.split("_")
    prefix = parts[0]
    if prefix.startswith("pr") and prefix[2:].isdigit():
        return f"PR-{prefix[2:].zfill(2)}"
    return "SAFE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print every case result")
    ap.add_argument("--json", metavar="PATH", help="also write a machine-readable JSON report")
    args = ap.parse_args()

    all_cases = list(BENCHMARK_CASES) + list(SAFE_BENCHMARK_CASES)
    names = [c.name for c in all_cases]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        print(f"WARNING: duplicate case names found: {sorted(dupes)}")

    bench_cases = [case_to_benchmark(c) for c in all_cases]

    print(f"Loaded {len(BENCHMARK_CASES)} must-block cases and "
          f"{len(SAFE_BENCHMARK_CASES)} must-allow cases "
          f"({len(all_cases)} total). Running against live Layer 2 pipeline...\n")

    report = pe.run_benchmark(bench_cases)

    # ------------------------------------------------------------------
    # Re-run individually (cheap; already-warm regex/pydantic) to capture
    # per-case gate attribution + full decision for the detailed report,
    # since run_benchmark() only returns aggregate + wrong[] labels.
    # ------------------------------------------------------------------
    results = []
    for case, bc in zip(all_cases, bench_cases):
        pe.reset_rate_bucket(bc.call.context_id)
        d = pe.run_layer2(bc.call)
        actual_block = d.decision == "BLOCK"
        correct_decision = actual_block == case.expected_block
        correct_rule = (case.expected_rule is None) or (d.rule_id == case.expected_rule)
        results.append({
            "name": case.name,
            "category": category_of(case.name),
            "tool": case.tool,
            "session_role": bc.call.session_role,
            "expected_block": case.expected_block,
            "expected_rule": case.expected_rule,
            "actual_decision": d.decision,
            "actual_rule": d.rule_id,
            "actual_gate": d.gate,
            "violation_type": d.violation_type,
            "correct_decision": correct_decision,
            "correct_rule": correct_rule,
            "pass": correct_decision and correct_rule,
            "latency_ms": d.latency_ms,
        })

    # ------------------------------------------------------------------
    # Aggregate per-category
    # ------------------------------------------------------------------
    by_cat = defaultdict(lambda: {"total": 0, "pass": 0, "fail": []})
    for r in results:
        c = by_cat[r["category"]]
        c["total"] += 1
        if r["pass"]:
            c["pass"] += 1
        else:
            c["fail"].append(r)

    # Which gate caught each blocked case (for must-block cases only)
    gate_counts = defaultdict(int)
    for r in results:
        if r["expected_block"] and r["actual_decision"] == "BLOCK":
            gate_counts[r["actual_gate"]] += 1

    total = len(results)
    passed = sum(r["pass"] for r in results)
    failed = total - passed

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print("=" * 78)
    print("LAYER 2 EFFECTIVENESS REPORT")
    print("=" * 78)
    print(f"Total cases:         {total}")
    print(f"Passed:              {passed}  ({passed/total:.1%})")
    print(f"Failed:              {failed}  ({failed/total:.1%})")
    print()
    print(f"Precision:           {report.precision}")
    print(f"Recall:              {report.recall}")
    print(f"F1:                  {report.f1}")
    print(f"Avg latency (ms):    {report.avg_latency_ms}")
    print(f"P95 latency (ms):    {report.p95_latency_ms}")
    print()
    print("Blocks caught by gate (true-positive must-block cases only):")
    for gate, n in sorted(gate_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {gate:10s} {n}")
    print()

    print("-" * 78)
    print(f"{'Category':10s} {'Pass/Total':12s} {'Rate':8s}")
    print("-" * 78)
    for cat in sorted(by_cat.keys()):
        c = by_cat[cat]
        rate = c["pass"] / c["total"] if c["total"] else 0.0
        flag = "" if c["pass"] == c["total"] else "  <-- FAILURES"
        print(f"{cat:10s} {c['pass']:>3d}/{c['total']:<8d} {rate:>6.1%}{flag}")

    if failed:
        print()
        print("=" * 78)
        print(f"FAILING CASES ({failed})")
        print("=" * 78)
        for r in results:
            if not r["pass"]:
                exp = "BLOCK" if r["expected_block"] else "ALLOW"
                print(f"[{r['name']}] tool={r['tool']} role={r['session_role']}")
                print(f"    expected: {exp}"
                      f"{' rule=' + r['expected_rule'] if r['expected_rule'] else ''}")
                print(f"    actual:   {r['actual_decision']} "
                      f"rule={r['actual_rule']} gate={r['actual_gate']} "
                      f"vtype={r['violation_type']}")
                print()
    else:
        print()
        print("All cases passed. Layer 2 pipeline matches every expected verdict.")

    if args.verbose:
        print()
        print("=" * 78)
        print("ALL CASE RESULTS")
        print("=" * 78)
        for r in results:
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"[{mark}] {r['name']:20s} {r['actual_decision']:6s} "
                  f"rule={str(r['actual_rule']):8s} "
                  f"({r['latency_ms']:.2f} ms)")

    if args.json:
        out = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "precision": report.precision,
                "recall": report.recall,
                "f1": report.f1,
                "avg_latency_ms": report.avg_latency_ms,
                "p95_latency_ms": report.p95_latency_ms,
            },
            "by_category": {
                cat: {"total": c["total"], "pass": c["pass"]}
                for cat, c in by_cat.items()
            },
            "results": results,
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nJSON report written to {args.json}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
