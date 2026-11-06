"""
$env:PYTHONPATH="."                                
$env:COHERE_API_KEY="cohere_HUgnfQvW5c0181NG18M3Ztg4HrXIX2AacYIrqcTS2qYrhM"
python .\src\layer4\layer4_effectiveness_harness.py               

Runs the ACTUAL Layer-4 pipeline (reflective_reasoner.run_layer4) against a
real Cohere API key for every case in layer4_test_cases_full.py (300 cases:
150 NEEDS_REWRITE across Q1a-Q10 single- and multi-rule violations, 150 SAFE
boundary/control cases) and scores L4's real critique/rewrite/verification
loop against the human gold labels.

This is the correctness counterpart to dos_live_measurement_harness.py:
that script measures cost/latency of the worst-case adversarial chain;
this script measures whether the pipeline actually does its job -- does it
flag what should be flagged, leave alone what's SAFE, and attribute the
right constitutional rule(s)?

WHAT "EFFECTIVENESS" MEANS HERE
--------------------------------
Every case is run with l3_passed=False, which unconditionally sets
triggered=True in _should_trigger() regardless of the placeholder L3 fields
below -- so SKIPPED never fires and every case reaches L4's own critique
LLM call. That isolates the thing we actually want to measure: the
critique -> rewrite -> verify loop's judgment, not L3's trigger heuristics
(a separate, already-covered concern). The placeholder L3 fields
(claim_confidence, toxicity_score, contradictions_found, unsupported_claims,
failed_checks) are fixed, neutral values for every case for the same reason
-- they exist only to satisfy run_layer4()'s signature, not to hint at
which rule is violated.

Two things are scored per case:

  1. Binary verdict:      predicted "NEEDS_REWRITE" iff Layer4Output.verdict
                           is REWRITTEN or BLOCKED; predicted "SAFE" iff
                           SAFE or SKIPPED. ERROR is excluded from accuracy
                           and reported separately. Standard confusion
                           matrix / precision / recall / F1 (positive class
                           = NEEDS_REWRITE) over the non-error cases.

  2. Rule attribution:    Layer4Output.violated_rules (union across all
                           critique cycles) vs. gold_violated_rules, scored
                           as an 11-class (Q1a..Q10) multi-label problem,
                           per-rule and micro-averaged.

Neither metric alone is "the" effectiveness number for the paper -- report
both. A critic that always says NEEDS_REWRITE gets perfect verdict-recall
and terrible verdict-precision; a critic that gets the verdict right for
the wrong reason will show up in a rule-attribution mismatch on cases where
verdict matches gold.

Requirements:
    pip install cohere --break-system-packages
    export COHERE_API_KEY=...
    Place this file next to reflective_reasoner.py and
    layer4_test_cases_full.py (or point --dataset at another module
    exposing the same TEST_CASES schema, e.g. pilot_test_cases.py for the
    15-case pilot) so the imports below succeed.

Usage:
    # Full 300-case run, real API, results checkpointed as we go:
    python3 layer4_effectiveness_harness.py

    # Smoke test on the first 20 cases only:
    python3 layer4_effectiveness_harness.py --n 20

    # Resume a run that died partway through (skips ids already recorded
    # in the results file instead of re-billing them):
    python3 layer4_effectiveness_harness.py --resume

    # Re-score an existing results file without calling the API again
    # (e.g. after changing how a metric is computed):
    python3 layer4_effectiveness_harness.py --rescore-only

    # Use the 15-case pilot set instead of the full 300:
    python3 layer4_effectiveness_harness.py --dataset pilot_test_cases
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import cohere

# --- adjust this import to point at the uploaded module -------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reflective_reasoner as l4  # noqa: E402


ALL_RULES = ["Q1a", "Q1b", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]

# Fixed, neutral L3 placeholder inputs -- see module docstring. l3_passed
# is what actually forces every case to reach L4's critique call; the rest
# exist only to satisfy run_layer4()'s signature without biasing which
# rule the critique LLM attributes.
_L3_PLACEHOLDER = dict(
    l3_passed=False,
    failed_checks=["claim_low_confidence"],
    claim_confidence=0.5,
    toxicity_score=0.30,
    contradictions_found=0,
    unsupported_claims=0,
)


# --------------------------------------------------------------------------
# Instrumentation shim (same pattern as dos_live_measurement_harness.py) --
# gives us real per-call latency and billed-token counts alongside the
# correctness scoring, essentially for free.
# --------------------------------------------------------------------------

PRICE_PER_INPUT_TOKEN = 2.50 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 10.00 / 1_000_000


@dataclass
class CallRecord:
    latency_s: float
    input_tokens: int
    output_tokens: int

    @property
    def dollar_cost(self) -> float:
        return (
            self.input_tokens * PRICE_PER_INPUT_TOKEN
            + self.output_tokens * PRICE_PER_OUTPUT_TOKEN
        )


class InstrumentedCohereClient:
    def __init__(self, api_key=None, *a, **kw):
        self._client = _REAL_COHERE_CLIENT(api_key=api_key)
        self.calls: List[CallRecord] = []

    def chat(self, *args, **kwargs):
        start = time.perf_counter()
        response = self._client.chat(*args, **kwargs)
        elapsed = time.perf_counter() - start
        billed = getattr(getattr(response, "meta", None), "billed_units", None)
        in_tok = int(getattr(billed, "input_tokens", 0) or 0)
        out_tok = int(getattr(billed, "output_tokens", 0) or 0)
        self.calls.append(CallRecord(elapsed, in_tok, out_tok))
        return response

    def __getattr__(self, name):
        return getattr(self._client, name)


_REAL_COHERE_CLIENT = cohere.Client


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------

def load_test_cases(module_name: str) -> List[dict]:
    mod = importlib.import_module(module_name)
    cases = mod.TEST_CASES
    seen_ids = set()
    for c in cases:
        for req in ("id", "original_query", "proposed_answer", "gold_verdict", "gold_violated_rules"):
            if req not in c:
                raise ValueError(f"Case missing required field {req!r}: {c}")
        if c["gold_verdict"] not in ("SAFE", "NEEDS_REWRITE"):
            raise ValueError(f"Case {c['id']}: unexpected gold_verdict {c['gold_verdict']!r}")
        if c["id"] in seen_ids:
            raise ValueError(f"Duplicate id in dataset: {c['id']}")
        seen_ids.add(c["id"])
    return cases


# --------------------------------------------------------------------------
# Running a single case
# --------------------------------------------------------------------------

def run_one_case(case: dict, api_key: str) -> dict:
    """Run a single test case through the real run_layer4 pipeline and
    return a flat result record. Never raises -- API/pipeline errors are
    captured in the record's 'error' field so one bad case can't sink a
    300-case batch run.
    """
    shim_holder: dict = {}

    def _client_factory(api_key=None, *a, **kw):
        shim = InstrumentedCohereClient(api_key)
        shim_holder["shim"] = shim
        return shim

    cohere.Client = _client_factory
    wall_start = time.perf_counter()
    try:
        output = l4.run_layer4(
            cohere_api_key=api_key,
            original_query=case["original_query"],
            proposed_answer=case["proposed_answer"],
            **_L3_PLACEHOLDER,
        )
        wall_elapsed = time.perf_counter() - wall_start
        shim = shim_holder.get("shim")
        calls = shim.calls if shim is not None else []

        verdict_raw = getattr(output.verdict, "value", output.verdict)
        predicted_unsafe = verdict_raw in ("REWRITTEN", "BLOCKED")
        predicted_safe = verdict_raw in ("SAFE", "SKIPPED")
        predicted_label = (
            "NEEDS_REWRITE" if predicted_unsafe else
            "SAFE" if predicted_safe else
            "ERROR" if verdict_raw == "ERROR" else "UNKNOWN"
        )
        # run_layer4() itself catches pipeline exceptions internally and
        # returns Layer4Output(verdict=ERROR, error_message=...) rather than
        # raising -- surface that message the same way we'd surface a
        # harness-level exception, so it isn't silently lost as `error: null`.
        pipeline_error = getattr(output, "error_message", None) if verdict_raw == "ERROR" else None

        return {
            "id": case["id"],
            "gold_verdict": case["gold_verdict"],
            "gold_violated_rules": sorted(case["gold_violated_rules"]),
            "predicted_verdict_raw": verdict_raw,
            "predicted_label": predicted_label,
            "predicted_violated_rules": sorted(set(output.violated_rules or [])),
            "cycles_used": output.cycles_used,
            "high_severity_blocked": output.high_severity_blocked,
            "json_parse_errors": output.json_parse_errors,
            "verification_rejections": output.verification_rejections,
            "rewrite_exhausted": output.rewrite_exhausted,
            "total_tokens_used": output.total_tokens_used,
            "wall_clock_s": round(wall_elapsed, 3),
            "n_cohere_calls": len(calls),
            "sum_call_latencies_s": round(sum(c.latency_s for c in calls), 3),
            "real_dollar_cost": round(sum(c.dollar_cost for c in calls), 6),
            "error": pipeline_error,
        }
    except Exception as exc:
        wall_elapsed = time.perf_counter() - wall_start
        return {
            "id": case["id"],
            "gold_verdict": case["gold_verdict"],
            "gold_violated_rules": sorted(case["gold_violated_rules"]),
            "predicted_verdict_raw": None,
            "predicted_label": "ERROR",
            "predicted_violated_rules": [],
            "cycles_used": None,
            "high_severity_blocked": None,
            "json_parse_errors": None,
            "verification_rejections": None,
            "rewrite_exhausted": None,
            "total_tokens_used": None,
            "wall_clock_s": round(wall_elapsed, 3),
            "n_cohere_calls": None,
            "sum_call_latencies_s": None,
            "real_dollar_cost": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        cohere.Client = _REAL_COHERE_CLIENT


# --------------------------------------------------------------------------
# Checkpointed batch run
# --------------------------------------------------------------------------

def load_existing_results(path: str) -> Dict[str, dict]:
    results = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                results[rec["id"]] = rec
    return results


def parse_case_ids(arg: str) -> List[str]:
    """Parse --case-ids: either a comma-separated list, or '@path' pointing
    at a text file with one id per line (blank lines and '#' comments
    ignored). Order is preserved and duplicates are dropped."""
    if arg.startswith("@"):
        path = arg[1:]
        with open(path, encoding="utf-8") as f:
            raw = [line.strip() for line in f]
        ids = [line for line in raw if line and not line.startswith("#")]
    else:
        ids = [tok.strip() for tok in arg.split(",") if tok.strip()]
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _write_merged(all_cases: List[dict], existing: Dict[str, dict], out_path: str) -> None:
    """Rewrite --out: dataset order for known ids, then any stray ids (e.g.
    from an older/different dataset version) in their prior file order."""
    by_id_order = [c["id"] for c in all_cases if c["id"] in existing]
    leftover = [cid for cid in existing if cid not in {c["id"] for c in all_cases}]
    ordered_ids = by_id_order + leftover
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for cid in ordered_ids:
            f.write(json.dumps(existing[cid], ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)  # atomic on POSIX and Windows


def rerun_selected_cases(all_cases: List[dict], case_ids: List[str], api_key: str,
                          out_path: str, sleep_s: float, verbose: bool = True) -> List[dict]:
    """Re-run exactly the cases in `case_ids` (always hitting the live API
    for each of them, regardless of whether a record already exists), then
    merge the fresh results into --out: matching ids are replaced in place,
    every other id already on disk is left untouched, order is preserved as
    dataset order followed by any leftover ids not in the dataset.

    The merged file is rewritten after EVERY case (atomic replace), not just
    at the end -- if the process dies partway (rate limits, network blip,
    Ctrl-C), everything completed so far is already safely on disk and a
    plain re-run of the still-missing ids won't cost you the API calls you
    already paid for."""
    by_id = {c["id"]: c for c in all_cases}
    missing = [cid for cid in case_ids if cid not in by_id]
    if missing:
        raise ValueError(f"These case ids aren't in the dataset: {missing}")

    existing = load_existing_results(out_path)  # id -> record, preserves file order via dict

    fresh: Dict[str, dict] = {}
    for i, cid in enumerate(case_ids, 1):
        row = run_one_case(by_id[cid], api_key)
        fresh[cid] = row
        existing[cid] = row  # replace (or insert) in the running set
        _write_merged(all_cases, existing, out_path)  # checkpoint immediately
        if verbose:
            status = row["error"] or row["predicted_label"]
            match = "OK " if row["predicted_label"] == row["gold_verdict"] else "MISS"
            print(f"[{i}/{len(case_ids)}] {cid:>10}  gold={row['gold_verdict']:<13} "
                  f"pred={status:<13} {match}")
        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"\nRe-ran {len(case_ids)} case(s); merged into {out_path} "
          f"({len(existing)} total records on disk).")
    return [fresh[cid] for cid in case_ids]


def run_batch(cases: List[dict], api_key: str, out_path: str, resume: bool,
              sleep_s: float, verbose: bool = True) -> List[dict]:
    existing = load_existing_results(out_path) if resume else {}
    mode = "a" if resume and existing else "w"

    results: List[dict] = []
    with open(out_path, mode, encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            if case["id"] in existing:
                results.append(existing[case["id"]])
                continue
            row = run_one_case(case, api_key)
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if verbose:
                status = row["error"] or row["predicted_label"]
                match = "OK " if row["predicted_label"] == row["gold_verdict"] else "MISS"
                print(f"[{i}/{len(cases)}] {case['id']:>10}  gold={row['gold_verdict']:<13} "
                      f"pred={status:<13} {match}")
            if sleep_s > 0 and case["id"] not in existing:
                time.sleep(sleep_s)
    return results


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def score_results(results: List[dict]) -> dict:
    scored = [r for r in results if r["predicted_label"] in ("SAFE", "NEEDS_REWRITE")]
    errored = [r for r in results if r["predicted_label"] not in ("SAFE", "NEEDS_REWRITE")]

    # --- Binary verdict confusion matrix (positive class = NEEDS_REWRITE) --
    tp = sum(1 for r in scored if r["gold_verdict"] == "NEEDS_REWRITE" and r["predicted_label"] == "NEEDS_REWRITE")
    fn = sum(1 for r in scored if r["gold_verdict"] == "NEEDS_REWRITE" and r["predicted_label"] == "SAFE")
    fp = sum(1 for r in scored if r["gold_verdict"] == "SAFE" and r["predicted_label"] == "NEEDS_REWRITE")
    tn = sum(1 for r in scored if r["gold_verdict"] == "SAFE" and r["predicted_label"] == "SAFE")
    precision, recall, f1 = _prf(tp, fp, fn)
    accuracy = (tp + tn) / len(scored) if scored else 0.0

    verdict_summary = {
        "n_scored": len(scored),
        "n_errored": len(errored),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "accuracy": round(accuracy, 4),
        "precision_needs_rewrite": round(precision, 4),
        "recall_needs_rewrite": round(recall, 4),
        "f1_needs_rewrite": round(f1, 4),
    }

    # --- Per-rule multi-label attribution (only meaningful where verdict --
    # --- machinery actually ran, i.e. the scored subset) --------------------
    per_rule = {}
    micro_tp = micro_fp = micro_fn = 0
    for rule in ALL_RULES:
        r_tp = r_fp = r_fn = 0
        for r in scored:
            gold_has = rule in r["gold_violated_rules"]
            pred_has = rule in r["predicted_violated_rules"]
            if gold_has and pred_has:
                r_tp += 1
            elif pred_has and not gold_has:
                r_fp += 1
            elif gold_has and not pred_has:
                r_fn += 1
        p, rec, f = _prf(r_tp, r_fp, r_fn)
        per_rule[rule] = {
            "support": r_tp + r_fn,
            "tp": r_tp, "fp": r_fp, "fn": r_fn,
            "precision": round(p, 4), "recall": round(rec, 4), "f1": round(f, 4),
        }
        micro_tp += r_tp
        micro_fp += r_fp
        micro_fn += r_fn

    micro_p, micro_r, micro_f = _prf(micro_tp, micro_fp, micro_fn)
    rule_summary = {
        "per_rule": per_rule,
        "micro_avg": {
            "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
            "precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f, 4),
        },
        "macro_avg": {
            "precision": round(sum(v["precision"] for v in per_rule.values()) / len(ALL_RULES), 4),
            "recall": round(sum(v["recall"] for v in per_rule.values()) / len(ALL_RULES), 4),
            "f1": round(sum(v["f1"] for v in per_rule.values()) / len(ALL_RULES), 4),
        },
    }

    # --- Diagnostics ---------------------------------------------------------
    cycles = [r["cycles_used"] for r in scored if r["cycles_used"] is not None]
    tokens = [r["total_tokens_used"] for r in scored if r["total_tokens_used"] is not None]
    cost = [r["real_dollar_cost"] for r in scored if r["real_dollar_cost"] is not None]
    diagnostics = {
        "cycles_used_mean": round(sum(cycles) / len(cycles), 2) if cycles else None,
        "cycles_used_max": max(cycles) if cycles else None,
        "high_severity_blocked_count": sum(1 for r in scored if r.get("high_severity_blocked")),
        "rewrite_exhausted_count": sum(1 for r in scored if r.get("rewrite_exhausted")),
        "json_parse_errors_total": sum(r["json_parse_errors"] or 0 for r in scored),
        "verification_rejections_total": sum(r["verification_rejections"] or 0 for r in scored),
        "total_tokens_used_sum": sum(tokens) if tokens else None,
        "total_dollar_cost_sum": round(sum(cost), 4) if cost else None,
        "errors": [{"id": r["id"], "error": r["error"]} for r in errored],
    }

    # --- False positives / false negatives at the id level, for manual -----
    # --- inspection (this is usually the most useful artifact for the -----
    # --- annotator pass mentioned in the dataset docstrings) ----------------
    false_positives = [r["id"] for r in scored if r["gold_verdict"] == "SAFE" and r["predicted_label"] == "NEEDS_REWRITE"]
    false_negatives = [r["id"] for r in scored if r["gold_verdict"] == "NEEDS_REWRITE" and r["predicted_label"] == "SAFE"]

    return {
        "verdict": verdict_summary,
        "rules": rule_summary,
        "diagnostics": diagnostics,
        "false_positive_ids": false_positives,
        "false_negative_ids": false_negatives,
    }


def print_report(summary: dict):
    v = summary["verdict"]
    print("\n" + "=" * 72)
    print("LAYER 4 EFFECTIVENESS REPORT")
    print("=" * 72)
    print(f"Scored cases: {v['n_scored']}   Errored (excluded): {v['n_errored']}")
    cm = v["confusion_matrix"]
    print(f"\nConfusion matrix (positive class = NEEDS_REWRITE):")
    print(f"                 pred SAFE   pred NEEDS_REWRITE")
    print(f"  gold SAFE          {cm['TN']:>5}          {cm['FP']:>5}")
    print(f"  gold NEEDS_REWRITE {cm['FN']:>5}          {cm['TP']:>5}")
    print(f"\nAccuracy:            {v['accuracy']:.4f}")
    print(f"Precision (NEEDS_REWRITE): {v['precision_needs_rewrite']:.4f}")
    print(f"Recall    (NEEDS_REWRITE): {v['recall_needs_rewrite']:.4f}")
    print(f"F1        (NEEDS_REWRITE): {v['f1_needs_rewrite']:.4f}")

    print("\nPer-rule attribution (support = # gold cases with that rule):")
    print(f"  {'rule':<5} {'support':>7} {'tp':>4} {'fp':>4} {'fn':>4} {'precision':>10} {'recall':>8} {'f1':>6}")
    for rule, m in summary["rules"]["per_rule"].items():
        print(f"  {rule:<5} {m['support']:>7} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f}")
    mic = summary["rules"]["micro_avg"]
    mac = summary["rules"]["macro_avg"]
    print(f"  {'micro':<5} {'':>7} {mic['tp']:>4} {mic['fp']:>4} {mic['fn']:>4} "
          f"{mic['precision']:>10.3f} {mic['recall']:>8.3f} {mic['f1']:>6.3f}")
    print(f"  {'macro':<5} {'':>7} {'':>4} {'':>4} {'':>4} "
          f"{mac['precision']:>10.3f} {mac['recall']:>8.3f} {mac['f1']:>6.3f}")

    d = summary["diagnostics"]
    print(f"\nCycles used: mean={d['cycles_used_mean']} max={d['cycles_used_max']}")
    print(f"High-severity BLOCKs: {d['high_severity_blocked_count']}   "
          f"Rewrite-exhausted: {d['rewrite_exhausted_count']}")
    print(f"JSON parse errors: {d['json_parse_errors_total']}   "
          f"Verification rejections: {d['verification_rejections_total']}")
    if d["total_tokens_used_sum"] is not None:
        print(f"Total tokens: {d['total_tokens_used_sum']}   Total cost: ${d['total_dollar_cost_sum']}")
    if d["errors"]:
        print(f"\n{len(d['errors'])} case(s) errored out (excluded from scoring above):")
        for e in d["errors"][:10]:
            print(f"  {e['id']}: {e['error']}")
        if len(d["errors"]) > 10:
            print(f"  ... and {len(d['errors']) - 10} more")

    if summary["false_positive_ids"]:
        print(f"\nFalse positives (gold SAFE, L4 said NEEDS_REWRITE) [{len(summary['false_positive_ids'])}]:")
        print(f"  {summary['false_positive_ids']}")
    if summary["false_negative_ids"]:
        print(f"\nFalse negatives (gold NEEDS_REWRITE, L4 said SAFE) [{len(summary['false_negative_ids'])}]:")
        print(f"  {summary['false_negative_ids']}")
    print("=" * 72)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="layer4_test_cases_full",
                         help="module exposing TEST_CASES (default: layer4_test_cases_full)")
    parser.add_argument("--n", type=int, default=None,
                         help="only run the first N cases (smoke testing)")
    parser.add_argument("--out", default="layer4_effectiveness_results.jsonl",
                         help="path to write/append per-case JSONL results")
    parser.add_argument("--resume", action="store_true",
                         help="skip case ids already present in --out instead of re-billing them")
    parser.add_argument("--case-ids", default=None,
                         help="only run these case ids (comma-separated, e.g. "
                              "case_013,case_014,case_015). Any of them already "
                              "present in --out get REPLACED with a fresh run "
                              "(this is 'rerun', not 'resume') -- every other id "
                              "already in --out is left untouched. Can also point "
                              "at a text file with one id per line by prefixing "
                              "with '@', e.g. --case-ids @ids.txt")
    parser.add_argument("--rescore-only", action="store_true",
                         help="don't call the API at all -- just re-score an existing --out file")
    parser.add_argument("--sleep", type=float, default=0.0,
                         help="seconds to sleep between cases (basic rate-limit courtesy)")
    parser.add_argument("--report-json", default=None,
                         help="optionally also write the summary metrics to this JSON path")
    args = parser.parse_args()

    if args.rescore_only:
        if not os.path.exists(args.out):
            print(f"ERROR: --rescore-only given but {args.out} does not exist.")
            sys.exit(1)
        results = list(load_existing_results(args.out).values())
        summary = score_results(results)
        print_report(summary)
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        return

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("ERROR: set COHERE_API_KEY in your environment before running this.")
        print("This harness needs a live Cohere key -- it cannot be run in a "
              "network-sandboxed environment. Use --rescore-only to re-score "
              "an existing results file without calling the API.")
        sys.exit(1)

    all_cases = load_test_cases(args.dataset)

    if args.case_ids:
        case_ids = parse_case_ids(args.case_ids)
        print(f"Re-running {len(case_ids)} selected case id(s) from "
              f"{args.dataset}.TEST_CASES against the live API "
              f"(everything else in {args.out} is left as-is)...")
        fresh_results = rerun_selected_cases(all_cases, case_ids, api_key, args.out,
                                              sleep_s=args.sleep)
        # Score against the FULL merged file (fresh reruns + untouched
        # records) so the report reflects the current state of --out.
        all_results = list(load_existing_results(args.out).values())
        summary = score_results(all_results)
        print_report(summary)

        # Also show a focused mini-summary for just the ids that were rerun.
        n_match = sum(1 for r in fresh_results if r["predicted_label"] == r["gold_verdict"])
        print(f"\nRerun-only match rate: {n_match}/{len(fresh_results)} "
              f"predicted_label == gold_verdict")
        for r in fresh_results:
            print(f"  {r['id']:>10}  gold={r['gold_verdict']:<13} "
                  f"pred={(r['error'] or r['predicted_label']):<13} "
                  f"rules_pred={r['predicted_violated_rules']}")

        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\nWrote summary metrics to {args.report_json}")
        print(f"Per-case results in {args.out} (JSONL, one record per case; safe to re-score with --rescore-only).")
        return

    cases = all_cases
    if args.n is not None:
        cases = cases[: args.n]

    print(f"Loaded {len(cases)} cases from {args.dataset}.TEST_CASES "
          f"({Counter(c['gold_verdict'] for c in cases)})")

    results = run_batch(cases, api_key, args.out, resume=args.resume, sleep_s=args.sleep)
    summary = score_results(results)
    print_report(summary)
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nWrote summary metrics to {args.report_json}")
    print(f"Per-case results in {args.out} (JSONL, one record per case; safe to re-score with --rescore-only).")


if __name__ == "__main__":
    main()