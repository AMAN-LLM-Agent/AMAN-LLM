from src.layer3.claim_verifier import run_claim_verification

r = run_claim_verification(
    "الدينار الجزائري ينقسم إلى مائة سنتيم.",
    "كم سنتيم في الدينار الجزائري؟",
    evidence_passages=None,
)

print("Passed:", r.passed)

if r.claims:
    print("Evidence:", r.claims[0].evidence_snippet)
else:
    print("No claims found.")