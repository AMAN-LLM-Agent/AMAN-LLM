#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AMAN-LLM Phase 5 — Docker Sandbox Build & Verify Script
# Usage:  bash scripts/phase5_build_and_verify.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SANDBOX_IMAGE="aman-sandbox:latest"
DOCKERFILE="docker/Dockerfile.sandbox"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
pass()  { echo -e "${GREEN}[PASS]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── Step 1: Build the image ───────────────────────────────────────────────────
info "Building sandbox image: ${SANDBOX_IMAGE}"
docker build -t "${SANDBOX_IMAGE}" -f "${DOCKERFILE}" . \
    --label "phase=5" \
    --label "project=aman-llm"
pass "Image built successfully."

# ── Step 2: Smoke test – valid tool call ──────────────────────────────────────
info "Smoke test: web_search"
SEARCH_RESULT=$(docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:size=32m,mode=1777 \
    --memory 256m \
    --security-opt no-new-privileges \
    -e 'TOOL_PAYLOAD={"tool":"web_search","args":{"query":"اختبار الحماية"}}' \
    "${SANDBOX_IMAGE}")

if echo "${SEARCH_RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['success']" 2>/dev/null; then
    pass "web_search returned success=true"
else
    fail "web_search did not return success=true. Output: ${SEARCH_RESULT}"
    exit 1
fi

# ── Step 3: code_interpreter ──────────────────────────────────────────────────
info "Smoke test: code_interpreter"
CODE_RESULT=$(docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:size=32m,mode=1777 \
    --memory 256m \
    --security-opt no-new-privileges \
    -e 'TOOL_PAYLOAD={"tool":"code_interpreter","args":{"script":"print(6*7)"}}' \
    "${SANDBOX_IMAGE}")

if echo "${CODE_RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); assert '42' in d['result']['stdout']" 2>/dev/null; then
    pass "code_interpreter executed 6*7=42 correctly"
else
    fail "code_interpreter test failed. Output: ${CODE_RESULT}"
    exit 1
fi

# ── Step 4: Isolation — no network ───────────────────────────────────────────
info "Isolation check: no network access"
NET_TEST=$(docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:size=32m,mode=1777 \
    "${SANDBOX_IMAGE}" \
    python3 -c "
import socket, sys
try:
    socket.setdefaulttimeout(3)
    s = socket.socket()
    s.connect(('8.8.8.8', 53))
    print('CONNECTED')
except Exception:
    print('BLOCKED')
" 2>/dev/null || echo "BLOCKED")

if [[ "${NET_TEST}" == *"BLOCKED"* ]]; then
    pass "Network access correctly blocked (--network none)"
else
    fail "Container reached the internet! Network isolation FAILED."
    exit 1
fi

# ── Step 5: Isolation — read-only FS ─────────────────────────────────────────
info "Isolation check: read-only filesystem"
FS_EXIT=0
docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:size=32m,mode=1777 \
    "${SANDBOX_IMAGE}" \
    python3 -c "open('/sandbox/evil.txt','w')" 2>/dev/null || FS_EXIT=$?

if [[ "${FS_EXIT}" -ne 0 ]]; then
    pass "Read-only FS enforced — write to /sandbox rejected (exit ${FS_EXIT})"
else
    fail "Container was able to write outside /tmp — FS isolation FAILED."
    exit 1
fi

# ── Step 6: Non-root user ─────────────────────────────────────────────────────
info "Isolation check: running as non-root"
UID_RESULT=$(docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:size=32m,mode=1777 \
    "${SANDBOX_IMAGE}" id -u)

if [[ "${UID_RESULT}" == "10001" ]]; then
    pass "Container runs as uid=10001 (sandbox_user) — not root"
else
    fail "Expected uid=10001, got '${UID_RESULT}'"
    exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
pass "Phase 5 — All sandbox build & isolation checks passed ✓"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  pytest tests/test_sandbox_phase5.py -v   # full test suite"
echo "  # Then proceed to Phase 5 Step 2: integrate into tool_registry.py"