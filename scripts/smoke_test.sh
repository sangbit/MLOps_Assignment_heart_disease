#!/usr/bin/env bash
# Smoke test for a running Heart Disease API.
#
# Usage:  scripts/smoke_test.sh [BASE_URL] [HOST_HEADER]
#   scripts/smoke_test.sh http://127.0.0.1:8000                    # Docker / LoadBalancer
#   scripts/smoke_test.sh http://127.0.0.1 heart-api.local         # through the Ingress
#
# Exits with code 1 if any check fails, so CI can use it as a gate.

set -euo pipefail
BASE_URL="${1:-http://127.0.0.1:8000}"
HOST_HEADER="${2:-}"
EXTRA=()
[[ -n "$HOST_HEADER" ]] && EXTRA=(-H "Host: $HOST_HEADER")
PATIENT='{"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
          "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3}'

pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; exit 1; }

echo "Smoke testing $BASE_URL ${HOST_HEADER:+(Host: $HOST_HEADER)}"

# 1. Health
body=$(curl -sf "${EXTRA[@]}" "$BASE_URL/health") || fail "GET /health did not answer"
[[ "$body" == *'"status":"ok"'* ]] && pass "GET /health -> $body" || fail "GET /health -> $body"

# 2. Valid prediction
body=$(curl -sf "${EXTRA[@]}" -X POST "$BASE_URL/predict" \
       -H "Content-Type: application/json" -d "$PATIENT") || fail "POST /predict failed"
[[ "$body" == *'"confidence"'* ]] && pass "POST /predict -> $body" || fail "POST /predict -> $body"

# 3. Invalid input must be rejected with 422
code=$(curl -s -o /dev/null -w "%{http_code}" "${EXTRA[@]}" -X POST "$BASE_URL/predict" \
       -H "Content-Type: application/json" -d '{"age": 250, "sex": 1}')
[[ "$code" == "422" ]] && pass "POST /predict (invalid input) -> HTTP 422" || fail "invalid input -> HTTP $code"

echo "All smoke tests passed."
