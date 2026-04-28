#!/usr/bin/env bash
# Smoke-test every /chat/<slug>/ mount discovered from the landing page.
# Exits non-zero if any mount returns != 200.
#
# Usage:
#   .agents/skills/deploy-website/scripts/smoke.sh [base-url]
#
# Default base URL is http://172.233.209.115. Override for staging/local:
#   .agents/skills/deploy-website/scripts/smoke.sh http://127.0.0.1:7777
set -euo pipefail

BASE="${1:-http://172.233.209.115}"
BASE="${BASE%/}"

echo "==> Discovering mounts from ${BASE}/"
landing=$(curl -fsS "${BASE}/")

# Extract unique /chat/<slug>/ paths from href="..." attributes on the landing page.
mapfile -t slugs < <(printf '%s' "${landing}" \
    | grep -oE 'href="/chat/[a-z0-9_-]+/?"' \
    | sed -E 's|href="/chat/([a-z0-9_-]+)/?"|\1|' \
    | sort -u)

if [[ ${#slugs[@]} -eq 0 ]]; then
    echo "ERROR: no /chat/<slug>/ links found on ${BASE}/"
    exit 1
fi

echo "==> Found ${#slugs[@]} mount(s); checking each"

failed=0
for slug in "${slugs[@]}"; do
    url="${BASE}/chat/${slug}/"
    code=$(curl -fsS -o /dev/null -w '%{http_code}' "${url}" || echo "000")
    if [[ "${code}" == "200" ]]; then
        printf '    %-60s %s\n' "${url}" "${code}"
    else
        printf '    %-60s %s  FAIL\n' "${url}" "${code}"
        failed=$((failed + 1))
    fi
done

if [[ "${failed}" -gt 0 ]]; then
    echo "==> ${failed} mount(s) failed"
    exit 1
fi

echo "==> All ${#slugs[@]} mount(s) returned 200"
