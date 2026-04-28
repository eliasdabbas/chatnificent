#!/usr/bin/env bash
# Deploy the Chatnificent website to production.
#
# Prerequisite: `git push` already done. This script DOES NOT push for you.
#
# Usage:
#   .agents/skills/deploy-website/scripts/deploy.sh           # default: reload (zero-downtime)
#   .agents/skills/deploy-website/scripts/deploy.sh --restart # full restart (env / unit changes)
set -euo pipefail

HOST="${CHATNIFICENT_HOST:-172.233.209.115}"
USER="${CHATNIFICENT_SSH_USER:-elias}"
SERVICE="chatnificent-website"
APP_DIR="/srv/chatnificent/app"
LOCAL_HEALTH_PATHS=("/" "/chat/quickstart/")

ACTION="reload"
if [[ "${1:-}" == "--restart" ]]; then
    ACTION="restart"
fi

echo "==> Deploying to ${USER}@${HOST} (action=${ACTION})"

# 1. Pull latest as service user, then reload/restart.
ssh -o BatchMode=yes "${USER}@${HOST}" bash -s -- "${APP_DIR}" "${SERVICE}" "${ACTION}" <<'REMOTE'
set -euo pipefail
APP_DIR="$1"
SERVICE="$2"
ACTION="$3"

echo "--> git fetch + reset on ${APP_DIR}"
sudo -u chatnificent bash -lc "
    set -euo pipefail
    cd '${APP_DIR}'
    git fetch --prune origin
    git reset --hard origin/main
    git rev-parse --short HEAD
"

echo "--> sudo systemctl ${ACTION} ${SERVICE}"
sudo systemctl "${ACTION}" "${SERVICE}"

echo "--> health check on 127.0.0.1:7777"
for i in $(seq 1 20); do
    if curl -fsS -o /dev/null -w '%{http_code}' http://127.0.0.1:7777/ | grep -q '^200$'; then
        echo "    OK after ${i} attempt(s)"
        break
    fi
    if [[ "$i" == "20" ]]; then
        echo "ERROR: 127.0.0.1:7777 did not return 200 after 20 attempts"
        sudo journalctl -u "${SERVICE}" -n 50 --no-pager
        exit 1
    fi
    sleep 1
done
REMOTE

# 2. Public health check from local machine.
echo "==> Public health check"
for path in "${LOCAL_HEALTH_PATHS[@]}"; do
    url="http://${HOST}${path}"
    for i in $(seq 1 10); do
        code=$(curl -fsS -o /dev/null -w '%{http_code}' "${url}" || echo "000")
        if [[ "${code}" == "200" ]]; then
            printf '    %-50s OK\n' "${url}"
            break
        fi
        if [[ "$i" == "10" ]]; then
            echo "ERROR: ${url} returned ${code} after 10 attempts"
            exit 1
        fi
        sleep 1
    done
done

# 3. Smoke test all mounted apps.
SMOKE="$(dirname "$0")/smoke.sh"
if [[ -x "${SMOKE}" ]]; then
    echo "==> Smoke test (every /chat/<slug>/ mount)"
    "${SMOKE}" "http://${HOST}"
fi

echo "==> Deploy complete"
