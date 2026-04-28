#!/usr/bin/env bash
# Roll the production website back to a specific git SHA.
#
# Usage:
#   .agents/skills/deploy-website/scripts/rollback.sh <git-sha>
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <git-sha>"
    exit 2
fi

SHA="$1"
HOST="${CHATNIFICENT_HOST:-172.233.209.115}"
USER="${CHATNIFICENT_SSH_USER:-elias}"
SERVICE="chatnificent-website"
APP_DIR="/srv/chatnificent/app"

echo "==> Rolling back ${USER}@${HOST} to ${SHA}"

ssh -o BatchMode=yes "${USER}@${HOST}" bash -s -- "${APP_DIR}" "${SERVICE}" "${SHA}" <<'REMOTE'
set -euo pipefail
APP_DIR="$1"
SERVICE="$2"
SHA="$3"

sudo -u chatnificent bash -lc "
    set -euo pipefail
    cd '${APP_DIR}'
    git fetch --prune origin
    git reset --hard '${SHA}'
    git rev-parse --short HEAD
"

sudo systemctl reload "${SERVICE}"

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

curl -fsS -o /dev/null -w '%{http_code}\n' "http://${HOST}/"
echo "==> Rollback complete"
