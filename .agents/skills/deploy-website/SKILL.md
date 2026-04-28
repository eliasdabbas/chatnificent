---
name: deploy-website
description: Deploy the Chatnificent website (examples/chatnificent_website.py) to the Linode production server. Use when the user asks to deploy, ship, push to prod, or release the website. Runs the push-already-done → pull → reload → verify loop.
---

# Deploy Chatnificent Website

Use this skill to release `examples/chatnificent_website.py` to
`http://172.233.209.115/`.

> **Prerequisite:** `git push` has already happened. This skill does NOT push
> for you — it pulls on the server and reloads the running service.

## Server Details

| Item | Value |
|---|---|
| Host | `172.233.209.115` |
| SSH user | `elias` (passwordless sudo) |
| Service user | `chatnificent` (system, no shell login) |
| Repo path | `/srv/chatnificent/app` |
| Data path | `/srv/chatnificent/data` (conversations / uploads, **outside** the repo) |
| Env file | `/srv/chatnificent/chatnificent.env` (mode 0600) |
| Service unit | `chatnificent-website.service` |
| nginx site | `/etc/nginx/sites-available/chatnificent` |
| Process bind | `127.0.0.1:7777` (proxied by nginx) |

## Standard Deploy

Run from the repo root on your local machine:

```bash
.agents/skills/deploy-website/scripts/deploy.sh
```

The script:

1. SSH to `elias@172.233.209.115`.
2. As `chatnificent`: `git fetch --prune origin && git reset --hard origin/main`.
3. `sudo systemctl reload chatnificent-website` (zero-downtime via SIGHUP).
4. Health-check loop on the server: `curl http://127.0.0.1:7777/` until `200`.
5. Public health check from your local machine: `/` and `/chat/quickstart/`.
6. Smoke-test every `/chat/<slug>/` mount discovered on the landing page.
7. Exit non-zero on any failure.

> No `uv sync` step. The website is a PEP 723 inline-script app — `uv run
> --script` resolves its declared deps on the fly into uv's global cache. The
> repo's top-level `pyproject.toml` is irrelevant when running with `--script`.

### When to use `--restart` instead of reload

```bash
.agents/skills/deploy-website/scripts/deploy.sh --restart
```

| Command | Behavior | When |
|---|---|---|
| `systemctl reload` (`SIGHUP`) | uvicorn graceful reload, in-flight requests finish | **Default** — code-only changes |
| `systemctl restart` | Full stop + start, ~1–2s of 502s | Unit file or `chatnificent.env` changed |
| `kill -TERM $MAINPID` | Graceful stop, no auto-restart | Manual maintenance |
| `kill -9 $MAINPID` | Force kill | Last resort |

## Rollback

```bash
.agents/skills/deploy-website/scripts/rollback.sh <git-sha>
```

`git reset --hard <sha>` on the server, reload, same health checks.

## Smoke Test (without deploying)

```bash
.agents/skills/deploy-website/scripts/smoke.sh                          # against prod
.agents/skills/deploy-website/scripts/smoke.sh http://127.0.0.1:7777    # against local dev
```

Discovers every `/chat/<slug>/` from the landing-page HTML and asserts each
returns `200`.

## First-Time Server Bootstrap

Only run once per server. Subsequent deploys use the script above.

### Phase 1 — System packages + service user

```bash
ssh elias@172.233.209.115
sudo apt update && sudo apt install -y git curl ca-certificates nginx
# uv manages Python itself — do NOT install python3 / python3-venv from apt.
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh
which uv  # must print /usr/local/bin/uv

sudo useradd --system --create-home --home-dir /srv/chatnificent --shell /bin/bash chatnificent
sudo -u chatnificent git clone https://github.com/eliasdabbas/chatnificent.git /srv/chatnificent/app
sudo -u chatnificent mkdir -p /srv/chatnificent/data
```

### Phase 2 — Secrets file

```bash
# Atomically create an empty file owned by chatnificent, mode 0600.
# install(1) is cp(1) with built-in mode/owner flags; /dev/null is the empty source.
sudo install -m 0600 -o chatnificent -g chatnificent /dev/null /srv/chatnificent/chatnificent.env

# Edit it. Add OPENAI_API_KEY=..., ANTHROPIC_API_KEY=..., GEMINI_API_KEY=..., OPENROUTER_API_KEY=...
sudo -u chatnificent nano /srv/chatnificent/chatnificent.env
```

### Phase 3 — systemd unit

`scp` cannot write to `/etc/...` directly (`elias` is not root). Land the file
in `/tmp`, then `sudo install` it:

```bash
# from the repo root, locally:
scp deploy/chatnificent-website.service elias@172.233.209.115:/tmp/
ssh elias@172.233.209.115 \
    'sudo install -m 0644 /tmp/chatnificent-website.service /etc/systemd/system/ \
     && rm /tmp/chatnificent-website.service \
     && sudo systemctl daemon-reload \
     && sudo systemctl enable --now chatnificent-website'

ssh elias@172.233.209.115 'systemctl status chatnificent-website --no-pager'
ssh elias@172.233.209.115 'curl -sI http://127.0.0.1:7777/'
```

First start downloads Python + PEP 723 deps via `uv run --script`. Watch the
log: `ssh elias@172.233.209.115 'sudo journalctl -u chatnificent-website -f'`.

### Phase 4 — nginx site

```bash
scp deploy/nginx.conf elias@172.233.209.115:/tmp/
ssh elias@172.233.209.115 \
    'sudo install -m 0644 /tmp/nginx.conf /etc/nginx/sites-available/chatnificent \
     && rm /tmp/nginx.conf \
     && sudo ln -sf /etc/nginx/sites-available/chatnificent /etc/nginx/sites-enabled/chatnificent \
     && sudo rm -f /etc/nginx/sites-enabled/default \
     && sudo nginx -t \
     && sudo systemctl reload nginx'

ssh elias@172.233.209.115 'sudo ufw allow "Nginx HTTP" || true'
curl -sI http://172.233.209.115/   # from your laptop
.agents/skills/deploy-website/scripts/smoke.sh   # full mount audit
```

## Verification (after every deploy)

1. `systemctl is-active chatnificent-website` → `active`
2. `curl -sI http://127.0.0.1:7777/` (on server) → `200 OK`
3. `curl -sI http://172.233.209.115/` (from laptop) → `200 OK`, `Server: nginx`
4. `smoke.sh` exits 0
5. Open `http://172.233.209.115/`, click into `/chat/quickstart/`, send one
   message. Reply must stream incrementally (validates `proxy_buffering off`).
6. `sudo journalctl -u chatnificent-website -n 50 --no-pager` shows no tracebacks.

## What This Skill Does NOT Do

- Does not run `git push`. Push first, then deploy.
- Does not edit the systemd unit or nginx config on the server. When
  `deploy/chatnificent-website.service` or `deploy/nginx.conf` changes, copy
  them manually using the commands at the top of each file.
- Does not handle secret rotation, schema migrations, or auto-deploy on push.

## Why These Choices

- **`git fetch --prune` + `git reset --hard origin/main`** (not `git pull`):
  the server checkout is a deployment artifact, not a workspace. `pull` can
  produce merge commits or conflicts; `reset --hard` makes the checkout
  exactly match what was pushed, every time.
- **No `uv sync`** in deploy: `uv sync` syncs from `pyproject.toml` /
  `uv.lock`. The website is a PEP 723 script and uses `uv run --script`,
  which resolves its inline deps independently.
- **Custom `CHATNIFICENT_WEBSITE_DATA=/srv/chatnificent/data`** on the
  service: every deploy `git reset --hard`s the repo. User conversations and
  uploaded files MUST live outside the checkout or they'd be wiped.
- **scp via `/tmp` + `sudo install`**: SFTP cannot elevate privileges
  mid-transfer. Two-step is the standard pattern.

## Troubleshooting

- **502 Bad Gateway** — uvicorn is down.
  `sudo systemctl status chatnificent-website`,
  `sudo journalctl -u chatnificent-website -n 200 --no-pager`.
- **Streaming feels chunked / laggy** — nginx is buffering. Confirm
  `proxy_buffering off;` and `proxy_http_version 1.1;` in the live config.
- **`uv` not found by systemd** — installer didn't put `uv` on
  `/usr/local/bin`. Symlink it: `sudo ln -sf $(which uv) /usr/local/bin/uv`.
- **404 on a `/chat/<slug>/` mount** — the `_StripMountPrefix` shim in
  `examples/chatnificent_website.py` must wrap the mounted child. Don't revert it.
- **First start hangs ~30s** — uv is downloading Python + deps. Normal once,
  fast on every subsequent boot (cache lives in `/srv/chatnificent/.cache/uv`).
