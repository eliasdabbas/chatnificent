# Deployment Reference Files

Version-controlled copies of the production server config for
`examples/chatnificent_website.py`, deployed at `http://172.233.209.115/`.

| File | Server destination |
|---|---|
| `chatnificent-website.service` | `/etc/systemd/system/chatnificent-website.service` |
| `nginx.conf` | `/etc/nginx/sites-available/chatnificent` |

These are **reference copies**. The deploy skill at
`.agents/skills/deploy-website/` does NOT push them automatically. When you
change one of these files, copy it to the server using the install commands
embedded at the top of each file.

## Architecture

```
client → nginx :80 (172.233.209.115)
            └─ proxy_pass → uvicorn 127.0.0.1:7777
                              └─ systemd: chatnificent-website.service
                                    └─ user: chatnificent (no shell login)
                                    └─ cwd:  /srv/chatnificent/app   (git checkout, reset --hard on every deploy)
                                    └─ data: /srv/chatnificent/data  (conversations / uploads, persistent)
                                    └─ exec: uv run --script examples/chatnificent_website.py
                                    └─ env:  /srv/chatnificent/chatnificent.env (0600)
```

Single uvicorn worker — `build_site()` builds shared in-memory state once at
startup; multi-worker would duplicate state and break per-process `InMemory`
stores. Revisit if traffic warrants it.

See `.agents/skills/deploy-website/SKILL.md` for the full bootstrap runbook.
