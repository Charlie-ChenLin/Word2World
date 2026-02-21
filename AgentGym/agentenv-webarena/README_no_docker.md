# WebArena (no-Docker) notes for AgentGym

This project assumes the WebArena *websites* are reachable via HTTP. The
official setup is Docker/AMI-based; there are **no official steps** for a
fully native, no-Docker local deployment of all websites. If you want to avoid
Docker, you must either:

1) Use an already-running WebArena website host (AWS AMI or a shared server), or
2) Manually install each website stack yourself (Magento, GitLab, Postmill,
   Kiwix, OpenStreetMap). This is complex and not maintained here.

This file documents the recommended no-Docker path: **point AgentGym at a
running WebArena website host**.

## Apptainer local deployment (host network, no portmap)

On this cluster, unprivileged Apptainer networking (`--net` + slirp4netns) is
disabled, so **port mapping is not available**. The only viable local option is
to run containers on the **host network** and make each service listen on a
unique high port (7770/7780/8023/8888/9999/3000/4399). This is best-effort and
may fail if images require root privileges.

Scripts:
- Download images: `AgentGym/agentenv-webarena/scripts/webarena_apptainer_download.sh`
- Build SIFs: `AgentGym/agentenv-webarena/scripts/webarena_apptainer_build.sh`
- Start services: `AgentGym/agentenv-webarena/scripts/webarena_apptainer_start.sh`
- Healthcheck: `AgentGym/agentenv-webarena/scripts/webarena_apptainer_healthcheck.sh`
- Stop services: `AgentGym/agentenv-webarena/scripts/webarena_apptainer_stop.sh`

**Map backend**: WebArena’s map backend requires a large dataset and long setup
time (AWS instructions in the upstream repo). For fully offline use, map is the
hardest component; expect it to be unavailable unless you provision it
separately.

## 1) Configure website URLs (required)

Export environment variables before running the server or `setup.sh`:

```bash
HOSTNAME="<your-webarena-hostname-or-ip>"
export SHOPPING="http://${HOSTNAME}:7770"
export SHOPPING_ADMIN="http://${HOSTNAME}:7780/admin"
export REDDIT="http://${HOSTNAME}:9999"
export GITLAB="http://${HOSTNAME}:8023"
export MAP="http://${HOSTNAME}:3000"
export WIKIPEDIA="http://${HOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://${HOSTNAME}:4399"

# Optional: for LLM judge
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="<your-openai-base-url>"
```

Note: `setup.sh` and `agentenv_webarena/__init__.py` will **use these env vars
if set**, otherwise default to localhost.

## 2) Python environment setup (no Docker)

Use a Python 3.10+ environment and run:

```bash
cd AgentGym/agentenv-webarena
source setup.sh
```

`setup.sh` installs Playwright; on some systems `playwright install-deps` may
require root. If it fails, install the required system libraries via your OS
package manager or use a prebuilt environment that already satisfies
Playwright's dependencies.

## 3) Sanity check URLs

```bash
bash scripts/webarena_env_check.sh --curl --strict
```

If you only want to print resolved URLs:

```bash
bash scripts/webarena_env_check.sh --basic
```

## 4) Launch the AgentGym WebArena server

```bash
webarena --host 0.0.0.0 --port 8000
```

## 5) CPU-only Slurm check (optional)

See: `projects/Word2World/slurm/webarena-cpu-check.slurm`.
