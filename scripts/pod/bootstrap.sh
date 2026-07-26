#!/usr/bin/env bash
# Bring a pod (this one or a fresh replacement) to a ready state:
#   ensure tools -> install job wrapper -> push code -> push prior results (resume)
#   -> install deps in a detached tmux job named "setup".
# Idempotent: safe to re-run. After it, watch with: scripts/pod/watch.sh setup
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_HERE/lib.sh"

echo "[bootstrap] ensuring tools on pod ($POD_HOST)"
pod_ssh 'command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1); \
         export DEBIAN_FRONTEND=noninteractive; \
         (command -v tmux >/dev/null 2>&1 && command -v rsync >/dev/null 2>&1) || \
           (apt-get update -qq && apt-get install -y -qq tmux rsync >/dev/null 2>&1); \
         mkdir -p '"$POD_DIR"'/.jobs '"$POD_DIR"'/results '"$POD_DIR"'/checkpoints '"$POD_DIR"'/hf-cache'

# The venv must live on the fast local container disk, not the MooseFS /workspace
# volume, or importing torch reads hundreds of .so files over the network (~48s vs
# ~4s). The HF model cache stays on the persistent volume: large sequential reads are
# fine there, and it survives a pod swap so models are not re-downloaded on resume.
echo "[bootstrap] writing pod env (local-disk venv, persistent model cache)"
pod_ssh "cat > $POD_DIR/.jobs/env" <<'ENV'
export PATH=$HOME/.local/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/root/frontier-venv
export HF_HOME=/workspace/frontier/hf-cache
# A cold install downloads GBs of torch wheels. The container disk is small enough that
# the cache and the venv cannot both sit on it, so the cache goes on the volume.
export UV_CACHE_DIR=/workspace/frontier/uv-cache
ENV

echo "[bootstrap] installing job wrapper"
pod_ssh "cat > $POD_DIR/.jobs/_wrap.sh" <<'WRAP'
#!/bin/bash
name="$1"; cd /workspace/frontier
source /workspace/frontier/.jobs/env
echo "RUNNING $(date -u +%FT%TZ)" > ".jobs/$name.status"
bash ".jobs/$name.cmd" > ".jobs/$name.log" 2>&1
ec=$?
if [ "$ec" -eq 0 ]; then echo "DONE $(date -u +%FT%TZ)" > ".jobs/$name.status";
else echo "FAILED $ec $(date -u +%FT%TZ)" > ".jobs/$name.status"; fi
WRAP

echo "[bootstrap] pushing code"
pod_push_code
echo "[bootstrap] pushing prior results and checkpoints (resume)"
pod_push_results
pod_push_checkpoints

echo "[bootstrap] launching dep install (uv sync --group hf) as tmux job 'setup'"
# The job wrapper sources .jobs/env first, so uv sync targets the local-disk venv.
"$_HERE/run_job.sh" setup 'cd /workspace/frontier && uv sync --group hf'
echo "[bootstrap] done. Watch: scripts/pod/watch.sh setup"
