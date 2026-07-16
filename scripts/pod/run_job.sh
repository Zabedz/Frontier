#!/usr/bin/env bash
# run_job.sh <name> <command...> : run a command detached in a tmux session on the pod,
# with a status sentinel (.jobs/<name>.status = RUNNING|DONE|FAILED) and a log
# (.jobs/<name>.log). Detached, so it survives the control SSH dropping.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_HERE/lib.sh"

name="${1:?usage: run_job.sh <name> <command...>}"; shift
cmd="$*"

printf '%s\n' "$cmd" | pod_ssh "cat > $POD_DIR/.jobs/$name.cmd"
pod_ssh "tmux kill-session -t job-$name 2>/dev/null || true; tmux new-session -d -s job-$name 'bash $POD_DIR/.jobs/_wrap.sh $name'"
echo "[run_job] launched job-$name in tmux on the pod"
