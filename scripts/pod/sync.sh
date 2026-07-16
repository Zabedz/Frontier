#!/usr/bin/env bash
# sync.sh down|up : one-shot manual sync.
#   down - pull results, job sentinels, and checkpoints from the pod to local
#   up   - push code and prior results/checkpoints to the pod (part of resume)
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_HERE/lib.sh"

case "${1:-down}" in
  down) pod_pull_results; pod_pull_jobs; pod_pull_checkpoints; echo "[sync] pulled results, jobs, checkpoints" ;;
  up)   pod_push_code; pod_push_results; pod_push_checkpoints; echo "[sync] pushed code, results, checkpoints" ;;
  *)    echo "usage: sync.sh down|up"; exit 1 ;;
esac
