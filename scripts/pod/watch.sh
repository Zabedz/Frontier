#!/usr/bin/env bash
# watch.sh [job] : the resilience watcher. On a cadence it mirrors the pod's results
# and job sentinels down to the local repo (so a pod timeout loses at most one
# interval), heartbeats to logs/watch.log, and exits when the named job finishes or
# the pod goes unreachable. Run in the background; its exit is the signal to act.
#   exit 0 = job DONE   exit 3 = job FAILED   exit 2 = pod unreachable
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_HERE/lib.sh"

job="${1:-}"
interval="${WATCH_INTERVAL:-90}"
mkdir -p "$LOCAL_REPO/logs/jobs" "$LOCAL_REPO/results"
wlog="$LOCAL_REPO/logs/watch.log"
fails=0
echo "$(date -u +%FT%TZ) watcher start job=[$job] interval=${interval}s" >> "$wlog"

while true; do
  ts=$(date -u +%FT%TZ)
  if pod_pull_results && pod_pull_jobs; then
    fails=0
    status="none"
    [ -n "$job" ] && status=$(cat "$LOCAL_REPO/logs/jobs/$job.status" 2>/dev/null || echo missing)
    echo "$ts ok job=$job status=[$status]" >> "$wlog"
    case "$status" in
      DONE*)   echo "$ts $job DONE, watcher exiting" >> "$wlog"; exit 0 ;;
      FAILED*) echo "$ts $job FAILED, watcher exiting" >> "$wlog"; exit 3 ;;
    esac
  else
    fails=$((fails + 1))
    echo "$ts POD_UNREACHABLE ($fails/3)" >> "$wlog"
    if [ "$fails" -ge 3 ]; then
      echo "$ts pod unreachable x$fails, watcher exiting" >> "$wlog"; exit 2
    fi
  fi
  sleep "$interval"
done
