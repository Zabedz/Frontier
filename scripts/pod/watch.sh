#!/usr/bin/env bash
# watch.sh [job] : the resilience watcher. On a cadence it mirrors the pod's results
# and job sentinels down to the local repo (so a pod timeout loses at most one
# interval), heartbeats to logs/watch.log, and exits when the named job finishes or
# the pod goes unreachable. Run in the background; its exit is the signal to act.
#   exit 0 = job DONE   exit 3 = job FAILED   exit 2 = pod unreachable
#   exit 4 = ran past WATCH_MAX_SECONDS (a likely hang; stops burning pod time)
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_HERE/lib.sh"

job="${1:-}"
interval="${WATCH_INTERVAL:-90}"
max="${WATCH_MAX_SECONDS:-1800}"   # money-safety: bail if a job runs longer than this
mkdir -p "$LOCAL_REPO/logs/jobs" "$LOCAL_REPO/results"
wlog="$LOCAL_REPO/logs/watch.log"
fails=0
start=$(date +%s)
last_log_size=-1
stalls=0
tick=0
ckpt_every="${WATCH_CKPT_EVERY:-10}"   # mirror large checkpoints every Nth interval, not every one
echo "$(date -u +%FT%TZ) watcher start job=[$job] interval=${interval}s max=${max}s" >> "$wlog"

while true; do
  ts=$(date -u +%FT%TZ)
  elapsed=$(( $(date +%s) - start ))
  if [ "$elapsed" -ge "$max" ]; then
    echo "$ts $job past ${max}s (elapsed ${elapsed}s), watcher exiting - check for a hang" >> "$wlog"
    exit 4
  fi
  if pod_pull_results && pod_pull_jobs; then
    # results + job sentinels are tiny (KBs), so mirror them every interval. Checkpoints
    # are GBs, so mirror them on a slower cadence to avoid re-pulling an in-progress write.
    tick=$((tick + 1))
    if [ $((tick % ckpt_every)) -eq 0 ]; then pod_pull_checkpoints; fi
    fails=0
    status="none"
    [ -n "$job" ] && status=$(cat "$LOCAL_REPO/logs/jobs/$job.status" 2>/dev/null || echo missing)
    # stall heartbeat: note when a RUNNING job's log has not grown (silent phases like
    # the latency rig are legitimately quiet, so this only warns, the max guard is the exit)
    log_size=$(wc -c < "$LOCAL_REPO/logs/jobs/$job.log" 2>/dev/null || echo 0)
    if [ "$log_size" = "$last_log_size" ]; then stalls=$((stalls + 1)); else stalls=0; fi
    last_log_size=$log_size
    echo "$ts ok job=$job elapsed=${elapsed}s status=[$status] log=${log_size}b stalls=${stalls}" >> "$wlog"
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
