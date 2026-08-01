#!/usr/bin/env bash
# Shared helpers for the pod scripts. Sources the (git-ignored) connection details
# and defines ssh/rsync wrappers. The local repo is the source of truth: results and
# checkpoints are mirrored down continuously and pushed back up to resume on a new pod.
set -uo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_HERE/env.sh"

_SSH_OPTS=(-p "$POD_PORT" -i "$POD_KEY" -o BatchMode=yes -o ConnectTimeout=15
           -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
_SSH_E="ssh -p $POD_PORT -i $POD_KEY -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"

# Never push generated, heavy, or local-only trees to the pod. .jobs and results live
# on the pod and are protected from --delete by being excluded here.
_PUSH_EXCLUDES=(--exclude '.venv' --exclude 'results' --exclude 'plots' --exclude 'logs'
                --exclude 'checkpoints' --exclude '.jobs' --exclude 'hf-cache'
                --exclude '__pycache__'
                --exclude '.mypy_cache' --exclude '.ruff_cache' --exclude '.pytest_cache'
                --exclude '.DS_Store')

pod_ssh() { ssh "${_SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "$@"; }

# The RunPod /workspace volume (MooseFS) rejects chown, so pushes must not preserve
# owner/group/perms; -rltz carries content, links, and mtimes, which is all we need.
_PUSH_FLAGS=(-rltz --no-owner --no-group --no-perms)
_PULL_FLAGS=(-az)   # local receiver, ownership is fine

pod_push_code() {
  rsync "${_PUSH_FLAGS[@]}" --delete "${_PUSH_EXCLUDES[@]}" -e "$_SSH_E" \
    "$LOCAL_REPO/" "$POD_USER@$POD_HOST:$POD_DIR/"
}
pod_push_results() {
  [ -d "$LOCAL_REPO/results" ] || return 0
  pod_ssh "mkdir -p $POD_DIR/results"
  rsync "${_PUSH_FLAGS[@]}" -e "$_SSH_E" "$LOCAL_REPO/results/" "$POD_USER@$POD_HOST:$POD_DIR/results/"
}
pod_push_checkpoints() {
  [ -d "$LOCAL_REPO/checkpoints" ] || return 0
  pod_ssh "mkdir -p $POD_DIR/checkpoints"
  rsync "${_PUSH_FLAGS[@]}" -e "$_SSH_E" "$LOCAL_REPO/checkpoints/" "$POD_USER@$POD_HOST:$POD_DIR/checkpoints/"
}
pod_pull_results() {
  mkdir -p "$LOCAL_REPO/results"
  rsync "${_PULL_FLAGS[@]}" -e "$_SSH_E" "$POD_USER@$POD_HOST:$POD_DIR/results/" "$LOCAL_REPO/results/"
}
pod_pull_jobs() {
  mkdir -p "$LOCAL_REPO/logs/jobs"
  rsync "${_PULL_FLAGS[@]}" -e "$_SSH_E" "$POD_USER@$POD_HOST:$POD_DIR/.jobs/" "$LOCAL_REPO/logs/jobs/" 2>/dev/null || true
}
pod_pull_checkpoints() {
  mkdir -p "$LOCAL_REPO/checkpoints"
  rsync "${_PULL_FLAGS[@]}" -e "$_SSH_E" "$POD_USER@$POD_HOST:$POD_DIR/checkpoints/" "$LOCAL_REPO/checkpoints/" 2>/dev/null || true
}
