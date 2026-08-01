# Pod scripts

Drives a RunPod GPU pod over SSH with two guarantees: work survives an SSH drop, and a
pod can be terminated and resumed on a fresh one without losing data.

## Guarantees

- **Jobs run detached.** `run_job.sh` starts each command in a `tmux` session on the
  pod with a status sentinel, so a dropped control connection never kills work.
- **The local repo is the source of truth.** `watch.sh` mirrors the pod's `results/`
  and job sentinels down on a cadence, so a pod timeout loses at most one interval.
- **The store is append-only.** A resumed run skips variants already recorded.

## Connection

`env.sh` (git-ignored) holds the current pod's host, port, key, and dirs. It is the
only file to edit when the pod changes.

## Everyday use

```bash
scripts/pod/bootstrap.sh            # ready a pod: tools, code, prior results, venv-A deps
scripts/pod/bootstrap_trackb.sh     # link or build venv-B and write the Track-B job env
scripts/pod/watch.sh setup          # mirrors down in the background; exits when 'setup' is DONE
scripts/pod/run_job.sh <name> '<cmd>'   # run <cmd> detached on the pod
scripts/pod/run_variants.sh <variant>...  # batch: picks the venv per variant
scripts/pod/sync.sh down            # pull results, job sentinels, checkpoints
scripts/pod/sync.sh up              # push code + results + checkpoints
```

`TRACKB_LATENCY=1` enables the Track-B latency probes in `run_variants.sh`.

The watcher exits 0 (job DONE), 3 (job FAILED), or 2 (pod unreachable x3). Inspect
`logs/` on a non-zero exit.

## Terminate one pod, resume on another

1. `scripts/pod/sync.sh down` to bring local up to date.
2. Terminate the old pod in the RunPod console.
3. Provision a new pod, edit the five values in `env.sh`.
4. `scripts/pod/bootstrap.sh` pushes code plus the prior results and checkpoints up,
   and reinstalls deps.
5. Re-run the pipeline command; the append-only store skips finished variants.
