# Pod harness

Drives a RunPod GPU pod over SSH with two guarantees: work survives an SSH drop,
and a pod can be terminated and resumed on a fresh one without losing data.

## How it holds up

- **Jobs run detached.** `run_job.sh` starts each command in a `tmux` session on the
  pod with a status sentinel, so the control SSH connection dropping never kills work.
- **The local repo is the source of truth.** `watch.sh` mirrors the pod's `results/`
  and job sentinels down on a cadence, so a pod timeout loses at most one interval.
- **The store is append-only.** A resumed run skips variants already recorded, so
  re-running the same command picks up where it left off.

## Connection

`env.sh` (git-ignored) holds the current pod's host, port, key, and dirs. It is the
only file to edit when the pod changes.

## Everyday use

```bash
scripts/pod/bootstrap.sh            # ready a pod: tools, code, prior results, deps (tmux job 'setup')
scripts/pod/watch.sh setup          # run in the background; mirrors down, exits when 'setup' is DONE
scripts/pod/run_job.sh <name> '<cmd>'   # run <cmd> detached on the pod
scripts/pod/sync.sh down            # one-shot pull of results, job sentinels, checkpoints
scripts/pod/sync.sh up              # push code + results + checkpoints (used by resume)
```

The watcher exits with 0 (job DONE), 3 (job FAILED), or 2 (pod unreachable x3); its
exit is the signal to inspect `logs/` and act.

## Terminate one pod, resume on another

1. `scripts/pod/sync.sh down` to make sure local has the latest results and checkpoints.
2. Terminate the old pod in the RunPod console.
3. Provision a new pod, edit the five values in `env.sh`.
4. `scripts/pod/bootstrap.sh` pushes code + the prior results/checkpoints up and
   reinstalls deps.
5. Re-run the pipeline command; the append-only store skips finished variants.
