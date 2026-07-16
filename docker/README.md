# Pod image

A pre-baked container image for the RunPod pods. It carries both virtual
environments and every heavy dependency, so a fresh pod is ready in one to two
minutes (rsync the code, link it) instead of the roughly twenty-five minutes a
cold install takes.

## What is in it

Base `nvidia/cuda:12.8.1-devel-ubuntu22.04`, plus Python 3.11, uv, and git,
tmux, rsync, cmake, ninja.

Two virtual environments on the fast local disk (`/root`), because the two
tracks pin different transformers versions and cannot share one venv:

- `/root/frontier-venv` (venv-A): the CPU and Hugging Face stack (the base, dev,
  hf, and oracles groups from `uv.lock`; latest transformers 5.x, bitsandbytes).
  This runs Track A: FP16 and bitsandbytes NF4/INT8.
- `/root/.venv-trackB` (venv-B): the serving and quantise stack (vLLM,
  llm-compressor, compressed-tensors, gptqmodel, torchao, and llama.cpp;
  transformers pinned to 5.10.1). This runs Track B: GPTQ/AWQ/W8A8 served by
  vLLM, GGUF by llama.cpp.

Neither venv installs the `frontier` package. The code is rsynced to the pod at
start and linked with `uv pip install -e .`, so the image does not go stale when
the source changes.

## The Track-B version pins

venv-B is installed in a single `uv pip install` pass with a `compressed-tensors`
override. The single pass matters:

- vLLM 0.25.1 pins `transformers>=5.5.3` with no upper bound. A second, separate
  install re-resolves transformers to the newest release, which is past
  llmcompressor's supported window (`<=5.10.1`). That drops llmcompressor back to
  its transformers-4 line, whose compressed-tensors does
  `from transformers import PreTrainedModel` against the 4.x lazy-module layout
  and fails with `Could not import module 'PreTrainedModel'`. That is the error
  the first pod hit.
- vLLM pins `compressed-tensors==0.17.0` and llmcompressor pins `==0.17.1`, an
  API-identical patch bump. The `--override` forces `0.17.1` past vLLM's exact
  pin so the single pass resolves.

The Dockerfile holds the full pin list and a CPU-only import probe that trips the
`PreTrainedModel` path at build time, so a bad combination fails the build rather
than a paid pod. The same pins live in `scripts/pod/bootstrap_trackb.sh` for the
case where a pod comes up without the pre-baked venv.

## Build and push

Build for linux/amd64 (the pods are x86; do not build natively on an arm64 Mac).
From the repo root:

```
docker buildx build --platform linux/amd64 -f docker/Dockerfile \
  -t <dockerhub-user>/frontier-pod:cu128 --push .
```

This needs a Docker Hub account and `docker login` first. buildx cross-builds the
x86 image on the Mac, and `--push` uploads it. The build runs the venv-B import
probe, so a green build means the Track-B stack resolved and imported.

## Use it on RunPod

1. Create or edit a RunPod template. Set Container Image to
   `<dockerhub-user>/frontier-pod:cu128`.
2. Attach the persistent volume at `/workspace`. Weights, checkpoints, and
   results live there, not in the image.
3. Start a pod from the template, edit the five values in
   `scripts/pod/env.sh`, and run `scripts/pod/bootstrap.sh`.

`bootstrap.sh` links venv-A and the code. `bootstrap_trackb.sh` detects the
pre-baked venv-B (it imports transformers, compressed_tensors, and llmcompressor
to confirm the venv is intact), skips the reinstall, and links the code. If the
image ever lacks venv-B, the same script rebuilds it in one pass, so the pod
still comes up.

## Note

The image is not built or run yet. The pins come from current PyPI metadata; the
build validates them, and it runs offline with no pod cost. Build and push once,
then every pod after that starts from it.
