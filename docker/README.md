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

## CUDA 13 and the driver floor

Every published vLLM wheel's compiled extension links `libcudart.so.13`, confirmed
with `ldd` on 0.25.1 and 0.21.0 alike. Metadata is not the tell: 0.21.0 carries a
plain `nvidia-cutlass-dsl==4.4.2` with no `[cu13]` extras and still links the CUDA
13 runtime. There is no cu128 vLLM wheel to pick, so downgrading does not help.

That runtime is already in the image, under
`site-packages/nvidia/cu13/lib`, just off the default loader path. The
`LD_LIBRARY_PATH` set at the end of the Dockerfile puts it on, and
`scripts/pod/bootstrap_trackb.sh` sets the same path for a pod running an older
image.

What remains is a provisioning rule: **CUDA 13 needs an r580 or newer driver.** On
an r570 pod the import resolves and CUDA init returns error 35,
`cudaErrorInsufficientDriver`. The HF, quantise, and GGUF paths are unaffected and
run fine on r570.

The venv therefore mixes a cu128 torch with a cu13 vLLM. That imports cleanly
because vLLM builds its extension against torch's stable ABI, which is what
`vllm/_C_stable_libtorch` is.

The Dockerfile holds the full pin list and a CPU-only import probe that trips the
`PreTrainedModel` path at build time, so a bad combination fails the build rather
than a paid pod. The probe imports vLLM too: the version that skipped it is how
the CUDA mismatch below reached a pod. The same pins live in
`scripts/pod/bootstrap_trackb.sh` for the case where a pod comes up without the
pre-baked venv.

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
3. Set the container disk to 60GB. The image's own layers do not consume it (a
   pod running this image reads 143MB used on a 60GB container disk), so the
   number is headroom for anything written to `/root`. It matters when the
   pre-baked venv-B is missing and `bootstrap_trackb.sh` falls back to a cold
   install: venv-A alone measured 7.6GB and venv-B is larger, which overruns a
   20GB disk.
4. Expose TCP port 22. The `ssh.runpod.io` proxy requires a PTY, which corrupts
   rsync's binary stream, and the harness pushes code and pulls results over
   rsync. Exposed ports are fixed when the pod is created.
5. Start a pod from the template, edit the five values in
   `scripts/pod/env.sh`, and run `scripts/pod/bootstrap.sh`.

`bootstrap.sh` links venv-A and the code. `bootstrap_trackb.sh` detects the
pre-baked venv-B (it imports transformers, compressed_tensors, and llmcompressor
to confirm the venv is intact), skips the reinstall, and links the code. If the
image ever lacks venv-B, the same script rebuilds it in one pass, so the pod
still comes up.

## Status

`zabedz/frontier-pod:cu128` (digest `sha256:38296298`) was built and pushed on
2026-07-17 and first run on a pod on 2026-07-26. It is the current image and needs
no rebuild: its HF, quantise, and GGUF paths all work, and the three GGUF rows were
banked on it. vLLM needs an r580+ pod, as above.

Two things it gets wrong that a running pod has to work around. Both are fixed in
this Dockerfile for whenever a rebuild happens for some other reason, and
`scripts/pod/bootstrap_trackb.sh` already applies the second to a live pod:

- It ends on `CMD ["/bin/bash"]`. An interactive shell with no TTY reads EOF and
  exits as soon as the container starts, so RunPod shows the pod running while its
  container is dead and the SSH proxy answers `container <id> is not running`. A
  pod started from this digest needs its Container Start Command set to
  `sleep infinity`.
- Its `/usr/lib` nccl (2.25.1) predates `ncclCommShrink`, and llama_cpp pulls that
  copy in ahead of the venv's 2.28.9, which breaks any torch imported afterwards.
