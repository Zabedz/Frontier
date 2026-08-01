# Pod image

A pre-baked container image for the RunPod pods, carrying both venvs and every heavy
dependency. A fresh pod is ready in one to two minutes (rsync the code, link it)
against roughly twenty-five minutes for a cold install.

## What is in it

Base `nvidia/cuda:12.8.1-devel-ubuntu22.04`, plus Python 3.11, uv, git, tmux, rsync,
cmake, ninja.

Two venvs on the fast local disk (`/root`), since the two tracks pin different
transformers versions:

- `/root/frontier-venv` (venv-A): the CPU and Hugging Face stack (the base, dev, hf,
  and oracles groups from `uv.lock`; latest transformers 5.x, bitsandbytes). Runs
  Track A: FP16 and bitsandbytes NF4/INT8.
- `/root/.venv-trackB` (venv-B): the serving and quantise stack (vLLM,
  llm-compressor, compressed-tensors, gptqmodel, torchao, llama.cpp; transformers
  pinned to 5.10.1). Runs Track B: GPTQ/AWQ/W8A8 on vLLM, GGUF on llama.cpp.

Neither venv installs the `frontier` package. The code is rsynced to the pod at start
and linked with `uv pip install -e .`, so the image does not go stale when the source
changes.

## The Track-B version pins

venv-B installs in a single `uv pip install` pass with a `compressed-tensors`
override. A split install breaks in two places:

- vLLM 0.25.1 pins `transformers>=5.5.3` with no upper bound. A second, separate
  install re-resolves transformers to the newest release, past llmcompressor's
  supported window (`<=5.10.1`). That drops llmcompressor back to its transformers-4
  line, whose compressed-tensors does `from transformers import PreTrainedModel`
  against the 4.x lazy-module layout and fails with
  `Could not import module 'PreTrainedModel'`. That is the error the first pod hit.
- vLLM pins `compressed-tensors==0.17.0` and llmcompressor pins `==0.17.1`, an
  API-identical patch bump. The `--override` forces `0.17.1` past vLLM's exact pin so
  the single pass resolves.

The torch trio is pinned to its `+cu128` builds by name. `UV_TORCH_BACKEND=cu128`
routes every torch-family package to the cu128 index exclusively, and vLLM's
`torchcodec>=0.14` only exists on PyPI (the cu128 index stops at 0.11.1). torchcodec
0.14.0 is the torch 2.11 pairing, which the resolver cannot check because its
metadata declares no torch requirement, and `--index-strategy unsafe-best-match` lets
each pin pull from whichever index carries that exact build.

## CUDA 13 and the driver floor

Every published vLLM wheel's compiled extension links `libcudart.so.13`, confirmed
with `ldd` on 0.25.1 and 0.21.0 alike. Package metadata hides this: 0.21.0 carries a
plain `nvidia-cutlass-dsl==4.4.2` with no `[cu13]` extras and still links the CUDA 13
runtime. There is no cu128 vLLM wheel to pick, so downgrading does not help.

That runtime ships in the image under `site-packages/nvidia/cu13/lib`, off the default
loader path. The `LD_LIBRARY_PATH` set at the end of the Dockerfile puts it on, and
`scripts/pod/bootstrap_trackb.sh` sets the same path for a pod running an older image.

What remains is a provisioning rule: **CUDA 13 needs an r580 or newer driver.** On an
r570 pod the import resolves and CUDA init returns error 35,
`cudaErrorInsufficientDriver`. The HF, quantise, and GGUF paths run fine on r570.

The venv therefore mixes a cu128 torch with a cu13 vLLM. That imports cleanly because
vLLM builds its extension against torch's stable ABI, which is what
`vllm/_C_stable_libtorch` is.

The Dockerfile holds the full pin list and a CPU-only import probe that trips the
`PreTrainedModel` path at build time, so a bad combination fails at build time, before
a pod is paid for. The probe now imports vLLM as well; the version that skipped that import
is how the CUDA mismatch above reached a pod. The same pins live in
`scripts/pod/bootstrap_trackb.sh` for a pod that comes up without the pre-baked venv.

## Build and push

Build for linux/amd64 (the pods are x86; a native arm64 Mac build will not run). From
the repo root:

```
docker buildx build --platform linux/amd64 -f docker/Dockerfile \
  -t <dockerhub-user>/frontier-pod:cu128 --push .
```

Needs a Docker Hub account and `docker login` first. A green build means the Track-B
stack resolved and imported.

## Use it on RunPod

1. Create or edit a RunPod template. Set Container Image to
   `<dockerhub-user>/frontier-pod:cu128`.
2. Set the Container Start Command to `sleep infinity`.
3. Attach the persistent volume at `/workspace`. Weights, checkpoints, and results
   live there.
4. Set the container disk to 60GB. A pod running this image reads 143MB used, so the
   number is headroom for anything written to `/root`. It matters when the pre-baked
   venv-B is missing and `bootstrap_trackb.sh` falls back to a cold install: venv-A
   alone measured 7.6GB and venv-B is larger, which overruns a 20GB disk.
5. Expose TCP port 22. The `ssh.runpod.io` proxy requires a PTY, which corrupts
   rsync's binary stream, and the pod scripts push code and pull results over rsync.
   Exposed ports are fixed when the pod is created.
6. Start a pod from the template, edit the five values in `scripts/pod/env.sh`, and
   run `scripts/pod/bootstrap.sh`.

`bootstrap.sh` links venv-A and the code. `bootstrap_trackb.sh` detects the pre-baked
venv-B (importing transformers, compressed_tensors, and llmcompressor to confirm it is
intact), skips the reinstall, and links the code. If the image lacks venv-B, the same
script rebuilds it in one pass.

## Status

`zabedz/frontier-pod:cu128` (digest `sha256:38296298`) was built and pushed on
2026-07-17 and first run on a pod on 2026-07-26. It is the current image and needs no
rebuild: the HF, quantise, and GGUF paths all work, and the three GGUF rows were
banked on it. vLLM needs an r580+ pod, as above.

Two defects a running pod works around. Both are fixed in this Dockerfile for
whenever a rebuild happens, and `scripts/pod/bootstrap_trackb.sh` already applies the
second to a live pod:

- The pushed digest ends on `CMD ["/bin/bash"]`. An interactive shell with no TTY
  reads EOF and exits as soon as the container starts, so RunPod shows the pod running
  while its container is dead and the SSH proxy answers `container <id> is not
  running`. Hence the `sleep infinity` start command above.
- Its `/usr/lib` nccl (2.25.1) predates `ncclCommShrink`, and llama_cpp pulls that
  copy in ahead of the venv's 2.28.9, which breaks any torch imported afterwards.
