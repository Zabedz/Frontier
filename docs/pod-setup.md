# Pod setup

The GPU stack is pod-only. It is deliberately not in `pyproject.toml` or `uv.lock`
during CPU development: its versions get pinned against the real CUDA driver and
card on the pod, not resolved on a laptop. Add the group below to `pyproject.toml`
when the pod is provisioned (WP4), then `uv lock` on the pod so the lock reflects
the Linux/CUDA resolution.

## The GPU dependency group

Add to `[dependency-groups]` in `pyproject.toml` on the pod, and re-verify each
version against current releases before pinning (the toolchain moved a lot through
2025-2026):

```toml
gpu = [
    "torch>=2.5",             # install from the CUDA index matching the pod driver
    "transformers>=4.44",
    "accelerate>=0.33",
    "peft>=0.12",
    "trl>=0.11",              # GKD trainer, for the single online reference run only
    "vllm>=0.6",              # serves compressed-tensors W4A16 / W8A8 on marlin
    "llmcompressor>=0.3",     # GPTQ + AWQ producer (compressed-tensors)
    "compressed-tensors>=0.6",
    "bitsandbytes>=0.43",     # NF4 (QLoRA path) and LLM.int8 weight-only
    "torchao>=0.7",           # QAT (prepare/convert) + int8 weight-only
    "gptqmodel>=1.0",         # secondary GPTQ path, for cross-checking scale selection
    # llama-cpp-python is built from source with CUDA, see below
]
```

`llama-cpp-python` needs a CUDA build so GGUF runs on the GPU, not the CPU:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
```

Unsloth (optional QAT accelerator, GPU-only) is installed per its current install
instructions; it is never on the CPU smoke path.

## Bring-up checklist

1. Log the pod. Every change to the pod's state goes in `SSH_CHANGELOG.md`
   (git-ignored, local), newest first, with a revert command. Keep work in the
   container / a scratch dir, off shared disks.
2. Record the environment for provenance: `nvidia-smi` (GPU model, driver, CUDA),
   card memory, and the pod id. These populate the result-row `Provenance` and
   `Backend` fields.
3. Probe clock control (it usually fails on RunPod; find out on the real pod):
   ```bash
   nvidia-smi -pm 1
   nvidia-smi --lock-gpu-clocks=<MHz> --lock-memory-clocks=<MHz>
   ```
   If it returns "Insufficient Permissions", set `clocks_locked=false` and fall
   back to the quiet-machine protocol in `docs/methodology.md` (log clocks and
   temperature per measurement, warm up, many repeats, report the clock range).
4. First real spend is WP3's latency rig, which measures the pod's true throughput.
   Re-confirm the WP6/WP7 training-token budgets against that number before running
   the expensive arms. Measure, do not assume.
