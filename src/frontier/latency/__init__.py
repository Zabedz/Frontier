"""Latency, memory, and machine-state capture: TTFT and inter-token latency timed
separately with CUDA events, warmup discarded, median and p95 over repeated trials;
peak VRAM and KV-cache growth; and per-measurement GPU clock, temperature, and
power so latency numbers are defensible on a machine whose clocks cannot be locked.
"""
