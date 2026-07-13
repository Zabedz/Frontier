"""The result store: append-only parquet with a jsonl mirror, plus provenance
stamping (git SHA, config hash, model revision, hardware id, driver + CUDA
version). Analysis and plotting read from here; no consumer recomputes from raw
model outputs.
"""
