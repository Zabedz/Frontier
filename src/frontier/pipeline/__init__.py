"""The config-to-row runner and the CLI. Resolves a variant config (smoke or full
mode), dispatches to the right quantise / eval / latency stages, stamps provenance,
and appends one result row. Adding a variant is adding a config file; this package
does not change per variant.
"""
