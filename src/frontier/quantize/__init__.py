"""Post-training quantisation producers and QAT: llm-compressor (GPTQ/AWQ,
compressed-tensors), bitsandbytes (NF4, LLM.int8), torchao (int8 weight-only and QAT
prepare/convert), and GGUF via llama.cpp. The calibration corpus is an explicit argument
throughout, because it is an axis of the study.
"""
