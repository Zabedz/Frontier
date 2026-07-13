"""Distillation: the one-time offline teacher top-k logit cache (k plus a kept
"rest" bucket so truncation does not bias calibration), the hard-label sequence-KD
baseline, and the offline soft-label KL student trainer. Students are full
fine-tuned, never LoRA, so the training regime does not confound the calibration
comparison.
"""
