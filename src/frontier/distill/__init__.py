"""Distillation: a one-time offline teacher cache (top-k probabilities plus a kept "rest"
bucket, so truncation does not bias calibration) and three full fine-tuned student arms on
its rows: soft labels (the top-k-plus-rest KL), token-level hard labels (the teacher's
top-1), and the corpus-label control (the corpus's own next token). Full fine-tuning keeps
the training regime out of the calibration comparison.
"""
