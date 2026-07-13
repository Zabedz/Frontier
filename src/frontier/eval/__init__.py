"""Task evaluation: dataset loaders (MMLU / MMLU-Redux / ARC), prompt building,
answer-letter confidence extraction with cyclic permutation debiasing, and
exact-match correctness. Produces the per-item (confidence, correct) pairs the
metrics package consumes.
"""
