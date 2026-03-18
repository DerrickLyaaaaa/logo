import numpy as np


def rbo_item_scores(rank_a, rank_b, p=0.9):
    """
    Item-level score induced by Rank-Biased Overlap cumulative overlaps.
    Higher means an item appears early in both rankings.
    """
    if len(rank_a) != len(rank_b):
        raise ValueError("rank_a and rank_b must have the same length")

    n = len(rank_a)
    if n == 0:
        return np.array([], dtype=np.float32)

    seen_a = set()
    seen_b = set()
    scores = np.zeros(n, dtype=np.float64)

    for d in range(1, n + 1):
        seen_a.add(int(rank_a[d - 1]))
        seen_b.add(int(rank_b[d - 1]))
        overlap = seen_a.intersection(seen_b)
        if not overlap:
            continue

        # Prefix-weighted contribution from RBO truncation.
        w_d = (1.0 - p) * (p ** (d - 1))
        contrib = w_d / float(d)
        for idx in overlap:
            scores[idx] += contrib

    if scores.max() > 0:
        scores = scores / scores.max()

    return scores.astype(np.float32)
