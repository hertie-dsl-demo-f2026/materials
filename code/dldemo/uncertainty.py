"""Session 9 - what a network's confidence is worth.

A softmax score is not a probability of being right. These helpers make that concrete on
predictions you already have, with no framework and no training loop: pass in the
probability vectors your model produced and read off how well-calibrated they were.

Standard library only, so it installs and runs anywhere.
"""

from __future__ import annotations

import math


def entropy(probs: list[float], base: float = math.e) -> float:
    """Shannon entropy of one predictive distribution.

    High entropy means the network spread its mass around - it is unsure *which* class.
    This is the cheapest uncertainty signal there is, and the first one to reach for.

    Args:
        probs: one probability vector, summing to about 1.
        base: log base. Pass 2 for bits, or `len(probs)` to normalise onto [0, 1].

    Returns:
        The entropy, in units set by `base`. Zero for a one-hot prediction.
    """
    if not probs:
        raise ValueError("probs is empty")
    total = 0.0
    for p in probs:
        if p > 0.0:
            total -= p * math.log(p, base)
    return total


def confidence(probs: list[float]) -> tuple[int, float]:
    """The predicted class and the mass the network put on it.

    Returns:
        (index, probability) for the arg-max. Ties go to the lowest index.
    """
    if not probs:
        raise ValueError("probs is empty")
    best = max(range(len(probs)), key=lambda i: (probs[i], -i))
    return best, probs[best]


def reliability_bins(probs: list[list[float]], labels: list[int], n_bins: int = 10):
    """Group predictions by confidence, and compare confidence with accuracy.

    This is the table behind a reliability diagram. A well-calibrated model has
    `mean_confidence` approximately equal to `accuracy` in every populated bin; a modern
    over-confident network sits well above the line in the top bins.

    Args:
        probs: one probability vector per example.
        labels: the true class index per example.
        n_bins: how many equal-width confidence bins to use.

    Returns:
        A list of `n_bins` dicts with keys `lower`, `upper`, `count`, `accuracy` and
        `mean_confidence`. Empty bins report zeros.
    """
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} predictions but {len(labels)} labels")
    bins = [{"lower": i / n_bins, "upper": (i + 1) / n_bins,
             "count": 0, "correct": 0, "conf_total": 0.0} for i in range(n_bins)]
    for row, truth in zip(probs, labels):
        predicted, conf = confidence(row)
        # conf is in (0, 1]; the final bin owns its right edge.
        index = min(int(conf * n_bins), n_bins - 1)
        b = bins[index]
        b["count"] += 1
        b["conf_total"] += conf
        b["correct"] += int(predicted == truth)
    out = []
    for b in bins:
        n = b["count"]
        out.append({
            "lower": b["lower"], "upper": b["upper"], "count": n,
            "accuracy": b["correct"] / n if n else 0.0,
            "mean_confidence": b["conf_total"] / n if n else 0.0,
        })
    return out


def expected_calibration_error(probs: list[list[float]], labels: list[int],
                               n_bins: int = 10) -> float:
    """The single number summarising `reliability_bins`: the ECE.

    A count-weighted mean of |accuracy - confidence| across bins. 0.0 is perfect
    calibration. Report it beside accuracy whenever a model's output feeds a decision.
    """
    rows = reliability_bins(probs, labels, n_bins)
    total = sum(r["count"] for r in rows)
    if not total:
        return 0.0
    return sum(r["count"] * abs(r["accuracy"] - r["mean_confidence"]) for r in rows) / total
