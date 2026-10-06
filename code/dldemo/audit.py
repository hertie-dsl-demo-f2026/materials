"""Session 10 - the numbers a regulator asks for.

Aggregate accuracy hides who the errors fall on. These helpers split a set of predictions
by group and report the disparities the EU AI Act's conformity assessment expects a
provider to have measured.

Standard library only.
"""

from __future__ import annotations


def confusion(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) for 0/1 labels, in that order."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} labels but {len(y_pred)} predictions")
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if pred:
            tp, fp = (tp + 1, fp) if truth else (tp, fp + 1)
        else:
            fn, tn = (fn + 1, tn) if truth else (fn, tn + 1)
    return tp, fp, tn, fn


def group_rates(y_true: list[int], y_pred: list[int], groups: list[str]):
    """Selection rate, TPR and FPR for each group.

    Args:
        y_true: true 0/1 outcomes.
        y_pred: the model's 0/1 decisions - AFTER thresholding, because fairness is a
            property of the decision, not of the score.
        groups: the group label of each row (any hashable string).

    Returns:
        `{group: {"n", "selection_rate", "tpr", "fpr"}}`. A rate whose denominator is
        empty is reported as `None`, not 0.0 - "no positives in this group" and "never
        catches a positive" are different findings and must not be averaged together.
    """
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("y_true, y_pred and groups must be the same length")
    buckets: dict[str, tuple[list[int], list[int]]] = {}
    for truth, pred, group in zip(y_true, y_pred, groups):
        t, p = buckets.setdefault(group, ([], []))
        t.append(truth)
        p.append(pred)

    out = {}
    for group, (t, p) in sorted(buckets.items()):
        tp, fp, tn, fn = confusion(t, p)
        positives, negatives = tp + fn, fp + tn
        out[group] = {
            "n": len(t),
            "selection_rate": sum(p) / len(p) if p else None,
            "tpr": tp / positives if positives else None,
            "fpr": fp / negatives if negatives else None,
        }
    return out


def disparity(rates: dict, metric: str = "selection_rate") -> dict:
    """Compare every group against the best-off group on one metric.

    Returns:
        `{"metric", "best", "worst", "ratio", "difference"}`. `ratio` is the worst group's
        rate over the best group's - the "four-fifths rule" reads it directly, and a value
        below 0.8 is the conventional red flag. Groups whose metric is `None` are skipped;
        `None` comes back throughout if fewer than two groups remain.
    """
    usable = {g: r[metric] for g, r in rates.items() if r.get(metric) is not None}
    if len(usable) < 2:
        return {"metric": metric, "best": None, "worst": None,
                "ratio": None, "difference": None}
    best = max(usable, key=lambda g: usable[g])
    worst = min(usable, key=lambda g: usable[g])
    top, bottom = usable[best], usable[worst]
    return {
        "metric": metric,
        "best": best,
        "worst": worst,
        "ratio": (bottom / top) if top else None,
        "difference": top - bottom,
    }
