"""Session 11 - what has to be true before a model leaves the notebook.

Three small checks that catch the failures we actually see in deployed student and
ministry projects: an environment nobody can rebuild, a silent schema change, and a
distribution that has drifted away from the training set.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math


def environment_fingerprint(requirements: list[str]) -> str:
    """A short, stable hash of a pinned dependency list.

    Record it beside every result you report. If two runs disagree and the fingerprints
    differ, the environment is the first suspect - and an unpinned requirement makes the
    fingerprint meaningless, which is why this refuses one.

    Args:
        requirements: lines such as `torch==2.4.1`. Blanks and `#` comments are ignored.

    Raises:
        ValueError: if any requirement is not pinned with `==`.
    """
    pinned = []
    for line in requirements:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"requirement is not pinned: {line!r}")
        pinned.append(line.lower())
    if not pinned:
        raise ValueError("no requirements given")
    joined = "\n".join(sorted(pinned)).encode()
    return hashlib.sha256(joined).hexdigest()[:12]


def check_schema(row: dict, expected: dict[str, type]) -> list[str]:
    """Compare one input record against the schema the model was trained on.

    Returns:
        A list of human-readable problems - missing fields, unexpected extra fields, and
        wrong types. Empty means the row is servable. Returning problems rather than
        raising lets a caller log every fault in a batch instead of only the first.
    """
    problems = []
    for field, kind in sorted(expected.items()):
        if field not in row:
            problems.append(f"missing field {field!r}")
        elif not isinstance(row[field], kind):
            got = type(row[field]).__name__
            problems.append(f"field {field!r} should be {kind.__name__}, got {got}")
    for field in sorted(set(row) - set(expected)):
        problems.append(f"unexpected field {field!r}")
    return problems


def population_stability_index(baseline: list[float], live: list[float],
                               n_bins: int = 10) -> float:
    """PSI between a training sample and a live sample of one feature.

    Bin edges come from the BASELINE quantiles, because the question is where the live
    data sits relative to what the model was trained on. The usual reading: below 0.1 is
    stable, 0.1-0.25 warrants a look, above 0.25 means retrain.

    Empty bins are floored at a small epsilon so a single vanished bin gives a large but
    finite score rather than infinity.
    """
    if not baseline or not live:
        raise ValueError("both samples must be non-empty")
    ordered = sorted(baseline)
    edges = [ordered[min(int(i * len(ordered) / n_bins), len(ordered) - 1)]
             for i in range(1, n_bins)]

    def spread(sample):
        counts = [0] * n_bins
        for value in sample:
            index = 0
            while index < len(edges) and value >= edges[index]:
                index += 1
            counts[index] += 1
        eps = 1e-6
        return [max(c / len(sample), eps) for c in counts]

    return sum((live_share - base_share) * math.log(live_share / base_share)
               for base_share, live_share in zip(spread(baseline), spread(live)))


def model_card(name: str, version: str, metrics: dict, limitations: list[str]) -> str:
    """Render a minimal model card as JSON.

    Not a substitute for the full template - the point is that the card is generated from
    the same run that produced the metrics, so it cannot drift away from them.
    """
    return json.dumps({
        "name": name,
        "version": version,
        "metrics": metrics,
        "limitations": limitations,
    }, indent=2, sort_keys=True)
