"""Classification metrics with zero-division safe values (None, never 0) and errors counted as wrong."""


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_metrics(rows: list[tuple[bool, str]]) -> dict:
    """rows are (is_anomalous, predicted) with predicted in Flagged, Clear or Error.

    An Error counts as wrong: on an anomalous claim it is a false negative,
    on a clean claim a false positive. Errors are also reported separately.
    """
    tp = fp = tn = fn = err_anom = err_clean = 0
    for is_anomalous, predicted in rows:
        if predicted not in ("Flagged", "Clear", "Error"):
            raise ValueError(f"unknown prediction: {predicted!r}")
        if predicted == "Error":
            if is_anomalous:
                fn += 1
                err_anom += 1
            else:
                fp += 1
                err_clean += 1
        elif predicted == "Flagged":
            tp, fp = (tp + 1, fp) if is_anomalous else (tp, fp + 1)
        else:
            fn, tn = (fn + 1, tn) if is_anomalous else (fn, tn + 1)
    n = len(rows)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "errors": err_anom + err_clean,
        "errors_on_anomalous": err_anom,
        "errors_on_clean": err_clean,
        "accuracy": _ratio(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "fpr": _ratio(fp, fp + tn),
        "f1": f1,
    }


def per_flag_counts(expected: dict[str, set[str]], predicted: dict[str, set[str] | None]) -> dict[str, dict]:
    """Per flag: hit and total counts over claims that should raise it, plus false alarms.

    expected maps claim id to its expected flags (clean claims map to an empty set).
    predicted maps claim id to raised flags, or None when processing failed;
    a failed claim is a miss for every flag it should have raised.
    """
    flags = set()
    for fl in expected.values():
        flags |= fl
    for fl in predicted.values():
        flags |= fl or set()
    out = {}
    for flag in sorted(flags):
        should = [c for c, fl in expected.items() if flag in fl]
        hit = sum(1 for c in should if predicted.get(c) is not None and flag in predicted[c])
        false_alarm = sum(1 for c, fl in predicted.items()
                          if fl is not None and flag in fl and flag not in expected.get(c, set()))
        out[flag] = {"hit": hit, "total": len(should), "miss": len(should) - hit, "false_alarm": false_alarm}
    return out


def format_ratio(hit: int, total: int) -> str:
    """Format a count pair like '1 of 1'."""
    return f"{hit} of {total}"
