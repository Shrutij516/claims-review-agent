import pytest

from claims_review.metrics import compute_metrics, format_ratio, per_flag_counts


def test_basic_confusion_and_metrics():
    rows = [(True, "Flagged"), (True, "Flagged"), (True, "Clear"), (False, "Flagged"), (False, "Clear"), (False, "Clear")]
    m = compute_metrics(rows)
    assert (m["tp"], m["fp"], m["tn"], m["fn"], m["n"]) == (2, 1, 2, 1, 6)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["errors"] == 0


def test_empty_input_is_all_none_not_zero():
    m = compute_metrics([])
    assert m["n"] == 0
    assert all(m[k] is None for k in ("accuracy", "precision", "recall", "fpr", "f1"))


def test_no_anomalies_makes_recall_none():
    m = compute_metrics([(False, "Clear"), (False, "Flagged")])
    assert m["recall"] is None and m["precision"] == 0.0 and m["fpr"] == 0.5
    assert m["f1"] is None


def test_nothing_flagged_makes_precision_none():
    m = compute_metrics([(True, "Clear"), (False, "Clear")])
    assert m["precision"] is None and m["recall"] == 0.0 and m["f1"] is None


def test_no_clean_claims_makes_fpr_none():
    m = compute_metrics([(True, "Flagged")])
    assert m["fpr"] is None and m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_zero_precision_and_recall_gives_none_f1():
    m = compute_metrics([(True, "Clear"), (False, "Flagged")])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] is None


def test_errors_count_as_wrong_and_are_reported_separately():
    m = compute_metrics([(True, "Error"), (False, "Error"), (True, "Flagged"), (False, "Clear")])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["errors"] == 2 and m["errors_on_anomalous"] == 1 and m["errors_on_clean"] == 1
    assert m["accuracy"] == 0.5


def test_error_on_anomalous_claim_lowers_recall():
    assert compute_metrics([(True, "Flagged"), (True, "Error")])["recall"] == 0.5


def test_error_on_clean_claim_lowers_precision_and_raises_fpr():
    m = compute_metrics([(True, "Flagged"), (False, "Error"), (False, "Clear")])
    assert m["precision"] == 0.5 and m["fpr"] == 0.5


def test_unknown_prediction_raises():
    with pytest.raises(ValueError):
        compute_metrics([(True, "Maybe")])


def test_per_flag_counts_hits_misses_and_false_alarms():
    expected = {"c1": {"R_A"}, "c2": {"R_A", "R_B"}, "c3": set(), "c4": {"R_B"}}
    predicted = {"c1": {"R_A"}, "c2": {"R_A"}, "c3": {"R_A"}, "c4": set()}
    out = per_flag_counts(expected, predicted)
    assert out["R_A"] == {"hit": 2, "total": 2, "miss": 0, "false_alarm": 1}
    assert out["R_B"] == {"hit": 0, "total": 2, "miss": 2, "false_alarm": 0}
    assert format_ratio(out["R_A"]["hit"], out["R_A"]["total"]) == "2 of 2"
    assert format_ratio(0, 1) == "0 of 1"


def test_per_flag_counts_treats_errors_as_misses_without_false_alarms():
    out = per_flag_counts({"c1": {"R_A"}, "c2": set()}, {"c1": None, "c2": None})
    assert out["R_A"] == {"hit": 0, "total": 1, "miss": 1, "false_alarm": 0}


def test_per_flag_counts_includes_flags_only_seen_in_predictions():
    out = per_flag_counts({"c1": set()}, {"c1": {"R_X"}})
    assert out["R_X"] == {"hit": 0, "total": 0, "miss": 0, "false_alarm": 1}
