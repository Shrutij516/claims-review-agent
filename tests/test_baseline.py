import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import baseline  # noqa: E402


def test_baseline_confusion_matrix():
    claims = json.loads((ROOT / "data" / "claims.json").read_text())
    truth = {g["claim_id"]: g["is_anomalous"]
             for g in json.loads((ROOT / "data" / "ground_truth.json").read_text())}

    tp = fp = tn = fn = 0
    for c in claims:
        result, _latency = baseline.classify(c["raw_note"])
        flagged = result["decision"] == "Flagged"
        anomalous = truth[result["claim_id"]]
        if flagged and anomalous:
            tp += 1
        elif flagged and not anomalous:
            fp += 1
        elif not flagged and not anomalous:
            tn += 1
        else:
            fn += 1

    assert (tp, fp, tn, fn) == (15, 5, 10, 0)
