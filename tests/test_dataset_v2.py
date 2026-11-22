import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import generate_claims_v2 as gen  # noqa: E402

V2 = ROOT / "data" / "v2"
CODES = ROOT / "data" / "codes"

CLAIMS = json.loads((V2 / "claims.json").read_text(encoding="utf-8"))
GTS = json.loads((V2 / "ground_truth.json").read_text(encoding="utf-8"))
CATALOG = json.loads((V2 / "flag_catalog.json").read_text(encoding="utf-8"))
TABLES = gen.load_tables()
CRITERIA = gen.load_policy_criteria()
FLAG_IDS = {f["flag_id"] for f in CATALOG["flags"]}
GT_BY_ID = {g["claim_id"]: g for g in GTS}

EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)


def split(name):
    return [g for g in GTS if g["split"] == name]


def test_generator_output_is_byte_equal_to_committed_files():
    claims, gts = gen.generate()
    assert gen.render_json(claims) == (V2 / "claims.json").read_text(encoding="utf-8")
    assert gen.render_json(gts) == (V2 / "ground_truth.json").read_text(encoding="utf-8")


def test_counts_and_anomaly_share():
    assert len(CLAIMS) == 60 and len(GTS) == 60
    dev, test = split("dev"), split("test")
    assert (len(dev), len(test)) == (20, 40)
    assert sum(g["is_anomalous"] for g in dev) == 9
    assert sum(g["is_anomalous"] for g in test) == 18
    for part in (dev, test):
        share = sum(g["is_anomalous"] for g in part) / len(part)
        assert 0.40 <= share <= 0.50
        assert sum(g["category"] == "hard_negative" for g in part) >= 4


def test_unique_ids_and_disjoint_splits():
    ids = [c["claim_id"] for c in CLAIMS]
    assert len(ids) == len(set(ids))
    assert ids == [g["claim_id"] for g in GTS]
    dev_ids = {g["claim_id"] for g in split("dev")}
    test_ids = {g["claim_id"] for g in split("test")}
    assert dev_ids.isdisjoint(test_ids) and dev_ids | test_ids == set(ids)
    patients = {}
    for c in CLAIMS:
        patients.setdefault(GT_BY_ID[c["claim_id"]]["split"], set()).add(c["patient_ref"])
    assert patients["dev"].isdisjoint(patients["test"])


def test_every_ground_truth_flag_is_in_the_catalog():
    for g in GTS:
        assert set(g["expected_flags"]) <= FLAG_IDS, g["claim_id"]
        for ev in g["evidence"]:
            if ev["kind"] == "flag":
                assert ev["id"] in FLAG_IDS, g["claim_id"]


def test_every_flag_type_appears_in_test():
    seen = {f for g in split("test") for f in g["expected_flags"]}
    assert seen == FLAG_IDS


def test_policy_reading_share_of_test_anomalies():
    anomalies = [g for g in split("test") if g["is_anomalous"]]
    reading = [g for g in anomalies if g["needs_policy_reading"]]
    assert len(reading) / len(anomalies) >= 0.30


def test_policy_evidence_points_to_an_existing_criterion():
    for g in GTS:
        for ev in g["evidence"]:
            if ev["kind"] == "policy":
                assert re.fullmatch(r"POL-00\d\.\d+", ev["id"]), (g["claim_id"], ev)
                assert ev["id"] in CRITERIA, (g["claim_id"], ev)
    policy_files = {p.stem for p in (ROOT / "docs" / "policies").glob("POL-*.md")}
    for flag in CATALOG["flags"]:
        assert set(flag["policy_ids"]) <= policy_files, flag["flag_id"]
    for pid in policy_files:
        crit = [k for k in CRITERIA if k.startswith(pid + ".")]
        assert 3 <= len(crit) <= 5, pid


def test_codes_are_in_the_tables_unless_labeled_invalid():
    for c in CLAIMS:
        absent = (c["cpt"] not in TABLES["procedures"]
                  or any(i not in TABLES["icd10"] for i in c["icd10"])
                  or any(m not in TABLES["modifiers"] for m in c["modifiers"]))
        labeled = "R_INVALID_CODE" in GT_BY_ID[c["claim_id"]]["expected_flags"]
        assert absent == labeled, c["claim_id"]
    for g in GTS:
        for ev in g["evidence"]:
            if ev["kind"] == "code" and "R_INVALID_CODE" not in g["expected_flags"]:
                assert ev["id"] in TABLES["procedures"] or ev["id"] in TABLES["icd10"], g["claim_id"]


def test_schema_has_no_extra_fields():
    for c in CLAIMS:
        assert list(c.keys()) == gen.CLAIM_FIELDS
        assert set(c["provider"].keys()) == {"name", "specialty"}
        assert isinstance(c["modifiers"], list) and isinstance(c["icd10"], list)
        assert isinstance(c["units"], int) and isinstance(c["billed_amount"], float)
        assert 2 <= len(re.findall(r"[.!?](?:\s|$)", c["clinical_note"])) <= 4, c["claim_id"]
        assert date(2026, 1, 1) <= date.fromisoformat(c["date_of_service"]) <= date(2026, 3, 31)
    for g in GTS:
        assert list(g.keys()) == gen.GT_FIELDS
        assert all(set(ev.keys()) == {"kind", "id"} for ev in g["evidence"])
        assert all(ev["kind"] in ("policy", "flag", "code") for ev in g["evidence"])
        assert g["split"] in ("dev", "test")
        assert g["category"] in ("clean", "hard_negative", "anomaly")
        assert isinstance(g["needs_policy_reading"], bool)


def test_patient_ref_format():
    assert all(re.fullmatch(r"^P-\d{4}$", c["patient_ref"]) for c in CLAIMS)


def test_rule_flags_match_ground_truth():
    # Re-derives every R_ flag from the catalog parameters.
    gen.verify_labels(CLAIMS, GTS, TABLES, CATALOG["parameters"])


def test_notes_do_not_state_the_label():
    banned = re.compile(r"duplicate|upcod|anomal|fraud|outlier|invalid|violat|flag|R_[A-Z]|P_[A-Z]|POL-\d", re.I)
    for c in CLAIMS:
        assert not banned.search(c["clinical_note"]), c["claim_id"]


def test_test_split_hash_matches_frozen_file():
    frozen = (V2 / "TEST_SPLIT.sha256").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"[0-9a-f]{64}", frozen)
    assert gen.test_split_hash(CLAIMS, GTS) == frozen
    # Independent recomputation, so the helper cannot hide a bug.
    ids = {g["claim_id"] for g in GTS if g["split"] == "test"}
    payload = {
        "claims": sorted((c for c in CLAIMS if c["claim_id"] in ids), key=lambda c: c["claim_id"]),
        "ground_truth": sorted((g for g in GTS if g["claim_id"] in ids), key=lambda g: g["claim_id"]),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == frozen


def _text_files(path):
    if path.is_file():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix in (".md", ".json", ".txt", ".sha256")]


@pytest.mark.parametrize("rel", ["docs/policies", "data", "DECISIONS.md"])
def test_no_em_or_en_dash(rel):
    for path in _text_files(ROOT / rel):
        text = path.read_text(encoding="utf-8")
        assert EM_DASH not in text and EN_DASH not in text, str(path)


@pytest.mark.parametrize("rel", ["data/v2", "data/codes"])
def test_no_escaped_dash_in_new_data(rel):
    # JSON can hide a dash as an escape sequence; the new data must not.
    for path in _text_files(ROOT / rel):
        text = path.read_text(encoding="utf-8").lower()
        assert "\\u2014" not in text and "\\u2013" not in text, str(path)
