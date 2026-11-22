import math
from datetime import date

import pytest

from claims_review.parsing import ClaimParseError, parse_claim
from factories import raw


def test_valid_claim_parses_and_normalizes_codes():
    p = parse_claim(raw(cpt=" 99213 ", modifiers=["rt", " 25"], icd10=[" j06.9", "i10 "]))
    assert p.cpt == "99213"
    assert p.modifiers == ("RT", "25")
    assert p.icd10 == ("J06.9", "I10")
    assert p.date_of_service == date(2026, 2, 10)
    assert p.units == 1 and p.billed_amount == 100.0
    assert p.provider_name == "Dr. Test One" and p.provider_specialty == "Family Medicine"


def test_integer_billed_amount_becomes_float():
    assert isinstance(parse_claim(raw(billed_amount=100)).billed_amount, float)


def test_empty_clinical_note_is_allowed():
    assert parse_claim(raw(clinical_note="")).clinical_note == ""


def test_not_a_dict_is_an_error():
    with pytest.raises(ClaimParseError, match="must be a dict"):
        parse_claim(["not", "a", "dict"])


def test_missing_claim_id_message_says_so():
    r = raw()
    del r["claim_id"]
    with pytest.raises(ClaimParseError, match=r"\(no claim_id\).*claim_id"):
        parse_claim(r)


@pytest.mark.parametrize("override, field", [
    ({"patient_ref": ""}, "patient_ref"),
    ({"patient_ref": None}, "patient_ref"),
    ({"patient_age_band": 5}, "patient_age_band"),
    ({"patient_sex": " "}, "patient_sex"),
    ({"place_of_service": 11}, "place_of_service"),
    ({"provider": "Dr. Test"}, "provider"),
    ({"provider": {"name": "Dr. Test"}}, "specialty"),
    ({"provider": {"specialty": "Family Medicine"}}, "name"),
    ({"date_of_service": "2026-13-01"}, "date_of_service"),
    ({"date_of_service": "20260210"}, "date_of_service"),
    ({"date_of_service": 20260210}, "date_of_service"),
    ({"cpt": ""}, "cpt"),
    ({"cpt": 99213}, "cpt"),
    ({"modifiers": "25"}, "modifiers"),
    ({"modifiers": [25]}, "modifier"),
    ({"icd10": []}, "icd10"),
    ({"icd10": "J06.9"}, "icd10"),
    ({"icd10": [None]}, "icd10"),
    ({"units": 1.0}, "units"),
    ({"units": True}, "units"),
    ({"units": 0}, "units"),
    ({"units": "1"}, "units"),
    ({"billed_amount": "100"}, "billed_amount"),
    ({"billed_amount": True}, "billed_amount"),
    ({"billed_amount": -1}, "billed_amount"),
    ({"billed_amount": 0}, "billed_amount"),
    ({"billed_amount": 0.0}, "billed_amount"),
    ({"billed_amount": -math.inf}, "billed_amount"),
    ({"units": -3}, "units"),
    ({"billed_amount": math.nan}, "billed_amount"),
    ({"billed_amount": math.inf}, "billed_amount"),
    ({"clinical_note": None}, "clinical_note"),
])
def test_bad_values_raise_with_claim_id_and_field_name(override, field):
    with pytest.raises(ClaimParseError) as err:
        parse_claim(raw(**override))
    assert "T-0001" in str(err.value)
    assert field in str(err.value)


def test_smallest_valid_units_and_amount_are_accepted():
    p = parse_claim(raw(units=1, billed_amount=0.01))
    assert p.units == 1 and p.billed_amount == 0.01


def test_missing_optional_looking_fields_are_still_required():
    for key in ["patient_ref", "provider", "date_of_service", "cpt", "modifiers", "icd10", "units",
                "billed_amount", "clinical_note", "place_of_service"]:
        r = raw()
        del r[key]
        with pytest.raises(ClaimParseError):
            parse_claim(r)
