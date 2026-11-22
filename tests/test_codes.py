from claims_review.codes import load_tables, validate_codes
from factories import make


def test_tables_load_and_contain_known_codes():
    t = load_tables()
    assert "M23.221" in t.icd10 and "99213" in t.cpt and "J3301" in t.hcpcs and "25" in t.modifiers


def test_valid_cpt_claim_all_codes_valid_with_descriptions():
    checks = validate_codes(make(cpt="99213", modifiers=["25"], icd10=["J06.9"]))
    assert [c.system for c in checks] == ["cpt", "modifier", "icd10"]
    assert all(c.valid and c.description for c in checks)


def test_hcpcs_code_is_recognized():
    c = validate_codes(make(cpt="J3301"))[0]
    assert (c.system, c.valid) == ("hcpcs", True)


def test_unknown_procedure_code_is_invalid():
    c = validate_codes(make(cpt="99216"))[0]
    assert (c.system, c.valid, c.description) == ("cpt_or_hcpcs", False, None)


def test_invalid_modifier_and_invalid_diagnosis():
    checks = validate_codes(make(modifiers=["ZZ"], icd10=["I1O", "I10"]))
    by_code = {c.code: c for c in checks}
    assert by_code["ZZ"].valid is False
    assert by_code["I1O"].valid is False and by_code["I1O"].laterality is None
    assert by_code["I10"].valid is True


def test_laterality_comes_from_the_icd_table():
    checks = {c.code: c.laterality for c in validate_codes(make(icd10=["M23.221", "M23.222", "M17.0", "M23.229", "I10"]))
              if c.system == "icd10"}
    assert checks == {"M23.221": "right", "M23.222": "left", "M17.0": "bilateral",
                      "M23.229": "unspecified", "I10": "none"}


def test_procedure_is_always_first():
    assert validate_codes(make(cpt="99216", modifiers=["25"], icd10=["I10"]))[0].code == "99216"
