"""
Seeded generator for dataset v2 (seed 42). Plain JSON in, plain JSON out.

Reads:   data/codes/*.json, data/v2/flag_catalog.json, docs/policies/*.md
Writes:  data/v2/claims.json, data/v2/ground_truth.json
         data/v2/TEST_SPLIT.sha256 (only with --freeze)

Everything here is SYNTHETIC. Codes come from data/codes (not yet verified
against official files). Patients, providers, notes and amounts are invented.

Design rules:
  - Deterministic flags (names starting R_) are re-derived from the catalog
    parameters and must equal the ground truth, or generation fails.
  - Flags starting P_ need a reader (clinical note vs policy text). They are
    set by construction in the note templates.
  - Claim ids and patient refs are assigned after a seeded shuffle so ids do
    not reveal the label or the split.
  - The test split is frozen by hash. The hash file is only written with
    --freeze, so editing test data without freezing makes the hash test fail.
"""
import argparse
import hashlib
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_V2 = ROOT / "data" / "v2"
CODES_DIR = ROOT / "data" / "codes"
POLICY_DIR = ROOT / "docs" / "policies"

SEED = 42
START = date(2026, 1, 1)
DAYS_IN_RANGE = 90  # 2026-01-01 to 2026-03-31

CLAIM_FIELDS = [
    "claim_id", "patient_ref", "patient_age_band", "patient_sex",
    "date_of_service", "provider", "place_of_service", "cpt", "modifiers",
    "icd10", "units", "billed_amount", "clinical_note",
]
GT_FIELDS = [
    "claim_id", "split", "category", "is_anomalous", "expected_flags",
    "evidence", "needs_policy_reading", "notes",
]

PROVIDERS = {
    "fm": {"name": "Dr. Imogen Hartwell", "specialty": "Family Medicine"},
    "im": {"name": "Dr. Dev Raghunathan", "specialty": "Internal Medicine"},
    "or1": {"name": "Dr. Calista Moreau", "specialty": "Orthopedics"},
    "or2": {"name": "Dr. Bram Oduya", "specialty": "Orthopedics"},
    "ca": {"name": "Dr. Wren Lindqvist", "specialty": "Cardiology"},
    "pt": {"name": "Dr. Nadia Castellanos", "specialty": "Physical Medicine and Rehabilitation"},
}
AGE_BANDS = ["18-39", "40-49", "50-59", "60-69", "70+"]
POS_BY_CPT = {"29881": "22", "73721": "22"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_tables():
    icd = json.loads((CODES_DIR / "icd10.json").read_text())
    cpt = json.loads((CODES_DIR / "cpt.json").read_text())
    hcpcs = json.loads((CODES_DIR / "hcpcs.json").read_text())
    return {
        "icd10": {c["code"]: c for c in icd["codes"]},
        "procedures": {c["code"] for c in cpt["codes"]} | {c["code"] for c in hcpcs["codes"]},
        "modifiers": {c["code"] for c in cpt["modifiers"]},
    }


def load_catalog():
    return json.loads((DATA_V2 / "flag_catalog.json").read_text())


CRITERION_RE = re.compile(r"^(POL-\d{3}\.\d+)\s+(.*)$", re.M)


def load_policy_criteria():
    """Return {criterion_id: text} parsed from docs/policies/POL-*.md."""
    out = {}
    for path in sorted(POLICY_DIR.glob("POL-*.md")):
        for m in CRITERION_RE.finditer(path.read_text()):
            out[m.group(1)] = m.group(2).strip()
    return out


# --------------------------------------------------------------------------
# Reference rule engine (used to verify labels, not a product baseline)
# --------------------------------------------------------------------------
def derive_rule_flags(claim, claims, tables, params):
    """Deterministic R_ flags for one claim, using only catalog parameters."""
    flags = set()
    cpt = claim["cpt"]
    icds = claim["icd10"]
    mods = claim["modifiers"]
    cpt_ok = cpt in tables["procedures"]
    valid_icds = [i for i in icds if i in tables["icd10"]]

    if (not cpt_ok) or len(valid_icds) != len(icds) or any(m not in tables["modifiers"] for m in mods):
        flags.add("R_INVALID_CODE")
    if not cpt_ok:
        return flags

    band = params["price_bands_per_unit"].get(cpt)
    if band:
        per_unit = claim["billed_amount"] / claim["units"]
        if per_unit > params["price_outlier_multiplier"] * band[1]:
            flags.add("R_PRICE_OUTLIER")

    limit = params["units_per_day_limit"].get(cpt)
    if limit is not None and claim["units"] > limit:
        flags.add("R_UNITS_EXCEEDED")

    if cpt in params["laterality_required_codes"]:
        has_mod = any(m in ("RT", "LT") for m in mods)
        has_icd = any(tables["icd10"][i]["laterality"] in ("right", "left", "bilateral") for i in valid_icds)
        if not (has_mod or has_icd):
            flags.add("R_LATERALITY_MISSING")

    prefixes = params["dx_support_prefixes"].get(cpt)
    if prefixes and valid_icds and not any(i.startswith(p) for i in valid_icds for p in prefixes):
        flags.add("R_DX_NOT_SUPPORTING")

    if cpt in params["em_min_tier"]:
        tiers = [params["icd_tier"][i] for i in valid_icds if i in params["icd_tier"]]
        if tiers and max(tiers) < params["em_min_tier"][cpt]:
            flags.add("R_EM_LEVEL_VS_DX")

    me = (claim["date_of_service"], claim["claim_id"])
    d_me = date.fromisoformat(claim["date_of_service"])
    for other in claims:
        if other["claim_id"] == claim["claim_id"]:
            continue
        same_patient = other["patient_ref"] == claim["patient_ref"]
        if not same_patient:
            continue
        same_provider = other["provider"]["name"] == claim["provider"]["name"]
        if (other["cpt"] == cpt and same_provider
                and (other["date_of_service"], other["claim_id"]) < me
                and (d_me - date.fromisoformat(other["date_of_service"])).days <= params["duplicate_window_days"]):
            flags.add("R_DUPLICATE_CLAIM")
        if (cpt in params["mod25_em_codes"] and "25" not in mods and same_provider
                and other["date_of_service"] == claim["date_of_service"]
                and other["cpt"] in params["mod25_procedure_codes"]):
            flags.add("R_MOD25_MISSING")
    return flags


# --------------------------------------------------------------------------
# Clinical note templates (SYNTHETIC; none states the label)
# --------------------------------------------------------------------------
def pick(r, options):
    return options[r.randrange(len(options))]


def n_uri(r):
    d = r.randint(2, 6)
    c = pick(r, [f"sore throat and congestion for {d} days",
                 f"cough and nasal congestion for {d} days",
                 f"runny nose and mild sore throat for {d} days"])
    e = pick(r, ["Afebrile with clear lungs and a mildly red throat.",
                 "Vitals are normal and the lungs are clear.",
                 "No fever, and the ears and chest are unremarkable."])
    p = pick(r, ["Supportive care and fluids advised.",
                 "Advised rest, fluids and over the counter symptom relief."])
    return f"Established patient seen for {c}. {e} {p}"


def n_htn(r):
    s, d, m = r.randint(126, 142), r.randint(76, 88), pick(r, [3, 6])
    return (f"Established patient seen for hypertension follow up. Home readings average {s}/{d} "
            f"with no chest pain or headaches. Current medication continued and a recheck is planned in {m} months.")


def n_dm(r):
    a = round(r.uniform(6.6, 7.9), 1)
    return (f"Established patient seen for diabetes follow up. A1c is {a} on current therapy with no episodes "
            f"of low blood sugar. Diet and activity counseling provided.")


def n_asthma(r):
    n = r.randint(0, 3)
    return (f"Established patient seen for asthma follow up. Symptoms are well controlled with {n} rescue inhaler "
            f"uses in the past month. Inhaler technique reviewed and the controller medication continued.")


def n_lipid(r):
    m = r.randint(1, 3)
    return (f"Established patient seen to review cholesterol management. Fasting lipid panel from {m} months ago "
            f"reviewed with the patient. Diet changes discussed and the statin continued.")


def n_wellness(r):
    return pick(r, [
        "Annual wellness visit with no new complaints. Vitals and physical exam are normal and screening labs were reviewed. Counseled on diet and exercise.",
        "Routine yearly exam with no concerns raised. Exam and vitals are normal and preventive screening was reviewed. Healthy lifestyle counseling provided.",
    ])


def n_back(r):
    d = r.randint(3, 9)
    return (f"Low back pain for {d} days after lifting, with no leg weakness or numbness. Exam shows paraspinal "
            f"tenderness and normal strength. Stretching and ibuprofen advised.")


def n_hf(r):
    d, w = r.randint(4, 9), r.randint(3, 7)
    return (f"Established patient with heart failure reports worsening shortness of breath and ankle swelling over {d} days. "
            f"Exam shows crackles at both lung bases and 2+ leg edema, with weight up {w} pounds. "
            f"Diuretic dose increased and extended counseling on fluid restriction provided, with labs ordered.")


def n_ckd(r):
    e = r.randint(15, 28)
    return (f"Established patient with stage 4 chronic kidney disease seen for lab review. eGFR is {e} with a rising potassium, "
            f"and medications were adjusted for kidney function. Extended counseling on diet and kidney replacement planning provided.")


def n_shoulder_em(r):
    w = r.randint(3, 8)
    return (f"Established patient with right shoulder pain with overhead reaching for {w} weeks. Exam shows tenderness over "
            f"the joint and a painful arc. A corticosteroid injection was performed at this visit.")


def n_asthma_em25(r):
    return (f"Separate evaluation today of worsening wheeze for {r.randint(2, 5)} days, with a full respiratory exam and review of "
            f"controller medication. An injection was given in clinic during the same visit.")


def n_proc_20610(r):
    return ("Right shoulder joint injected with corticosteroid after consent and skin preparation. "
            "Patient tolerated the procedure and had no immediate complications.")


def n_proc_96372(r):
    return (f"Intramuscular injection given in clinic for wheeze. Patient observed for {pick(r, [15, 20, 30])} minutes with improvement.")


def n_ecg_htn(r):
    return (f"ECG obtained for hypertension with occasional palpitations. Sinus rhythm at {r.randint(62, 88)} beats per minute "
            f"with no acute changes.")


def n_ecg_cp(r):
    return "ECG obtained for chest discomfort. Normal sinus rhythm with no ST segment changes."


def n_cmp(r):
    return ("Comprehensive metabolic panel ordered to monitor diabetes and kidney function. "
            "Patient was fasting and reports no new symptoms.")


def n_blood_draw(r):
    return "Blood draw for fasting labs related to diabetes follow up. No complications."


def n_cxr_uri(r):
    return (f"Chest x-ray with two views obtained for cough and low grade fever for {r.randint(4, 10)} days. "
            f"No focal consolidation seen.")


def n_cxr_back(r):
    return "Chest x-ray with two views obtained during a routine visit. Patient also reports ongoing low back pain."


def n_pt(r, minutes):
    v, s = r.randint(2, 9), r.randint(6, 11)
    return (f"Supervised therapeutic exercise for knee strengthening, visit {v} of 12 in the current episode. "
            f"Functional goal is to climb a flight of stairs without pain and the patient can now manage {s} steps. "
            f"Session lasted {minutes} minutes.")


def n_inj_triam(r, dose):
    return (f"Triamcinolone {dose} mg injected for a flare of right shoulder tendinopathy. "
            f"Patient tolerated the injection without complication.")


def n_mri_no_therapy(r, variant):
    w = r.randint(2, 4)
    if variant == 0:
        return (f"Established patient with knee pain for about {w} weeks after increased walking. No physical therapy, bracing "
                f"or medication has been tried to date. Exam shows mild joint line tenderness with stable ligaments, "
                f"and a knee MRI was requested.")
    if variant == 1:
        m = r.randint(1, w - 1)
        return (f"Patient reports knee pain for {w} weeks. Home exercises only for the past {m} weeks, with no formal therapy "
                f"and no anti-inflammatory medication. There is no locking or giving way, and a knee MRI was ordered.")
    return (f"Recent onset knee pain, started {w} weeks ago. Conservative treatment has not yet been started. "
            f"Exam is unremarkable apart from a small effusion, and an MRI was ordered at the patient's request.")


def n_mri_with_therapy(r, kind):
    w = r.randint(10, 14)
    if kind == "supervised":
        x = r.randint(6, 8)
        return (f"Knee pain for {w} weeks despite {x} weeks of supervised physical therapy and daily anti-inflammatory medication. "
                f"Exam shows joint line tenderness. MRI was ordered to evaluate for internal derangement.")
    return (f"Knee pain for {w} weeks that has not improved with 8 weeks of a home exercise program and a knee brace. "
            f"Exam shows mild swelling and joint line tenderness. MRI was ordered to evaluate further.")


def n_mri_redflag(r):
    d = r.randint(2, 5)
    return (f"Twisting injury {d} days ago with a locked knee that cannot fully extend. No therapy was tried given the acute "
            f"presentation. MRI was ordered for a suspected displaced meniscal tear.")


def n_scope_tear(r, comp, side):
    w = r.randint(6, 12)
    where = f" of the {side} knee" if side else ""
    return (f"MRI shows a posterior horn {comp} meniscus tear{where} with mechanical catching, after {w} weeks of conservative care "
            f"without relief. Arthroscopic partial meniscectomy was performed and the patient tolerated it well.")


def n_scope_degen(r, variant):
    k = r.randint(1, 3)
    if variant == 0:
        return ("MRI shows diffuse cartilage loss and joint space narrowing consistent with degenerative joint disease. "
                "No discrete meniscal tear was identified. Arthroscopic debridement and partial meniscectomy performed for "
                "chronic activity related pain.")
    if variant == 1:
        return ("Long standing knee pain in a patient with advanced degenerative changes on weight bearing films. "
                "Imaging does not show a focal tear. Partial meniscectomy performed to address ongoing pain.")
    return (f"Imaging demonstrates osteophytes, joint space narrowing and degenerative fraying only, with no tear documented. "
            f"Arthroscopy with partial meniscectomy done for persistent pain unresponsive to {k} injections.")


def n_scope_tear_with_oa(r):
    return ("MRI confirms a posterior horn lateral meniscus tear with locking episodes. Mild osteoarthritis is also present "
            "but is not the reason for surgery. Arthroscopic partial meniscectomy performed.")


def n_scope_neutral(r):
    return "Right knee pain and swelling. Arthroscopy was performed under regional anesthesia and tolerated well."


# --------------------------------------------------------------------------
# Scenario specs
# --------------------------------------------------------------------------
def F(flag):
    return {"kind": "flag", "id": flag}


def P(crit):
    return {"kind": "policy", "id": crit}


def C(code):
    return {"kind": "code", "id": code}


def S(split, category, cpt, icd10, provider, note, why, *, units=1, price="mid",
      modifiers=(), flags=(), evidence=(), reading=False, pkey=None, offset=0):
    return {
        "split": split, "category": category, "cpt": cpt, "icd10": list(icd10),
        "provider": provider, "note": note, "why": why, "units": units,
        "price": price, "modifiers": list(modifiers), "flags": list(flags),
        "evidence": list(evidence), "reading": reading, "pkey": pkey, "offset": offset,
    }


def build_specs(r):
    s = []
    add = s.append

    # ------------------------------ DEV (20) ------------------------------
    add(S("dev", "anomaly", "73721", ["M25.562"], "or1", n_mri_no_therapy(r, 0),
          "Knee MRI with a few weeks of symptoms and no conservative therapy or red flag documented in the note.",
          flags=["P_NO_CONSERVATIVE_THERAPY"], evidence=[F("P_NO_CONSERVATIVE_THERAPY"), P("POL-003.1"), P("POL-003.2")], reading=True))
    add(S("dev", "anomaly", "29881", ["M23.222"], "or2", n_scope_degen(r, 0),
          "Meniscus code on the claim, but the note describes only degenerative findings and no tear.",
          flags=["P_DEGENERATIVE_ONLY"], evidence=[F("P_DEGENERATIVE_ONLY"), P("POL-001.1"), C("M23.222")], reading=True))
    add(S("dev", "anomaly", "99213", ["E78.5"], "im", n_lipid(r),
          "Billed amount per unit is far above the top of the synthetic band.",
          price="out", flags=["R_PRICE_OUTLIER"], evidence=[F("R_PRICE_OUTLIER"), C("99213")]))
    add(S("dev", "anomaly", "97110", ["M54.50"], "pt", n_pt(r, 90),
          "Six units of 97110 on one day, above the limit of four.",
          units=6, flags=["R_UNITS_EXCEEDED"], evidence=[F("R_UNITS_EXCEEDED"), P("POL-005.1"), P("POL-005.2")]))
    add(S("dev", "anomaly", "99214", ["M54.50"], "fm", n_back(r),
          "Level 4 visit with only a tier 1 diagnosis.",
          flags=["R_EM_LEVEL_VS_DX"], evidence=[F("R_EM_LEVEL_VS_DX"), P("POL-002.2"), C("M54.50")]))
    add(S("dev", "anomaly", "73721", ["M25.569"], "or1", n_mri_with_therapy(r, "supervised"),
          "Knee MRI with no side given by a modifier or by the diagnosis code.",
          flags=["R_LATERALITY_MISSING"], evidence=[F("R_LATERALITY_MISSING"), P("POL-003.4"), C("M25.569")]))
    add(S("dev", "anomaly", "29881", ["M17.11"], "or2", n_scope_neutral(r),
          "Knee arthroscopy billed with an osteoarthritis diagnosis only.",
          flags=["R_DX_NOT_SUPPORTING"], evidence=[F("R_DX_NOT_SUPPORTING"), P("POL-001.1"), C("M17.11")]))
    dup_note = n_htn(r)
    add(S("dev", "anomaly", "99213", ["I10"], "fm", dup_note,
          "Same patient, code and provider as another claim 6 days earlier.",
          flags=["R_DUPLICATE_CLAIM"], evidence=[F("R_DUPLICATE_CLAIM"), P("POL-004.1")], pkey="dev_dup", offset=6))
    add(S("dev", "clean", "99213", ["I10"], "fm", dup_note,
          "Routine hypertension visit. It is the earlier claim of a pair.", pkey="dev_dup", offset=0))
    add(S("dev", "anomaly", "99213", ["I10", "I1O"], "im", n_htn(r),
          "Second diagnosis code has a letter O in place of a zero and is not in the code tables.",
          flags=["R_INVALID_CODE"], evidence=[F("R_INVALID_CODE"), C("I1O")]))
    add(S("dev", "hard_negative", "99215", ["I50.9"], "im", n_hf(r),
          "Looks like an expensive level 5 visit, but heart failure supports it and the price is inside the band.",
          price="top", evidence=[P("POL-002.3"), C("I50.9")]))
    add(S("dev", "clean", "99214", ["I10"], "im", n_htn(r),
          "Routine visit. It is the earlier claim of a pair.", pkey="dev_gap", offset=0))
    add(S("dev", "hard_negative", "99214", ["I10"], "im", n_htn(r),
          "Same patient, code and provider as another claim, but 21 days later, outside the 14 day window.",
          evidence=[P("POL-004.3")], pkey="dev_gap", offset=21))
    add(S("dev", "hard_negative", "29881", ["M23.222"], "or2", n_scope_tear(r, "medial", "left"),
          "No RT or LT modifier, but the diagnosis code itself specifies the left knee.",
          evidence=[P("POL-001.2"), C("M23.222")], reading=True))
    add(S("dev", "hard_negative", "73721", ["M25.561"], "or1", n_mri_redflag(r),
          "No conservative therapy, but the note documents a locked knee, which is an allowed exception.",
          evidence=[P("POL-003.3")], reading=True))
    add(S("dev", "clean", "99213", ["J06.9"], "fm", n_uri(r), "Routine minor illness visit."))
    add(S("dev", "clean", "93000", ["I10"], "ca", n_ecg_htn(r), "Routine ECG for hypertension."))
    add(S("dev", "clean", "80053", ["E11.9"], "im", n_cmp(r), "Routine metabolic panel for diabetes."))
    add(S("dev", "clean", "97110", ["M25.562"], "pt", n_pt(r, 45), "Three units, within the limit, with goals documented.",
          units=3, evidence=[P("POL-008.2")]))
    add(S("dev", "clean", "71046", ["J06.9"], "fm", n_cxr_uri(r), "Chest x-ray for a respiratory complaint."))

    # ------------------------------ TEST (40) -----------------------------
    # Anomalies (18)
    add(S("test", "anomaly", "99216", ["J06.9"], "fm", n_uri(r),
          "Procedure code 99216 is not in the code tables.", price="unknown",
          flags=["R_INVALID_CODE"], evidence=[F("R_INVALID_CODE"), C("99216")]))
    add(S("test", "anomaly", "99213", ["E11.9", "M17.99"], "im", n_dm(r),
          "Second diagnosis code M17.99 is not in the code tables.",
          flags=["R_INVALID_CODE"], evidence=[F("R_INVALID_CODE"), C("M17.99")]))
    t_dup_note = n_htn(r)
    add(S("test", "anomaly", "99213", ["I10"], "im", t_dup_note,
          "Same patient, code and provider as another claim 8 days earlier.",
          flags=["R_DUPLICATE_CLAIM"], evidence=[F("R_DUPLICATE_CLAIM"), P("POL-004.1")], pkey="test_dup", offset=8))
    add(S("test", "anomaly", "97110", ["M25.561"], "pt", n_pt(r, 90),
          "Six units of 97110 on one day, above the limit of four.",
          units=6, flags=["R_UNITS_EXCEEDED"], evidence=[F("R_UNITS_EXCEEDED"), P("POL-005.1"), P("POL-005.2")]))
    add(S("test", "anomaly", "J3301", ["M25.511"], "or1", n_inj_triam(r, 40),
          "Twelve units of J3301 billed for a documented 40 mg dose, above the limit of eight.",
          units=12, flags=["R_UNITS_EXCEEDED"], evidence=[F("R_UNITS_EXCEEDED"), P("POL-007.2"), P("POL-007.3")]))
    add(S("test", "anomaly", "80053", ["E11.9"], "im", n_cmp(r),
          "Billed amount far above the top of the synthetic band.",
          price="out", flags=["R_PRICE_OUTLIER"], evidence=[F("R_PRICE_OUTLIER"), C("80053")]))
    add(S("test", "anomaly", "93000", ["I10"], "ca", n_ecg_htn(r),
          "Billed amount far above the top of the synthetic band.",
          price="out", flags=["R_PRICE_OUTLIER"], evidence=[F("R_PRICE_OUTLIER"), C("93000")]))
    add(S("test", "anomaly", "29881", ["M23.229"], "or1", n_scope_tear(r, "medial", ""),
          "Knee arthroscopy with no modifier and a diagnosis code that does not give the side.",
          flags=["R_LATERALITY_MISSING"], evidence=[F("R_LATERALITY_MISSING"), P("POL-001.2"), C("M23.229")]))
    add(S("test", "anomaly", "71046", ["M54.50"], "fm", n_cxr_back(r),
          "Chest x-ray billed with a back pain diagnosis only.",
          flags=["R_DX_NOT_SUPPORTING"], evidence=[F("R_DX_NOT_SUPPORTING"), C("M54.50")]))
    add(S("test", "anomaly", "99213", ["M25.511"], "or1", n_shoulder_em(r),
          "Office visit and a joint injection billed on the same day for the same patient and provider, with no modifier 25 on the visit.",
          flags=["R_MOD25_MISSING"], evidence=[F("R_MOD25_MISSING"), P("POL-006.1")], pkey="test_mod25_a"))
    add(S("test", "anomaly", "99215", ["Z00.00"], "fm", n_wellness(r),
          "Level 5 visit with a routine exam diagnosis.",
          flags=["R_EM_LEVEL_VS_DX"], evidence=[F("R_EM_LEVEL_VS_DX"), P("POL-002.2"), P("POL-002.4")]))
    add(S("test", "anomaly", "99215", ["J06.9"], "im", n_uri(r),
          "Level 5 visit with a minor illness diagnosis.",
          flags=["R_EM_LEVEL_VS_DX"], evidence=[F("R_EM_LEVEL_VS_DX"), P("POL-002.2"), P("POL-002.4")]))
    add(S("test", "anomaly", "73721", ["M25.561"], "or2", n_mri_no_therapy(r, 0),
          "Knee MRI with a few weeks of symptoms and no therapy tried.",
          flags=["P_NO_CONSERVATIVE_THERAPY"], evidence=[F("P_NO_CONSERVATIVE_THERAPY"), P("POL-003.1"), P("POL-003.2")], reading=True))
    add(S("test", "anomaly", "73721", ["M25.562"], "or1", n_mri_no_therapy(r, 1),
          "Knee MRI with only a short home exercise period documented.",
          flags=["P_NO_CONSERVATIVE_THERAPY"], evidence=[F("P_NO_CONSERVATIVE_THERAPY"), P("POL-003.1"), P("POL-003.2")], reading=True))
    add(S("test", "anomaly", "73721", ["M25.561"], "or2", n_mri_no_therapy(r, 2),
          "Knee MRI for recent onset pain with conservative treatment not yet started.",
          flags=["P_NO_CONSERVATIVE_THERAPY"], evidence=[F("P_NO_CONSERVATIVE_THERAPY"), P("POL-003.1"), P("POL-003.2")], reading=True))
    add(S("test", "anomaly", "29881", ["M23.221"], "or1", n_scope_degen(r, 1),
          "Meniscus code on the claim, but the note describes only degenerative changes and no tear.",
          flags=["P_DEGENERATIVE_ONLY"], evidence=[F("P_DEGENERATIVE_ONLY"), P("POL-001.1"), C("M23.221")], reading=True))
    add(S("test", "anomaly", "29881", ["M23.251"], "or2", n_scope_degen(r, 2),
          "Meniscus code on the claim, but imaging in the note shows degenerative fraying only.",
          flags=["P_DEGENERATIVE_ONLY"], evidence=[F("P_DEGENERATIVE_ONLY"), P("POL-001.1"), C("M23.251")], reading=True))
    add(S("test", "anomaly", "29881", ["M23.252"], "or1", n_scope_degen(r, 0),
          "Meniscus code on the claim, but the note says no discrete tear was found.",
          flags=["P_DEGENERATIVE_ONLY"], evidence=[F("P_DEGENERATIVE_ONLY"), P("POL-001.1"), C("M23.252")], reading=True))

    # Hard negatives (9)
    add(S("test", "hard_negative", "99215", ["I50.9"], "ca", n_hf(r),
          "Level 5 visit at the top of the price band, supported by a heart failure diagnosis.",
          price="top", evidence=[P("POL-002.3"), C("I50.9")]))
    add(S("test", "hard_negative", "99214", ["E11.9"], "im", n_dm(r),
          "Same patient, code and provider as another claim, but 21 days later, outside the 14 day window.",
          evidence=[P("POL-004.3")], pkey="test_gap", offset=21))
    add(S("test", "hard_negative", "29881", ["M23.222"], "or2", n_scope_tear(r, "medial", "left"),
          "No RT or LT modifier, but the diagnosis code itself specifies the left knee.",
          evidence=[P("POL-001.2"), C("M23.222")], reading=True))
    add(S("test", "hard_negative", "73721", ["M25.561"], "or1", n_mri_redflag(r),
          "No conservative therapy, but a locked knee is documented, which is an allowed exception.",
          evidence=[P("POL-003.3")], reading=True))
    add(S("test", "hard_negative", "73721", ["M25.562"], "or2", n_mri_with_therapy(r, "home"),
          "Note mentions only home exercises and a brace, but for 8 weeks, which meets the requirement.",
          evidence=[P("POL-003.1")], reading=True))
    add(S("test", "hard_negative", "99214", ["J45.909"], "im", n_asthma_em25(r),
          "Office visit billed with a same-day injection, and modifier 25 is present.",
          modifiers=["25"], evidence=[P("POL-006.1"), P("POL-006.2")], pkey="test_mod25_b"))
    add(S("test", "hard_negative", "97110", ["M25.561"], "pt", n_pt(r, 60),
          "Four units of 97110, exactly at the limit.", units=4, evidence=[P("POL-005.3")]))
    add(S("test", "hard_negative", "J3301", ["M25.511"], "or1", n_inj_triam(r, 80),
          "Eight units of J3301 for a documented 80 mg dose, exactly at the limit.",
          units=8, evidence=[P("POL-005.3"), P("POL-007.2")]))
    add(S("test", "hard_negative", "29881", ["M23.251", "M17.11"], "or2", n_scope_tear_with_oa(r),
          "Osteoarthritis code is also on the claim, but the note documents a meniscal tear as the reason for surgery.",
          evidence=[P("POL-001.1")], reading=True))

    # Clean (13)
    add(S("test", "clean", "99213", ["I10"], "im", t_dup_note,
          "Routine hypertension visit. It is the earlier claim of a pair.", pkey="test_dup", offset=0))
    add(S("test", "clean", "99214", ["E11.9"], "im", n_dm(r),
          "Routine visit. It is the earlier claim of a pair.", pkey="test_gap", offset=0))
    add(S("test", "clean", "20610", ["M25.511"], "or1", n_proc_20610(r),
          "Joint injection. It is the procedure claim of a same-day pair.", pkey="test_mod25_a"))
    add(S("test", "clean", "96372", ["J45.909"], "im", n_proc_96372(r),
          "Injection given in clinic. It is the procedure claim of a same-day pair.", pkey="test_mod25_b"))
    add(S("test", "clean", "99213", ["J06.9"], "fm", n_uri(r), "Routine minor illness visit."))
    add(S("test", "clean", "99214", ["I10"], "im", n_htn(r), "Routine hypertension visit."))
    add(S("test", "clean", "99214", ["J45.909"], "fm", n_asthma(r), "Routine asthma follow up."))
    add(S("test", "clean", "36415", ["E11.9"], "im", n_blood_draw(r), "Routine blood draw."))
    add(S("test", "clean", "71046", ["J06.9"], "im", n_cxr_uri(r), "Chest x-ray for a respiratory complaint."))
    add(S("test", "clean", "93000", ["R07.9"], "ca", n_ecg_cp(r), "ECG for chest pain."))
    add(S("test", "clean", "99215", ["N18.4"], "im", n_ckd(r),
          "Level 5 visit supported by a stage 4 kidney disease diagnosis.", evidence=[P("POL-002.3"), C("N18.4")]))
    add(S("test", "clean", "29881", ["M23.221"], "or1", n_scope_tear(r, "medial", "right"),
          "Documented meniscal tear, with an RT modifier and a right knee diagnosis code.",
          modifiers=["RT"], evidence=[P("POL-001.1"), P("POL-001.2")], reading=True))
    add(S("test", "clean", "73721", ["M25.561"], "or2", n_mri_with_therapy(r, "supervised"),
          "Knee MRI after at least 6 weeks of documented supervised therapy.",
          evidence=[P("POL-003.1")], reading=True))
    return s


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def generate():
    r = random.Random(SEED)
    tables, catalog = load_tables(), load_catalog()
    params = catalog["parameters"]
    specs = build_specs(r)

    # Patients: one per pkey, ref numbers assigned after a seeded shuffle.
    keys = []
    for i, sp in enumerate(specs):
        sp["pkey"] = sp["pkey"] or f"solo_{i}"
        if sp["pkey"] not in keys:
            keys.append(sp["pkey"])
    order = keys[:]
    r.shuffle(order)
    patient = {}
    for n, k in enumerate(order, start=1):
        patient[k] = {"ref": f"P-{n:04d}", "band": pick(r, AGE_BANDS), "sex": pick(r, ["F", "M"])}

    # Dates: one base per patient group, plus the spec offset.
    base = {}
    for k in keys:
        max_off = max(sp["offset"] for sp in specs if sp["pkey"] == k)
        base[k] = r.randint(0, DAYS_IN_RANGE - 1 - max_off)

    rows = []
    for sp in specs:
        band = params["price_bands_per_unit"].get(sp["cpt"])
        if sp["price"] == "unknown":
            unit = round(r.uniform(180, 320), 2)
        elif sp["price"] == "top":
            unit = round(0.97 * band[1], 2)
        elif sp["price"] == "out":
            unit = round(band[1] * r.uniform(2.4, 3.4), 2)
        else:
            unit = round(band[0] + (band[1] - band[0]) * r.uniform(0.15, 0.85), 2)
        p = patient[sp["pkey"]]
        claim = {
            "claim_id": "",
            "patient_ref": p["ref"],
            "patient_age_band": p["band"],
            "patient_sex": p["sex"],
            "date_of_service": (START + timedelta(days=base[sp["pkey"]] + sp["offset"])).isoformat(),
            "provider": dict(PROVIDERS[sp["provider"]]),
            "place_of_service": POS_BY_CPT.get(sp["cpt"], "11"),
            "cpt": sp["cpt"],
            "modifiers": list(sp["modifiers"]),
            "icd10": list(sp["icd10"]),
            "units": sp["units"],
            "billed_amount": round(unit * sp["units"], 2),
            "clinical_note": sp["note"],
        }
        rows.append((claim, sp))

    r.shuffle(rows)
    claims, gts = [], []
    for n, (claim, sp) in enumerate(rows, start=1):
        claim["claim_id"] = f"V2-{n:04d}"
        claims.append({k: claim[k] for k in CLAIM_FIELDS})
        gts.append({
            "claim_id": claim["claim_id"],
            "split": sp["split"],
            "category": sp["category"],
            "is_anomalous": sp["category"] == "anomaly",
            "expected_flags": sorted(sp["flags"]),
            "evidence": sp["evidence"],
            "needs_policy_reading": sp["reading"],
            "notes": sp["why"],
        })
    order_by_id = sorted(range(len(claims)), key=lambda i: claims[i]["claim_id"])
    claims = [claims[i] for i in order_by_id]
    gts = [gts[i] for i in order_by_id]
    verify_labels(claims, gts, tables, params)
    return claims, gts


def verify_labels(claims, gts, tables, params):
    """Fail generation if any deterministic flag disagrees with the ground truth."""
    for c, g in zip(claims, gts):
        assert c["claim_id"] == g["claim_id"]
        derived = derive_rule_flags(c, claims, tables, params)
        expected_rule = {f for f in g["expected_flags"] if f.startswith("R_")}
        assert derived == expected_rule, (c["claim_id"], "derived", sorted(derived), "expected", sorted(expected_rule))
        assert g["is_anomalous"] == bool(g["expected_flags"]), c["claim_id"]
        assert g["category"] in ("clean", "hard_negative", "anomaly"), c["claim_id"]


# --------------------------------------------------------------------------
# Output and freeze
# --------------------------------------------------------------------------
def render_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def test_split_hash(claims, gts):
    ids = {g["claim_id"] for g in gts if g["split"] == "test"}
    payload = {
        "claims": sorted((c for c in claims if c["claim_id"] in ids), key=lambda c: c["claim_id"]),
        "ground_truth": sorted((g for g in gts if g["claim_id"] in ids), key=lambda g: g["claim_id"]),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Generate dataset v2 (seed 42).")
    ap.add_argument("--freeze", action="store_true", help="also write data/v2/TEST_SPLIT.sha256")
    args = ap.parse_args()
    claims, gts = generate()
    DATA_V2.mkdir(parents=True, exist_ok=True)
    (DATA_V2 / "claims.json").write_text(render_json(claims))
    (DATA_V2 / "ground_truth.json").write_text(render_json(gts))
    digest = test_split_hash(claims, gts)
    if args.freeze:
        (DATA_V2 / "TEST_SPLIT.sha256").write_text(digest + "\n")
    print(f"wrote {len(claims)} claims; test split sha256 {digest}" + (" (frozen)" if args.freeze else " (not frozen)"))


if __name__ == "__main__":
    main()
