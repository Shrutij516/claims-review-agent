"""
Billing rules / policy table shared by both the baseline classifier and the
agentic pipeline. This is the "small policy table" both approaches consult —
the baseline applies it with fixed if/else logic, the agent uses it as
evidence for chain-of-thought reasoning.

All thresholds are deliberately a little sensitive (they will also trip on a
handful of legitimate, clinically-justified claims in the synthetic test
set). That's intentional: it's what lets us measure whether the agent's
reasoning step can tell a real anomaly apart from a rule tripped by normal
clinical variation, instead of just re-deriving the rule table's verdict.
"""

# CPT code -> billing/coding policy info.
# typical_range: (low, high) dollars for a routine, correctly-coded claim.
# em_tier: complexity tier for Evaluation & Management (office visit) codes
#          only; used for the upcoding check. None for non-E/M procedures.
# compatible_icd_prefixes: ICD-10 code prefixes this procedure is clinically
#          expected to pair with. None means "no restriction checked" (the
#          procedure is broad/general enough that any diagnosis is plausible).
CPT_CATALOG = {
    "99213": dict(
        description="Office/outpatient visit, established patient — low/moderate complexity",
        typical_range=(75, 150),
        em_tier=1,
        compatible_icd_prefixes=None,
    ),
    "99214": dict(
        description="Office/outpatient visit, established patient — moderate complexity",
        typical_range=(130, 220),
        em_tier=2,
        compatible_icd_prefixes=None,
    ),
    "99215": dict(
        description="Office/outpatient visit, established patient — high complexity",
        typical_range=(200, 320),
        em_tier=3,
        compatible_icd_prefixes=None,
    ),
    "93000": dict(
        description="Electrocardiogram, routine ECG with interpretation",
        typical_range=(30, 70),
        em_tier=None,
        compatible_icd_prefixes=["I", "R00", "R07"],
    ),
    "93017": dict(
        description="Cardiovascular stress test, tracing only",
        typical_range=(150, 280),
        em_tier=None,
        compatible_icd_prefixes=["I", "R07"],
    ),
    "71046": dict(
        description="Chest X-ray, 2 views",
        typical_range=(60, 120),
        em_tier=None,
        compatible_icd_prefixes=["J", "R07", "R05", "I"],
    ),
    "80053": dict(
        description="Comprehensive metabolic panel",
        typical_range=(30, 60),
        em_tier=None,
        compatible_icd_prefixes=None,
    ),
    "29881": dict(
        description="Knee arthroscopy with meniscectomy",
        typical_range=(900, 1600),
        em_tier=None,
        compatible_icd_prefixes=["M23", "M17", "S83"],
    ),
    "20610": dict(
        description="Arthrocentesis/injection, major joint",
        typical_range=(80, 150),
        em_tier=None,
        compatible_icd_prefixes=["M25", "M17", "M23"],
    ),
    "36415": dict(
        description="Venipuncture",
        typical_range=(10, 25),
        em_tier=None,
        compatible_icd_prefixes=None,
    ),
}

# ICD-10 code -> documented complexity tier (1=low, 2=moderate, 3=high).
# Used only to sanity-check E/M (office visit) code level against the
# diagnosis on the claim. Codes not listed here simply skip the upcoding
# check for that claim (tier_gap stays None).
ICD_TIER = {
    "Z00.00": 1, "J06.9": 1, "M54.5": 1, "R51.9": 1, "R10.9": 1,
    "I10": 2, "E11.9": 2, "J45.909": 2, "M25.561": 2,
    "I50.9": 3, "N17.9": 3, "C50.911": 3, "I21.4": 3,
}

AMOUNT_OUTLIER_HIGH_MULT = 1.3   # billed > 1.3x typical max -> outlier
AMOUNT_OUTLIER_LOW_MULT = 0.5    # billed < 0.5x typical min -> outlier
UPCODING_TIER_GAP_THRESHOLD = 1  # em_tier - icd_tier >= this -> upcoding flag


def get_cpt_info(cpt_code):
    if cpt_code not in CPT_CATALOG:
        raise KeyError(f"Unknown CPT code '{cpt_code}' — not in policy table")
    return CPT_CATALOG[cpt_code]


def rule_check(claim):
    """
    Deterministically check one structured claim against the policy table.

    claim: dict with at least cpt_code, icd10_code, billed_amount
    returns: dict of individual flags plus supporting numbers, e.g.
        {
          "amount_outlier": bool, "amount_ratio_to_typical_max": float,
          "upcoding": bool, "tier_gap": int|None,
          "mismatch": bool,
          "any_flag": bool,
        }
    """
    info = get_cpt_info(claim["cpt_code"])
    icd = claim["icd10_code"]
    billed = float(claim["billed_amount"])
    lo, hi = info["typical_range"]

    amount_outlier = billed > AMOUNT_OUTLIER_HIGH_MULT * hi or billed < AMOUNT_OUTLIER_LOW_MULT * lo

    tier_gap = None
    upcoding = False
    if info["em_tier"] is not None and icd in ICD_TIER:
        tier_gap = info["em_tier"] - ICD_TIER[icd]
        upcoding = tier_gap >= UPCODING_TIER_GAP_THRESHOLD

    mismatch = False
    if info["compatible_icd_prefixes"] is not None:
        mismatch = not any(icd.startswith(p) for p in info["compatible_icd_prefixes"])

    return {
        "cpt_description": info["description"],
        "typical_range": [lo, hi],
        "amount_outlier": amount_outlier,
        "amount_ratio_to_typical_max": round(billed / hi, 2),
        "upcoding": upcoding,
        "tier_gap": tier_gap,
        "mismatch": mismatch,
        "any_flag": amount_outlier or upcoding or mismatch,
    }
