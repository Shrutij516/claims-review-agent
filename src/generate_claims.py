"""
Builds the synthetic claims test set. Fully hand-specified (no randomness)
so it is deterministic and every label is traceable to a reason.

Writes two files:
  data/claims.json       - what the pipeline actually sees: claim_id + a
                            semi-structured raw claim line (this is the
                            "input" both the baseline and the agent parse).
  data/ground_truth.json - the answer key: is this claim anomalous, what
                            type, and why. Never passed to the pipeline;
                            evaluate.py only uses it for scoring afterward.

CPT/ICD-10 codes are real code *formats* used in realistic combinations;
all patients, providers, dates, and amounts are fabricated. No PHI.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Each entry: structured fields + a ground-truth explanation.
# anomaly_type in {None, "upcoding", "mismatch", "amount_outlier", "combined"}
CLAIMS = [
    # ---- Clean: straightforward, everything matches (10) ----
    dict(id="CLM-0001", date="2026-03-02", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="99213", icd="J06.9", amount=95.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Low-complexity office visit for a common cold; code level and amount both typical."),
    dict(id="CLM-0002", date="2026-03-02", provider="Dr. B. Chen", specialty="Internal Medicine",
         cpt="99214", icd="I10", amount=175.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Moderate-complexity visit for hypertension management; tier and amount match."),
    dict(id="CLM-0003", date="2026-03-03", provider="Dr. C. Whitfield", specialty="Cardiology",
         cpt="99215", icd="I50.9", amount=290.00,
         is_anomalous=False, anomaly_type=None,
         rationale="High-complexity visit for heart failure; diagnosis justifies the code level."),
    dict(id="CLM-0004", date="2026-03-03", provider="Dr. C. Whitfield", specialty="Cardiology",
         cpt="93000", icd="I10", amount=45.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Routine ECG for a hypertensive patient; standard cardiac indication and price."),
    dict(id="CLM-0005", date="2026-03-04", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="71046", icd="J06.9", amount=85.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Chest X-ray for a respiratory complaint; procedure/diagnosis pairing is expected."),
    dict(id="CLM-0006", date="2026-03-04", provider="Dr. B. Chen", specialty="Internal Medicine",
         cpt="80053", icd="E11.9", amount=42.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Metabolic panel for diabetes monitoring; routine lab order at typical price."),
    dict(id="CLM-0007", date="2026-03-05", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="36415", icd="M54.5", amount=15.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Routine venipuncture accompanying a back-pain workup; typical price."),
    dict(id="CLM-0008", date="2026-03-05", provider="Dr. D. Okafor", specialty="Orthopedics",
         cpt="20610", icd="M25.561", amount=110.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Joint injection for knee pain; procedure/diagnosis and price all typical."),
    dict(id="CLM-0009", date="2026-03-06", provider="Dr. D. Okafor", specialty="Orthopedics",
         cpt="29881", icd="M23.221", amount=1200.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Knee arthroscopy for a documented meniscus tear; expected pairing and price."),
    dict(id="CLM-0010", date="2026-03-06", provider="Dr. C. Whitfield", specialty="Cardiology",
         cpt="93017", icd="I20.9", amount=210.00,
         is_anomalous=False, anomaly_type=None,
         rationale="Cardiac stress test for angina; expected pairing and price."),

    # ---- Borderline clean: trip a sensitive rule threshold but are legitimate (5) ----
    dict(id="CLM-0011", date="2026-03-07", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="99214", icd="J06.9", amount=165.00,
         is_anomalous=False, anomaly_type="borderline_upcoding",
         rationale="Visit addressed the cold plus two additional documented complaints, justifying "
                    "one tier above what the primary diagnosis alone implies (tier_gap=1)."),
    dict(id="CLM-0012", date="2026-03-08", provider="Dr. B. Chen", specialty="Internal Medicine",
         cpt="99215", icd="I10", amount=270.00,
         is_anomalous=False, anomaly_type="borderline_upcoding",
         rationale="Hypertension visit with extensive medication reconciliation and counseling "
                    "documented; one tier above the primary diagnosis alone (tier_gap=1)."),
    dict(id="CLM-0013", date="2026-03-09", provider="Dr. C. Whitfield", specialty="Cardiology",
         cpt="93017", icd="I20.9", amount=370.00,
         is_anomalous=False, anomaly_type="borderline_amount",
         rationale="Extended stress test protocol with extra monitoring time documented; "
                    "billed above the typical band but within reason for the added time."),
    dict(id="CLM-0014", date="2026-03-09", provider="Dr. B. Chen", specialty="Internal Medicine",
         cpt="80053", icd="E11.9", amount=14.00,
         is_anomalous=False, anomaly_type="borderline_amount",
         rationale="Discounted panel rate under a patient assistance program; unusually low "
                    "but a legitimate billed amount, not a coding error."),
    dict(id="CLM-0015", date="2026-03-10", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="71046", icd="Z01.89", amount=95.00,
         is_anomalous=False, anomaly_type="borderline_mismatch",
         rationale="Pre-operative clearance chest X-ray ordered under a general-exam code; "
                    "not in the procedure's usual diagnosis list but a routine, legitimate order."),

    # ---- Anomalous: clear upcoding, tier_gap=2 (4) ----
    dict(id="CLM-0016", date="2026-03-11", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="99215", icd="Z00.00", amount=310.00,
         is_anomalous=True, anomaly_type="upcoding",
         rationale="Billed as high-complexity for a routine general exam (tier_gap=2) — upcoding."),
    dict(id="CLM-0017", date="2026-03-11", provider="Dr. B. Chen", specialty="Internal Medicine",
         cpt="99215", icd="J06.9", amount=300.00,
         is_anomalous=True, anomaly_type="upcoding",
         rationale="Billed as high-complexity for a common cold (tier_gap=2) — upcoding."),
    dict(id="CLM-0018", date="2026-03-12", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="99215", icd="M54.5", amount=305.00,
         is_anomalous=True, anomaly_type="upcoding",
         rationale="Billed as high-complexity for routine low back pain (tier_gap=2) — upcoding."),
    dict(id="CLM-0019", date="2026-03-12", provider="Dr. E. Larsen", specialty="Neurology",
         cpt="99215", icd="R51.9", amount=315.00,
         is_anomalous=True, anomaly_type="upcoding",
         rationale="Billed as high-complexity for an uncomplicated headache (tier_gap=2) — upcoding."),

    # ---- Anomalous: clear code/diagnosis mismatch (4) ----
    dict(id="CLM-0020", date="2026-03-13", provider="Dr. C. Whitfield", specialty="Cardiology",
         cpt="93017", icd="S93.401A", amount=220.00,
         is_anomalous=True, anomaly_type="mismatch",
         rationale="Cardiac stress test billed against an ankle sprain diagnosis — unrelated codes."),
    dict(id="CLM-0021", date="2026-03-13", provider="Dr. D. Okafor", specialty="Orthopedics",
         cpt="29881", icd="J06.9", amount=1100.00,
         is_anomalous=True, anomaly_type="mismatch",
         rationale="Knee arthroscopy billed against a common-cold diagnosis — unrelated codes."),
    dict(id="CLM-0022", date="2026-03-14", provider="Dr. D. Okafor", specialty="Orthopedics",
         cpt="20610", icd="R51.9", amount=120.00,
         is_anomalous=True, anomaly_type="mismatch",
         rationale="Joint injection billed against a headache diagnosis — unrelated codes."),
    dict(id="CLM-0023", date="2026-03-14", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="71046", icd="M54.5", amount=90.00,
         is_anomalous=True, anomaly_type="mismatch",
         rationale="Chest X-ray billed against a back-pain diagnosis — unrelated codes."),

    # ---- Anomalous: clear billed-amount outlier (4) ----
    dict(id="CLM-0024", date="2026-03-15", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="99213", icd="J06.9", amount=310.00,
         is_anomalous=True, anomaly_type="amount_outlier",
         rationale="Low-complexity visit billed at ~2x the typical maximum for its code."),
    dict(id="CLM-0025", date="2026-03-16", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="36415", icd="M54.5", amount=95.00,
         is_anomalous=True, anomaly_type="amount_outlier",
         rationale="Routine venipuncture billed at ~4x its typical maximum."),
    dict(id="CLM-0026", date="2026-03-16", provider="Dr. B. Chen", specialty="Internal Medicine",
         cpt="80053", icd="E11.9", amount=180.00,
         is_anomalous=True, anomaly_type="amount_outlier",
         rationale="Metabolic panel billed at ~3x its typical maximum."),
    dict(id="CLM-0027", date="2026-03-17", provider="Dr. C. Whitfield", specialty="Cardiology",
         cpt="93000", icd="I10", amount=210.00,
         is_anomalous=True, anomaly_type="amount_outlier",
         rationale="Routine ECG billed at ~3x its typical maximum."),

    # ---- Anomalous: multiple rules triggered at once, severe (3) ----
    dict(id="CLM-0028", date="2026-03-18", provider="Dr. A. Ramirez", specialty="Family Medicine",
         cpt="99215", icd="Z00.00", amount=650.00,
         is_anomalous=True, anomaly_type="combined",
         rationale="High-complexity code for a routine exam AND billed ~2x its own (already-high) "
                    "typical maximum — upcoding plus amount outlier together."),
    dict(id="CLM-0029", date="2026-03-19", provider="Dr. D. Okafor", specialty="Orthopedics",
         cpt="29881", icd="J06.9", amount=3200.00,
         is_anomalous=True, anomaly_type="combined",
         rationale="Knee arthroscopy for a common cold AND billed ~2x its typical maximum — "
                    "mismatch plus amount outlier together."),
    dict(id="CLM-0030", date="2026-03-20", provider="Dr. C. Whitfield", specialty="Cardiology",
         cpt="93017", icd="S93.401A", amount=700.00,
         is_anomalous=True, anomaly_type="combined",
         rationale="Cardiac stress test for an ankle sprain AND billed ~2.5x its typical maximum — "
                    "mismatch plus amount outlier together."),
]


def render_raw_note(c):
    """Render the structured claim as the semi-structured line the pipeline receives."""
    return (
        f"{c['id']} | DOS: {c['date']} | Provider: {c['provider']} ({c['specialty']}) | "
        f"CPT: {c['cpt']} | ICD-10: {c['icd']} | Billed: ${c['amount']:.2f} | Units: 1"
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    claims_out = [{"claim_id": c["id"], "raw_note": render_raw_note(c)} for c in CLAIMS]
    ground_truth_out = [
        {
            "claim_id": c["id"],
            "is_anomalous": c["is_anomalous"],
            "anomaly_type": c["anomaly_type"],
            "rationale": c["rationale"],
            "structured": {
                "cpt_code": c["cpt"], "icd10_code": c["icd"],
                "billed_amount": c["amount"], "provider": c["provider"],
                "specialty": c["specialty"], "date_of_service": c["date"],
            },
        }
        for c in CLAIMS
    ]

    (DATA_DIR / "claims.json").write_text(json.dumps(claims_out, indent=2))
    (DATA_DIR / "ground_truth.json").write_text(json.dumps(ground_truth_out, indent=2))

    n_anom = sum(c["is_anomalous"] for c in CLAIMS)
    print(f"Wrote {len(CLAIMS)} claims to {DATA_DIR / 'claims.json'} "
          f"({n_anom} anomalous, {len(CLAIMS) - n_anom} clean) "
          f"and ground truth to {DATA_DIR / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
