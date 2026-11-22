"""Deterministic claims review tools: parsing, code checks, rules, retrieval, metrics, baseline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_V2 = ROOT / "data" / "v2"
DATA_CODES = ROOT / "data" / "codes"
POLICY_DIR = ROOT / "docs" / "policies"
