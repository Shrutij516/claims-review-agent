# Code table sources and verification status

STATUS: NOT YET VERIFIED. None of the tables in this folder has been checked
against an official file. They were written from memory and must be verified
before anyone treats them as accurate.

## Where each table should be verified

| File | Verify against | Notes |
|---|---|---|
| icd10.json | CMS ICD-10-CM code files (current fiscal year release: code descriptions in tabular order, and the code list) | Check that every code exists and is billable. Check the laterality field against the official descriptions. The last character of the M23.2, M25.5 and M17.1 codes is expected to carry the side, but that is our reading and is unverified. |
| cpt.json | CPT is licensed by the AMA and is not redistributed here. Code numbers can be checked against the CMS Physician Fee Schedule relative value files, which list code numbers. | Only code numbers are used. Descriptions are original summaries and are not copied from any source. Check modifiers 25, RT and LT against the official modifier lists. |
| hcpcs.json | CMS HCPCS Level II quarterly update file | Check each J code exists and check its billing unit (per 10 mg, 1 mg, 40 mg, 250 mg). |

## Intentionally invalid codes used in the claims

The claims in data/v2 include a few codes that are meant to be absent from the
tables, so that R_INVALID_CODE can be tested. Verify these really do not exist
in the official files, otherwise the labels are wrong:

- CPT 99216 (office visit family stops at 99215)
- ICD-10 M17.99 (M17.9 is already a complete code)
- ICD-10 I1O (letter O in place of zero)

## Everything else is synthetic

Policies (docs/policies), price bands, unit limits and diagnosis tiers (in
data/v2/flag_catalog.json) are invented for this project and are not taken
from any real payer or from CMS.
