"""
patients.py — Patient scenario definitions for the Drug Interaction Checker.

Three difficulty levels:
  easy   →  6 medications,  exactly 1 known interaction
  medium → 10 medications,  exactly 3 known interactions (1 severe, 1 moderate, 1 mild)
  hard   → 15 medications,  exactly 5 known interactions (mixed severity)

Ground truth interactions are NOT stored here — they are derived at runtime
by scanning each patient's medications against DRUG_INTERACTIONS.

IMPORTANT: Medications are chosen carefully so that only the intended interactions
fire. "Safe" filler drugs (no entries in DRUG_INTERACTIONS) are used to pad lists.
"""

PATIENTS: dict[str, dict] = {

    # ═════════════════════════════════════════════════════════════════════════
    # EASY — 6 drugs, 1 interaction
    # ═════════════════════════════════════════════════════════════════════════
    # Intended interaction:
    #   (aspirin, warfarin) → severe / replace_drug
    "easy": {
        "patient_id": "P001",
        "age": 67,
        "conditions": ["hypertension", "type2_diabetes"],
        "medications": [
            "warfarin",
            "aspirin",
            "losartan",
            "amlodipine",
            "gabapentin",
            "pantoprazole",
        ],
    },

    # ═════════════════════════════════════════════════════════════════════════
    # MEDIUM — 10 drugs, 3 interactions
    # ═════════════════════════════════════════════════════════════════════════
    # Intended interactions:
    #   (clopidogrel, omeprazole)    → severe   / replace_drug
    #   (ciprofloxacin, theophylline)→ moderate / reduce_dose
    #   (calcium, levothyroxine)     → mild     / monitor
    #
    # No warfarin (interacts with too many things), no amiodarone, no atorvastatin
    "medium": {
        "patient_id": "P002",
        "age": 54,
        "conditions": ["asthma", "hypothyroidism", "cardiovascular_disease"],
        "medications": [
            "clopidogrel",
            "omeprazole",
            "ciprofloxacin",
            "theophylline",
            "levothyroxine",
            "calcium",
            "losartan",
            "gabapentin",
            "metoprolol",
            "acetaminophen",
        ],
    },

    # ═════════════════════════════════════════════════════════════════════════
    # HARD — 15 drugs, 5 interactions
    # ═════════════════════════════════════════════════════════════════════════
    # Intended interactions:
    #   (aspirin, warfarin)           → severe   / replace_drug
    #   (amiodarone, simvastatin)     → severe   / reduce_dose
    #   (ssri, tramadol)              → severe   / replace_drug
    #   (digoxin, furosemide)         → moderate / monitor
    #   (calcium, levothyroxine)      → mild     / monitor
    #
    # AVOID putting warfarin with amiodarone (that's an extra interaction!)
    # So we DON'T include warfarin+amiodarone together.
    # Solution: remove warfarin from hard, use ibuprofen for the severe one instead.
    # Wait — (ibuprofen, warfarin) is severe... let me restructure.
    # Actually let's keep warfarin+aspirin but remove amiodarone's conflict with warfarin
    # by not having warfarin here. Use different severe pairs.
    #
    # Revised plan — NO warfarin in hard:
    #   (amiodarone, simvastatin)     → severe   / reduce_dose
    #   (ssri, tramadol)              → severe   / replace_drug
    #   (amiodarone, digoxin)         → severe   / reduce_dose
    #   (digoxin, furosemide)         → moderate / monitor
    #   (calcium, levothyroxine)      → mild     / monitor
    #
    # Check: amiodarone+simvastatin ✓, ssri+tramadol ✓, amiodarone+digoxin ✓,
    #         digoxin+furosemide ✓, calcium+levothyroxine ✓
    # Any others? digoxin+omeprazole is mild — don't include omeprazole!
    #             amiodarone+atorvastatin is moderate — don't include atorvastatin!
    #             digoxin+spironolactone is moderate — don't include spironolactone!
    #             furosemide+metformin is mild — don't include metformin!
    #             furosemide+lisinopril is moderate — don't include lisinopril!
    "hard": {
        "patient_id": "P003",
        "age": 71,
        "conditions": ["heart_failure", "depression", "hypothyroidism"],
        "medications": [
            "amiodarone",
            "simvastatin",
            "ssri",
            "tramadol",
            "digoxin",
            "furosemide",
            "levothyroxine",
            "calcium",
            "losartan",
            "gabapentin",
            "pantoprazole",
            "metoprolol",
            "acetaminophen",
            "ranitidine",
            "hydroxyzine",
        ],
    },
}


# ─── Verification script ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from itertools import combinations
    from drug_database import DRUG_INTERACTIONS, lookup_pair

    expected = {"easy": 1, "medium": 3, "hard": 5}

    all_pass = True
    for level, patient in PATIENTS.items():
        meds = patient["medications"]
        interactions = [
            (a, b)
            for a, b in combinations(meds, 2)
            if lookup_pair(a, b) is not None
        ]
        count = len(interactions)
        status = "PASS ✓" if count == expected[level] else "FAIL ✗"
        if count != expected[level]:
            all_pass = False
        print(f"{level}: {count} interactions found (expected {expected[level]}) {status}")
        for a, b in interactions:
            info = lookup_pair(a, b)
            print(f"  → ({a}, {b}) severity={info['severity']} action={info['action']}")

    print(f"\n{'ALL CHECKS PASSED ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")
