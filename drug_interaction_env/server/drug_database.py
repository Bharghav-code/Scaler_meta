"""
drug_database.py — Hardcoded dictionary of real, publicly documented drug-drug interactions.

Keys are always tuple(sorted([drug_a.lower(), drug_b.lower()])) for deterministic lookup.
Each entry contains severity, recommended action, and a brief clinical explanation.

Sources: FDA drug interactions database, clinical pharmacology references.
"""

# ─── DRUG INTERACTIONS DATABASE ────────────────────────────────────────────────
# 35 real, well-documented interaction pairs
# Distribution: 12 severe, 13 moderate, 10 mild

DRUG_INTERACTIONS: dict[tuple[str, str], dict] = {

    # ══════════════════════════════════════════════════════════════════════════
    # SEVERE INTERACTIONS (12)
    # ══════════════════════════════════════════════════════════════════════════

    ("aspirin", "warfarin"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Increased bleeding risk — dual antiplatelet + anticoagulant"
    },
    ("ibuprofen", "warfarin"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Increased bleeding risk, GI complications"
    },
    ("amiodarone", "simvastatin"): {
        "severity": "severe",
        "action": "reduce_dose",
        "explanation": "Risk of myopathy and rhabdomyolysis — CYP3A4 inhibition"
    },
    ("ssri", "tramadol"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Serotonin syndrome risk — dual serotonergic agents"
    },
    ("amiodarone", "warfarin"): {
        "severity": "severe",
        "action": "reduce_dose",
        "explanation": "Potentiates warfarin — increased INR and bleeding risk"
    },
    ("amiodarone", "digoxin"): {
        "severity": "severe",
        "action": "reduce_dose",
        "explanation": "Amiodarone increases digoxin levels — risk of toxicity"
    },
    ("lithium", "nsaid"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "NSAIDs reduce lithium clearance — toxicity risk"
    },
    ("methotrexate", "trimethoprim"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Increased methotrexate toxicity — folate antagonism"
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Omeprazole inhibits CYP2C19 — reduces clopidogrel activation"
    },
    ("clarithromycin", "simvastatin"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "CYP3A4 inhibition — rhabdomyolysis risk"
    },
    ("fluconazole", "warfarin"): {
        "severity": "severe",
        "action": "reduce_dose",
        "explanation": "CYP2C9 inhibition — markedly increased INR"
    },
    ("maoi", "ssri"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Fatal serotonin syndrome risk — absolute contraindication"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODERATE INTERACTIONS (13)
    # ══════════════════════════════════════════════════════════════════════════

    ("ibuprofen", "metformin"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Reduced renal clearance of metformin"
    },
    ("digoxin", "furosemide"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Furosemide-induced hypokalemia increases digoxin toxicity risk"
    },
    ("ace_inhibitor", "spironolactone"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Risk of hyperkalemia — dual potassium retention"
    },
    ("lisinopril", "spironolactone"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Risk of hyperkalemia — ACE inhibitor + potassium-sparing diuretic"
    },
    ("amiodarone", "atorvastatin"): {
        "severity": "moderate",
        "action": "reduce_dose",
        "explanation": "Increased statin exposure — myopathy risk"
    },
    ("ciprofloxacin", "theophylline"): {
        "severity": "moderate",
        "action": "reduce_dose",
        "explanation": "CYP1A2 inhibition — theophylline toxicity"
    },
    ("contrast_dye", "metformin"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Risk of lactic acidosis — hold metformin around contrast procedures"
    },
    ("nsaid", "ssri"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Increased GI bleeding risk — dual serotonin + antiplatelet effect"
    },
    ("amlodipine", "simvastatin"): {
        "severity": "moderate",
        "action": "reduce_dose",
        "explanation": "Increased simvastatin levels — myopathy risk"
    },
    ("digoxin", "spironolactone"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Spironolactone reduces digoxin clearance — toxicity risk"
    },
    ("vitamin_e", "warfarin"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Vitamin E may potentiate warfarin anticoagulant effect"
    },
    ("furosemide", "lisinopril"): {
        "severity": "moderate",
        "action": "monitor",
        "explanation": "Risk of hypotension — first-dose effect with ACE inhibitor"
    },
    ("aspirin", "ibuprofen"): {
        "severity": "moderate",
        "action": "replace_drug",
        "explanation": "Ibuprofen antagonizes aspirin's antiplatelet effect"
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MILD INTERACTIONS (10)
    # ══════════════════════════════════════════════════════════════════════════

    ("metformin", "omeprazole"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "Omeprazole may slightly reduce vitamin B12 absorption affecting metformin users"
    },
    ("atorvastatin", "warfarin"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "Minor increase in warfarin effect — monitor INR"
    },
    ("lisinopril", "metformin"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "ACE inhibitors may slightly enhance insulin sensitivity"
    },
    ("amlodipine", "lisinopril"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "Additive blood pressure lowering — therapeutic but requires monitoring"
    },
    ("calcium", "levothyroxine"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "Calcium reduces levothyroxine absorption — separate by 4 hours"
    },
    ("omeprazole", "vitamin_d"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "Long-term PPI use may reduce calcium and vitamin D absorption"
    },
    ("aspirin", "metformin"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "Aspirin may slightly enhance metformin's glucose-lowering effect"
    },
    ("furosemide", "metformin"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "Furosemide may alter metformin plasma levels"
    },
    ("digoxin", "omeprazole"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "PPI may slightly increase digoxin absorption"
    },
    ("atorvastatin", "digoxin"): {
        "severity": "mild",
        "action": "monitor",
        "explanation": "Atorvastatin may slightly increase digoxin plasma concentration"
    },
}


def lookup_pair(drug_a: str, drug_b: str) -> dict | None:
    """Look up a drug pair in the interaction database.

    Keys are always normalized to sorted lowercase tuple, so
    lookup_pair("Warfarin", "aspirin") == lookup_pair("aspirin", "warfarin").

    Returns the interaction dict if found, else None.
    """
    key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
    return DRUG_INTERACTIONS.get(key, None)


# ─── Quick self-check ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    severe = [k for k, v in DRUG_INTERACTIONS.items() if v["severity"] == "severe"]
    moderate = [k for k, v in DRUG_INTERACTIONS.items() if v["severity"] == "moderate"]
    mild = [k for k, v in DRUG_INTERACTIONS.items() if v["severity"] == "mild"]
    print(f"Total interactions: {len(DRUG_INTERACTIONS)}")
    print(f"  Severe:   {len(severe)}")
    print(f"  Moderate: {len(moderate)}")
    print(f"  Mild:     {len(mild)}")
    # Verify lookup helper
    assert lookup_pair("Warfarin", "aspirin") is not None
    assert lookup_pair("aspirin", "warfarin") is not None
    assert lookup_pair("fakeDrug", "warfarin") is None
    print("All assertions passed ✓")
