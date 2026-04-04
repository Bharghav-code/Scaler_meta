from itertools import combinations
from drug_database import DRUG_INTERACTIONS, lookup_pair
from patients import PATIENTS

for level, patient in PATIENTS.items():
    meds = patient["medications"]
    interactions = []
    for a, b in combinations(meds, 2):
        result = lookup_pair(a, b)
        if result is not None:
            interactions.append((a, b, result["severity"], result["action"]))
    print(f"\n{level.upper()}: {len(interactions)} interactions found")
    for a, b, sev, act in interactions:
        print(f"  ({a}, {b}) => severity={sev}, action={act}")
