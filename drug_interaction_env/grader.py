"""
grader.py — OpenEnv standalone grading logic.

Evaluates an episode's final state and generates a normalized [0.0, 1.0] score.
Uses the identical penalty logic computed during the environment rollout.
"""

from drug_interaction_env.server.drug_database import DRUG_INTERACTIONS

TASK_TRUE_PAIRS = {
    "easy":   1,
    "medium": 3,
    "hard":   5
}

def grade_episode(task_level: str, final_state: dict) -> float:
    """
    Takes the final state dict from the environment at episode end.
    Computes normalized episode score in [0.0, 1.0].
    """
    n_true_pairs = TASK_TRUE_PAIRS.get(task_level, 1)
    max_possible = n_true_pairs * 0.8

    predictions = final_state.get("predictions", {})
    identified = set(final_state.get("identified_pairs", []))
    ground_truth_keys = set(final_state.get("ground_truth_keys", []))

    # Sum rewards from all predictions
    total_reward = sum(p.get("reward_received", 0.0) for p in predictions.values())

    # Apply severity-weighted termination penalties for unidentified pairs
    unidentified = ground_truth_keys - identified
    for key_str in unidentified:
        # Expected serialization format from environment.py: "drug_a|drug_b"
        key = tuple(key_str.split("|"))  
        gt = DRUG_INTERACTIONS.get(key, {})
        severity = gt.get("severity", "mild")
        if severity == "severe":
            total_reward -= 0.4
        elif severity == "moderate":
            total_reward -= 0.3
        else:
            total_reward -= 0.2

    # Normalize
    if max_possible == 0:
        return 1.0
        
    return max(0.0, min(1.0, total_reward / max_possible))

def verify_score_range(score: float) -> bool:
    """Helper purely to assert bound checking."""
    return 0.0 <= score <= 1.0
