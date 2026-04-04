"""
drug_interaction_environment.py — Core environment logic for the Drug Interaction Checker.

Implements the DrugInteractionEnvironment class with:
  reset()  → initialize episode for a task level
  step()   → process one agent action
  state()  → return current internal state
  validate()          → check action validity
  calculate_reward()  → score a valid action
"""

from itertools import combinations

from drug_interaction_env.server.drug_database import DRUG_INTERACTIONS, lookup_pair
from drug_interaction_env.server.patients import PATIENTS
from drug_interaction_env.models import (
    DrugInteractionAction,
    DrugInteractionObservation,
    DrugInteractionState,
    FlagEntry,
)


class DrugInteractionEnvironment:
    """OpenEnv-compatible environment for pharmaceutical drug interaction checking."""

    def __init__(self):
        self.task_level: str | None = None
        self.patient: dict | None = None
        self.ground_truth_keys: set[tuple[str, str]] = set()
        self.attempted_keys: set[tuple[str, str]] = set()
        self.identified_pairs: set[tuple[str, str]] = set()
        self.perfectly_completed_pairs: set[tuple[str, str]] = set()
        self.predictions: dict[str, dict] = {}
        self.flags_raised: list[FlagEntry] = []
        self.step_count: int = 0
        self.max_steps: int = 0
        self.episode_reward: float = 0.0
        self.done: bool = False

    # ── reset ───────────────────────────────────────────────────────────────

    def reset(self, task_level: str = "easy") -> dict:
        """Initialize a new episode for the given task level."""
        if task_level not in PATIENTS:
            raise ValueError(f"Invalid task_level: {task_level}. Must be one of {list(PATIENTS.keys())}")

        self.task_level = task_level
        self.patient = PATIENTS[task_level]

        # Derive ground truth by scanning all medication pairs
        meds = self.patient["medications"]
        self.ground_truth_keys = set()
        for a, b in combinations(meds, 2):
            key = tuple(sorted([a.lower(), b.lower()]))
            if key in DRUG_INTERACTIONS:
                self.ground_truth_keys.add(key)

        # Set step budget
        self.max_steps = len(self.ground_truth_keys) * 3

        # Reset tracking state
        self.attempted_keys = set()
        self.identified_pairs = set()
        self.perfectly_completed_pairs = set()
        self.predictions = {}
        self.flags_raised = []
        self.step_count = 0
        self.episode_reward = 0.0
        self.done = False

        return self._get_observation()

    # ── validate ────────────────────────────────────────────────────────────

    def validate(self, action: dict) -> bool | None:
        """Validate a flag_interaction action.

        Returns:
            True  — valid pair found in DRUG_INTERACTIONS
            False — invalid drug name or phantom pair
            None  — duplicate (already attempted)
        """
        drug_a = action.get("drug_a", "")
        drug_b = action.get("drug_b", "")

        # Step 1: Normalize key
        key = tuple(sorted([drug_a.lower(), drug_b.lower()]))

        # Step 2: Validate drug names are in patient's medication list
        meds_lower = [m.lower() for m in self.patient["medications"]]
        if drug_a.lower() not in meds_lower or drug_b.lower() not in meds_lower:
            return False  # invalid drug name

        # Step 3: Check for duplicate
        if key in self.attempted_keys:
            return None  # duplicate

        # Step 4: Add to attempted keys
        self.attempted_keys.add(key)

        # Step 5: Check against DRUG_INTERACTIONS
        if key not in DRUG_INTERACTIONS:
            return False  # phantom pair

        return True  # valid pair

    # ── calculate_reward ────────────────────────────────────────────────────

    def calculate_reward(self, key: tuple, severity: str, action: str) -> float:
        """Calculate step reward for a valid pair.

        Args:
            key: Normalized sorted tuple (drug_a, drug_b)
            severity: Agent's predicted severity
            action: Agent's predicted suggested action
        """
        gt = DRUG_INTERACTIONS[key]

        # Base score: pair correctly identified
        base_score = 0.4

        # Severity scoring
        if severity == gt["severity"]:
            severity_score = 0.2
        elif gt["severity"] == "severe":
            severity_score = -0.2
        elif gt["severity"] == "moderate":
            severity_score = -0.1
        else:  # mild
            severity_score = -0.05

        # Action scoring
        if action == gt["action"]:
            action_score = 0.2
        else:
            action_score = -0.1

        step_reward = base_score + severity_score + action_score

        # Update tracking
        self.identified_pairs.add(key)
        self.predictions[f"{key[0]}|{key[1]}"] = {
            "predicted_severity": severity,
            "predicted_action": action,
            "ground_truth_severity": gt["severity"],
            "ground_truth_action": gt["action"],
            "reward_received": step_reward,
            "perfectly_completed": (severity == gt["severity"] and action == gt["action"]),
        }
        if severity == gt["severity"] and action == gt["action"]:
            self.perfectly_completed_pairs.add(key)

        return step_reward

    # ── step ────────────────────────────────────────────────────────────────

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Process one agent action.

        Returns:
            (observation, step_reward, done, state)
        """
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")

        # Handle DONE action
        if action.get("action_type") == "DONE":
            reward = self._apply_termination_penalty()
            self.done = True
            return self._get_observation(), reward, True, self._get_state()

        # Process flag_interaction action
        key = tuple(sorted([
            action.get("drug_a", "").lower(),
            action.get("drug_b", "").lower(),
        ]))

        result = self.validate(action)

        if result is None:
            # Duplicate
            step_reward = -0.05
        elif result is False:
            # Phantom pair or invalid drug
            step_reward = -0.3
        else:
            # Valid pair — calculate full reward
            step_reward = self.calculate_reward(
                key,
                action.get("severity", ""),
                action.get("suggested_action", ""),
            )
            # Add to flags raised
            self.flags_raised.append(FlagEntry(
                drug_a=action.get("drug_a", ""),
                drug_b=action.get("drug_b", ""),
                severity=action.get("severity", ""),
                suggested_action=action.get("suggested_action", ""),
                step=self.step_count + 1,
                reward_received=step_reward,
            ))

        self.episode_reward += step_reward
        self.step_count += 1

        # Check perfect completion
        if (self.perfectly_completed_pairs
                and self.perfectly_completed_pairs == self.ground_truth_keys):
            self.done = True
            return self._get_observation(), step_reward, True, self._get_state()

        # Check step budget
        if self.step_count >= self.max_steps:
            penalty = self._apply_termination_penalty()
            self.done = True
            return self._get_observation(), penalty, True, self._get_state()

        return self._get_observation(), step_reward, False, self._get_state()

    # ── termination penalty ─────────────────────────────────────────────────

    def _apply_termination_penalty(self) -> float:
        """Calculate severity-weighted penalty for unidentified pairs."""
        unidentified = self.ground_truth_keys - self.identified_pairs
        penalty = 0.0
        for key in unidentified:
            gt_severity = DRUG_INTERACTIONS[key]["severity"]
            if gt_severity == "severe":
                penalty -= 0.4
            elif gt_severity == "moderate":
                penalty -= 0.3
            else:
                penalty -= 0.2
        self.episode_reward += penalty
        return penalty

    # ── observation ─────────────────────────────────────────────────────────

    def _get_observation(self) -> dict:
        """Build the observation dict visible to the agent."""
        return {
            "patient_id": self.patient["patient_id"],
            "age": self.patient["age"],
            "conditions": self.patient["conditions"],
            "medications": self.patient["medications"],
            "flags_raised_so_far": [f.model_dump() for f in self.flags_raised],
            "steps_remaining": self.max_steps - self.step_count,
        }

    # ── state ───────────────────────────────────────────────────────────────

    def _get_state(self) -> dict:
        """Build the internal state dict (for grading/debugging)."""
        return {
            "patient_id": self.patient["patient_id"],
            "step_count": self.step_count,
            "task_level": self.task_level,
            "attempted_keys": [f"{k[0]}|{k[1]}" for k in self.attempted_keys],
            "identified_pairs": [f"{k[0]}|{k[1]}" for k in self.identified_pairs],
            "perfectly_completed_pairs": [f"{k[0]}|{k[1]}" for k in self.perfectly_completed_pairs],
            "ground_truth_keys": [f"{k[0]}|{k[1]}" for k in self.ground_truth_keys],
            "predictions": self.predictions,
            "done": self.done,
        }

    @property
    def state(self) -> dict:
        """Property accessor for state."""
        return self._get_state()

    # ── episode score ───────────────────────────────────────────────────────

    def get_episode_score(self) -> float:
        """Normalize episode reward to [0.0, 1.0]."""
        max_possible = len(self.ground_truth_keys) * 0.8
        if max_possible == 0:
            return 1.0
        return max(0.0, min(1.0, self.episode_reward / max_possible))
