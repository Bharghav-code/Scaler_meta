"""Typed Pydantic models for the Drug Interaction Checker environment."""

from pydantic import BaseModel
from typing import Literal, Optional


# ── Action ──────────────────────────────────────────────────────────────────

class DrugInteractionAction(BaseModel):
    """Agent action: either flag an interaction or declare DONE."""
    action_type: Literal["flag_interaction", "DONE"]
    drug_a: Optional[str] = None
    drug_b: Optional[str] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    suggested_action: Optional[Literal["monitor", "reduce_dose", "replace_drug"]] = None


# ── Observation ─────────────────────────────────────────────────────────────

class FlagEntry(BaseModel):
    """A single flag raised by the agent during the episode."""
    drug_a: str
    drug_b: str
    severity: str
    suggested_action: str
    step: int
    reward_received: float


class DrugInteractionObservation(BaseModel):
    """Observation returned to the agent at each step."""
    patient_id: str
    age: int
    conditions: list[str]
    medications: list[str]
    flags_raised_so_far: list[FlagEntry]
    steps_remaining: int


# ── State ───────────────────────────────────────────────────────────────────

class PredictionEntry(BaseModel):
    """Detailed prediction record for a single drug pair."""
    key: str  # normalized sorted key as string, e.g. "('aspirin', 'warfarin')"
    predicted_severity: str
    predicted_action: str
    ground_truth_severity: str
    ground_truth_action: str
    reward_received: float
    perfectly_completed: bool


class DrugInteractionState(BaseModel):
    """Full internal state of the environment (for grading / debugging)."""
    patient_id: str
    step_count: int
    task_level: Literal["easy", "medium", "hard"]
    attempted_keys: list[str]
    identified_pairs: list[str]
    perfectly_completed_pairs: list[str]
    predictions: dict
    done: bool
