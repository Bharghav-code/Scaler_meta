"""
inference.py — Baseline inference script for the Drug Interaction Checker.

Runs all 3 task levels (easy, medium, hard) against the environment server
using an LLM via the OpenAI client. Emits structured stdout logs:

    [START] task=easy patient_id=P001
    [STEP] step=1 action=flag_interaction drug_a=warfarin drug_b=aspirin severity=severe suggested_action=replace_drug reward=0.8
    [STEP] step=2 action=DONE reward=0.0
    [END] task=easy patient_id=P001 episode_score=1.0

Required environment variables:
    API_BASE_URL  — https://api-inference.huggingface.co/v1/
    MODEL_NAME    — meta-llama/Meta-Llama-3-8B-Instruct
    HF_TOKEN      — set via environment variable
"""

import os
import sys
import json
import requests

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from openai import OpenAI
from grader import grade_episode




ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.environ.get("API_BASE_URL","https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.environ.get("MODEL_NAME","meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")

TASK_LEVELS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a clinical pharmacist reviewing a patient's medication list for dangerous drug interactions.

For each step, output ONLY a valid JSON object in one of these two formats:
1. Flag an interaction: {"action_type": "flag_interaction", "drug_a": "...", "drug_b": "...", "severity": "mild|moderate|severe", "suggested_action": "monitor|reduce_dose|replace_drug"}
2. Declare done: {"action_type": "DONE"}

Rules:
- Flag only ONE pair per step
- Do not repeat pairs already in flags_raised_so_far
- Only use drug names that appear in the patient's medications list — EXACTLY as spelled
- Only flag pairs you are confident interact dangerously
- severity must be exactly: "mild", "moderate", or "severe"
- suggested_action must be exactly: "monitor", "reduce_dose", or "replace_drug"
- Send DONE when you have found all interactions
- Output ONLY the JSON object, no explanations or markdown"""


def env_reset(task_level: str) -> dict:
    """Call POST /reset on the environment server."""
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_level": task_level})
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    """Call POST /step on the environment server."""
    resp = requests.post(f"{ENV_BASE_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    """Call GET /state on the environment server."""
    resp = requests.get(f"{ENV_BASE_URL}/state")
    resp.raise_for_status()
    return resp.json()


def build_user_message(observation: dict) -> str:
    """Format the observation as a user prompt for the LLM."""
    return f"""Patient Profile:
- Patient ID: {observation['patient_id']}
- Age: {observation['age']}
- Conditions: {', '.join(observation['conditions'])}
- Medications: {', '.join(observation['medications'])}

Flags raised so far: {json.dumps(observation.get('flags_raised_so_far', []), indent=2)}
Steps remaining: {observation.get('steps_remaining', 'unknown')}

Analyze the medication list and flag the next drug interaction, or send DONE if all interactions have been found."""


def parse_llm_action(response_text: str) -> dict:
    """Extract a valid action JSON from the LLM response."""
    text = response_text.strip()

    # Try to find JSON in the response
    # Handle markdown code blocks
    if "```" in text:
        lines = text.split("```")
        for block in lines:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
    # Direct Parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Fallback
    print("[WARN] Could not parse LLM response, sending DONE as fallback", file=sys.stderr)
    return {"action_type": "DONE"}




def run_inference():
    """Run inference across all task levels."""
    if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
        print("[ERROR] Missing required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN", file=sys.stderr)
        print("[INFO] Running in LOCAL-ONLY mode (no LLM calls)", file=sys.stderr)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_level in TASK_LEVELS:
        # Reset environment
        observation = env_reset(task_level)
        patient_id = observation["patient_id"]
        print(f"[START] task={task_level} patient_id={patient_id}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(observation)},
        ]

        step_num = 0
        done = False

        while not done:
            step_num += 1

            try:
                # LLM Calling is done here
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=256,
                )
                llm_text = response.choices[0].message.content
                action = parse_llm_action(llm_text)
            except Exception as e:
                print(f"[WARN] LLM call failed: {e}, sending DONE", file=sys.stderr)
                action = {"action_type": "DONE"}

            # action is sent to the env
            try:
                result = env_step(action)
            except Exception as e:
                print(f"[ERROR] env_step failed: {e}", file=sys.stderr)
                break

            reward = result["reward"]
            done = result["done"]
            observation = result["observation"]

            # step is stored
            if action["action_type"] == "flag_interaction":
                print(
                    f"[STEP] step={step_num} action=flag_interaction"
                    f" drug_a={action.get('drug_a', '')}"
                    f" drug_b={action.get('drug_b', '')}"
                    f" severity={action.get('severity', '')}"
                    f" suggested_action={action.get('suggested_action', '')}"
                    f" reward={reward}"
                )
            else:
                print(f"[STEP] step={step_num} action=DONE reward={reward}")

            # Update conversation for next LLM call
            messages.append({"role": "assistant", "content": json.dumps(action)})
            if not done:
                messages.append({
                    "role": "user",
                    "content": build_user_message(observation),
                })

        state_data = env_state()

        episode_score = grade_episode(task_level, state_data)

        print(f"[END] task={task_level} patient_id={patient_id} episode_score={episode_score}")


if __name__ == "__main__":
    run_inference()
