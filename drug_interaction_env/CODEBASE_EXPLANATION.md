# Drug Interaction Environment — Complete Codebase Explanation

## 📋 Executive Summary

This is a **clinical pharmacology RL environment** built on OpenEnv that trains and evaluates LLM agents on their ability to identify dangerous **drug-drug interactions** from patient medication lists. The agent acts as a clinical pharmacist, flagging pairs of drugs that interact dangerously, then receives rewards based on accuracy of the severity classification and recommended clinical action.

**Framework**: OpenEnv (OpenAI-style environment interface) + FastAPI server + LLM inference client

---

## 🏗️ Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER (app.py)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ /reset      │  │  /step       │  │ /state                 │ │
│  │ Initialize  │  │ Process      │  │ Get episode internal   │ │
│  │ new episode │  │ LLM action   │  │ state (for grading)    │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
│           ▲                ▲                    ▲                │
│           │                │                    │                │
│  ┌────────▼────────────────▼────────────────────▼──────────┐   │
│  │   DrugInteractionEnvironment (drug_interaction_env.py) │   │
│  │                                                          │   │
│  │  • reset(task_level)     → Initialize episode          │   │
│  │  • step(action)          → Process one action          │   │
│  │  • validate(action)      → Check action validity       │   │
│  │  • calculate_reward()    → Score the action            │   │
│  │  • state / observation   → Multi-level state tracking  │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ▲                ▲                    ▲                │
│           │                │                    │                │
│  ┌────────┴─────────┬──────┴──────────┬────────┴─────────────┐ │
│  │ PATIENTS         │ DRUG_DATABASE   │ MODELS (Pydantic)    │ │
│  │ ├─ easy (1 int)  │ 35 interactions │ ├─ Action            │ │
│  │ ├─ medium (3)    │ 12 severe       │ ├─ Observation       │ │
│  │ └─ hard (5)      │ 13 moderate     │ ├─ State             │ │
│  │                  │ 10 mild         │ └─ PredictionEntry   │ │
│  └─────────────────┴─────────────────┴──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
        ▲                                          ▲
        │                                          │
        │ HTTP (JSON)                              │
        │                                          │
        │ LLM Client (inference.py)                │
        │ • OpenAI API calls                       │
        │ • Structured system prompt               │
        │ • Multi-turn conversation loop           │
        │ • Episode grading (grader.py)            │
        │                                          │
        └────────────────────────────────────────────
```

---

## 🔧 Core Components

### 1. **DrugInteractionEnvironment** (`drug_interaction_environment.py`)

The main RL environment implementing the OpenAI Gym-like interface:

#### **State Variables** (initialized in `__init__`)
```python
self.task_level: str          # "easy", "medium", or "hard"
self.patient: dict            # Patient profile (age, conditions, meds)
self.ground_truth_keys: set   # Canonical interaction pairs to find
self.attempted_keys: set      # All pairs agent has tried flagging
self.identified_pairs: set    # Correct pairs identified by agent
self.perfectly_completed: set # Pairs with perfect severity + action
self.predictions: dict        # Detailed prediction records
self.flags_raised: list       # Log of all actions taken
self.step_count: int          # Current step number
self.max_steps: int           # Episode step budget
self.episode_reward: float    # Cumulative reward
self.done: bool               # Episode terminal state
```

#### **Key Methods**

**`reset(task_level: str) → dict`**
- Loads a patient scenario from `PATIENTS[task_level]`
- Derives `ground_truth_keys` by scanning all medication pairs against `DRUG_INTERACTIONS`
- Sets `max_steps = len(ground_truth_keys) * 3` (generous budget)
- Returns initial observation dict

**`validate(action: dict) → bool | None`**
- Returns `True` if pair is in DRUG_INTERACTIONS (valid)
- Returns `False` if drug names invalid OR pair not in database (phantom)
- Returns `None` if pair already attempted (duplicate)
- **Normalization**: `key = tuple(sorted([drug_a.lower(), drug_b.lower()]))`

**`calculate_reward(key, severity, action) → float`**
Reward breakdown:
| Component | Score | Condition |
|----|----|----|
| **Base (pair identified)** | +0.4 | Always for valid pair |
| **Severity match** | +0.2 | Predicted == ground truth |
| **Severity mismatch (GT severe)** | -0.2 | Predicted ≠ severe (when GT is severe) |
| **Severity mismatch (GT moderate)** | -0.1 | Predicted ≠ moderate |
| **Severity mismatch (GT mild)** | -0.05 | Predicted ≠ mild |
| **Action match** | +0.2 | Predicted action == ground truth |
| **Action mismatch** | -0.1 | Predicted action ≠ ground truth |

**Perfect reward**: 0.4 + 0.2 + 0.2 = **0.8** (when severity and action both correct)

**`step(action: dict) → (observation, reward, done, state)`**
Main game loop:
1. If action is `DONE`, apply termination penalty and end episode
2. Validate the flagged pair
3. If invalid/phantom: reward = -0.3
4. If duplicate: reward = -0.05
5. If valid: calculate full reward + add to flags_raised
6. Update episode_reward
7. **Early termination**: If perfectly_completed_pairs == ground_truth_keys, set done=True
8. **Budget exhaustion**: If step_count >= max_steps, apply penalty and end
9. Return observation (what agent sees), step reward, done flag, internal state

**`_apply_termination_penalty() → float`**
Penalizes unidentified interactions by severity:
- Severe unidentified: -0.4 each
- Moderate unidentified: -0.3 each
- Mild unidentified: -0.2 each

---

### 2. **Drug Database** (`drug_database.py`)

Hardcoded dictionary of **35 real FDA drug interactions**:

```python
DRUG_INTERACTIONS: dict[tuple[str, str], dict] = {
    ("aspirin", "warfarin"): {
        "severity": "severe",
        "action": "replace_drug",
        "explanation": "Increased bleeding risk — dual antiplatelet + anticoagulant"
    },
    # ... 34 more entries
}
```

**Distribution**:
- 12 severe (high clinical risk)
- 13 moderate (medium risk, requires monitoring)
- 10 mild (low risk, mostly monitoring)

**Key Lookup** is always deterministic:
```python
key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
interaction = DRUG_INTERACTIONS[key]
```

---

### 3. **Patient Scenarios** (`patients.py`)

Three difficulty levels with carefully curated medication lists:

| Level | Meds | Interactions | Structure |
|----|----|----|----|
| **easy** | 6 | 1 (severe) | Simple starting point |
| **medium** | 10 | 3 (1 severe, 1 moderate, 1 mild) | Curriculum step |
| **hard** | 15 | 5 (mixed severity, avoiding redundant pairs) | Full complexity |

**Critical Design**: Medications are chosen such that **only the intended interactions fire**. "Filler" drugs (losartan, gabapentin, acetaminophen, etc.) exist in the medication list but don't interact with anything else in that patient's list.

Example (easy):
```python
"easy": {
    "medications": ["warfarin", "aspirin", "losartan", "amlodipine", "gabapentin", "pantoprazole"],
    # Only interaction: (aspirin, warfarin) → severe/replace_drug
    # "losartan", "amlodipine" etc. = safe filler drugs for that combination
}
```

**Ground Truth Derivation** (runtime):
```python
meds = patient["medications"]
for a, b in combinations(meds, 2):
    key = tuple(sorted([a.lower(), b.lower()]))
    if key in DRUG_INTERACTIONS:
        ground_truth_keys.add(key)  # This is the true positive set
```

---

### 4. **Pydantic Models** (`models.py`)

Structured data models for serialization:

**`DrugInteractionAction`** (agent output):
```python
action_type: "flag_interaction" | "DONE"
drug_a: str (optional, only for flag_interaction)
drug_b: str (optional, only for flag_interaction)
severity: "mild" | "moderate" | "severe"
suggested_action: "monitor" | "reduce_dose" | "replace_drug"
```

**`DrugInteractionObservation`** (what agent sees each step):
```python
patient_id: str
age: int
conditions: list[str]
medications: list[str]
flags_raised_so_far: list[FlagEntry]  # History of all flags so far
steps_remaining: int
```

**`PredictionEntry`** (stored for each interaction flagged):
```python
key: str  # "('drug_a', 'drug_b')"
predicted_severity: str
predicted_action: str
ground_truth_severity: str
ground_truth_action: str
reward_received: float
perfectly_completed: bool
```

**`DrugInteractionState`** (full internal state for grading):
```python
patient_id: str
step_count: int
task_level: str
attempted_keys: list[str]
identified_pairs: list[str]
perfectly_completed_pairs: list[str]
predictions: dict
done: bool
```

---

## 🎮 RL Environment Design: The Game Loop

### **Episode Lifecycle**

```
┌─── RESET ───────────────────────────────────────────┐
│ Client calls: POST /reset {"task_level": "easy"}    │
│ Environment:                                         │
│  • Loads patient scenario & meds                    │
│  • Derives ground_truth (C(n,2) vs DRUG_DATABASE)  │
│  • Sets max_steps = |ground_truth| * 3              │
│  • Returns: initial observation                     │
└──────────────────────────────────────────────────────┘
                        ▼
┌─── STEP LOOP ────────────────────────────────────────┐
│ Repeats until done=True:                             │
│                                                      │
│  1. Agent receives observation (patient + history) │
│  2. Agent outputs JSON action via LLM              │
│  3. POST /step with action                         │
│  4. Environment validates & scores action:         │
│     - validate() → True/False/None                 │
│     - calculate_reward() if True                   │
│     - Check termination conditions                 │
│  5. Return (obs, reward, done, state)              │
│                                                      │
│  Termination conditions:                            │
│  • Agent sends DONE action                         │
│  • All pairs perfectly identified                  │
│  • Step budget exhausted (step_count >= max)       │
└──────────────────────────────────────────────────────┘
                        ▼
┌─── GRADING ───────────────────────────────────────┐
│ Client calls: GET /state → final state dict        │
│ Grader (grader.py) computes normalized score:     │
│                                                    │
│ total_reward = sum(all step rewards)              │
│              - penalties for unidentified pairs   │
│ max_possible = |ground_truth| * 0.8               │
│ final_score = clamp(total_reward / max_possible)  │
│ Range: [0.0, 1.0]                                 │
└────────────────────────────────────────────────────┘
```

### **Action Processing Flow**

```
Agent sends: {"action_type": "flag_interaction", "drug_a": "...", "drug_b": "...", "severity": "...", "suggested_action": "..."}
                              ▼
                   validate(action)
                    /    |      \
              True /     |       \ False/None
                 /       |         \
            Valid  Duplicate(None)  Invalid
             Pair     (-0.05)        (-0.3)
              |
         calculate_reward()
              |
         Base: +0.4
         + Severity match: +0.2 or -(0.05-0.2)
         + Action match: +0.2 or -0.1
              |
         Step reward ∈ [-0.3, +0.8]
              |
         Update:
         • episode_reward += step_reward
         • identified_pairs.add(key) if valid
         • perfectly_completed_pairs.add(key) if perfect
         • predictions[key] = {...}
         • flags_raised_so_far.append(...)
              |
         Check termination:
         • perfectly_completed? → done=True
         • max_steps reached? → apply penalty, done=True
         • else: done=False, continue loop
```

---

## 🏥 Core Assumptions

### **1. Environment Setup Assumptions**

| Assumption | Impact | Justification |
|----|----|---|
| **Medications are case-insensitive** | All lookups normalized to lowercase | Universal clinical practice |
| **Pair order doesn't matter** | Always sort to (min, max) tuple | Symmetric interaction property |
| **Only one patient per episode** | Simplified state management | Focus on single task |
| **No drug quantity/dosage in state** | Simplified action space | Agent learns interaction existence, not dosing |
| **Interactions are static** | DRUG_INTERACTIONS never changes | Real-world drugs have fixed interactions |

### **2. Reward Structure Assumptions**

| Assumption | Impact | Consequence |
|----|----|---|
| **Base reward (0.4) for identifying pair** | Incentivizes finding all pairs | Agent must explore medication combinations |
| **Severity scoring is asymmetric** | Severe misclassification: -0.2, mild: -0.05 | Penalizes underestimating serious interactions |
| **Action scoring is symmetric** | Right action: +0.2, wrong: -0.1 | Encourages correct clinical judgment |
| **Perfect completion ends episode immediately** | No further steps allowed | Avoids unnecessary exploration noise |
| **Step budget = 3x ground truth** | Allows ~3 wrong attempts per interaction | Reasonable exploration allowance |

### **3. Ground Truth Derivation Assumptions**

| Assumption | Implication |
|----|---|
| **Ground truth = all C(n,2) pairs in DRUG_INTERACTIONS** | Any pair not in DB is a phantom (invalid) |
| **Patient medication list is exhaustive** | Only medications in list can be checked |
| **Drug names match exactly** | "ibuprofen" ≠ "Ibuprofen" initially; case-normalized |
| **No active metabolite interactions** | Only direct drug-drug interactions considered |
| **Interactions are bidirectional** | (A, B) = (B, A) after sorting |

### **4. Task Design Assumptions**

| Level | # Pairs | Assumption | Implication |
|----|----|----|----|
| **easy** | 1 | Simple pattern recognition | Agent learns basic interaction flagging |
| **medium** | 3 | Curriculum learning | Diverse severity exposure (1 each level) |
| **hard** | 5 | Complex search | Real drug cocktail complexity |

**Assumption**: Medications in hard don't create phantom interactions. (e.g., hard patient excludes warfarin+amiodarone even though both exist, to avoid unintended pair count.)

### **5. Episode Termination Assumptions**

| Condition | Reward | Assumption |
|----|----|----|
| **Agent sends DONE** | Termination penalty only | Assumes agent honestly signals completion |
| **Perfect completion** | No additional penalty | Assumes agent explores optimally |
| **Max steps reached** | Termination penalty applied | Assumes agent ran out of budget |

**Penalty Logic** (severity-weighted):
- Severe missed: -0.4 per pair
- Moderate missed: -0.3 per pair
- Mild missed: -0.2 per pair

### **6. LLM Agent Assumptions** (`inference.py`)

| Assumption | Impact |
|----|---|
| **LLM can output valid JSON** | System prompt strongly constrains output format |
| **LLM respects medication names** | "Exactly as spelled in medications list" |
| **LLM has pharmacological knowledge** | Can predict severity + action correctly |
| **LLM follows one-pair-per-step rule** | Won't flag multiple pairs in single action |
| **LLM's reasoning aligns with clinical guidelines** | FDA + pharmacology references used |

---

## 📡 API Interface

### **FastAPI Endpoints** (`app.py`)

```python
POST /reset
├─ Request:  {"task_level": "easy|medium|hard"}
└─ Response: {
    "patient_id": "P001",
    "age": 67,
    "conditions": ["hypertension", "type2_diabetes"],
    "medications": ["warfarin", "aspirin", ...],
    "flags_raised_so_far": [],
    "steps_remaining": 3
}

POST /step
├─ Request: {
│  "action_type": "flag_interaction|DONE",
│  "drug_a": "warfarin",
│  "drug_b": "aspirin",
│  "severity": "severe",
│  "suggested_action": "replace_drug"
}
└─ Response: {
    "observation": {...},
    "reward": 0.8,
    "done": false,
    "state": {...}
}

GET /state
├─ Returns internal state (for grading/debugging)
└─ Includes: episode_score, predictions, attempted_keys, etc.

GET /health
└─ Returns: {"status": "ok"}
```

---

## 🎯 Episode Grading

### **grader.py Logic**

```python
def grade_episode(task_level: str, final_state: dict) -> float:
    # 1. Get expected # of true pairs for this level
    n_true_pairs = TASK_TRUE_PAIRS[task_level]  # 1, 3, or 5
    max_possible = n_true_pairs * 0.8  # Maximum achievable reward
    
    # 2. Sum all step rewards
    total_reward = sum(p["reward_received"] for p in predictions.values())
    
    # 3. Apply termination penalties for unidentified pairs
    unidentified = ground_truth_keys - identified_pairs
    for key in unidentified:
        severity = DRUG_INTERACTIONS[key]["severity"]
        if severity == "severe":
            total_reward -= 0.4
        elif severity == "moderate":
            total_reward -= 0.3
        else:
            total_reward -= 0.2
    
    # 4. Normalize to [0.0, 1.0]
    return max(0.0, min(1.0, total_reward / max_possible))
```

**Score Interpretation**:
- **1.0**: All interactions identified perfectly (severity + action both correct)
- **0.8**: All interactions identified but some severity/action misclassifications
- **0.0-0.4**: Some interactions missed or significant misclassifications
- **0.0**: All interactions missed OR severe penalties apply

---

## 🔄 Inference Loop

### **inference.py Execution Flow**

```python
for task_level in ["easy", "medium", "hard"]:
    # 1. Reset environment
    observation = env_reset(task_level)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(observation)}
    ]
    
    # 2. Multi-turn conversation loop
    while not done:
        # LLM generates next action
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,  # Low temperature for deterministic output
            max_tokens=256
        )
        
        # Parse JSON action from response
        action = parse_llm_action(response.text)
        
        # Submit to environment
        result = env_step(action)
        reward = result["reward"]
        done = result["done"]
        observation = result["observation"]
        
        # Log step
        print(f"[STEP] step={step_num} action={action_type} reward={reward}")
        
        # Update conversation for next turn
        messages.append({"role": "assistant", "content": json.dumps(action)})
        if not done:
            messages.append({
                "role": "user",
                "content": build_user_message(observation)
            })
    
    # 3. Grade episode
    state_data = env_state()
    episode_score = grade_episode(task_level, state_data)
    print(f"[END] task={task_level} patient_id={patient_id} episode_score={episode_score}")
```

**Key Points**:
- **System prompt** constrains LLM to JSON-only output
- **Temperature = 0.1** for deterministic behavior
- **Multi-turn**: Each LLM response gets added to message history
- **Observation updates**: After each step, agent sees updated flags_raised_so_far
- **Episode score** computed independently by grader module

---

## 🧪 Testing Strategy

### **Unit Tests** (`test_env.py`)

Tests environment mechanics:
1. **Validation tests**: Valid pairs, invalid drugs, phantom pairs, duplicates
2. **Reward calculation**: Perfect flag (0.8), severity mismatch, action mismatch
3. **Episode completion**: Easy, medium, hard scenarios
4. **Termination penalties**: DONE action with unidentified pairs

### **API Tests** (`test_app.py`)

Tests FastAPI endpoints:
1. Health check (GET /health)
2. Reset endpoint (POST /reset)
3. Step endpoint (POST /step with valid action)
4. State endpoint (GET /state)

---

## 📊 Example Episode Trace

```
[START] task=easy patient_id=P001

RESET
→ Medications: [warfarin, aspirin, losartan, amlodipine, gabapentin, pantoprazole]
→ Ground truth: [(aspirin, warfarin)]
→ Max steps: 3
→ Initial observation returned

STEP 1
Agent action: {"action_type": "flag_interaction", "drug_a": "warfarin", "drug_b": "aspirin", "severity": "severe", "suggested_action": "replace_drug"}
Validation: ✓ Valid pair in DRUG_INTERACTIONS
Reward: 0.4 (base) + 0.2 (severity match) + 0.2 (action match) = 0.8
Perfectly completed? Yes → done=True
[STEP] step=1 action=flag_interaction drug_a=warfarin drug_b=aspirin severity=severe suggested_action=replace_drug reward=0.8

[END] task=easy patient_id=P001 episode_score=1.0

GRADING
total_reward = 0.8
unidentified = {} (empty)
max_possible = 1 * 0.8 = 0.8
episode_score = 0.8 / 0.8 = 1.0 ✓
```

---

## 🎓 Key Takeaways

| Component | Role | Critical Assumption |
|----|----|---|
| **Environment** | Manages episode state, validates actions, scores | Interactions are deterministic, pair order irrelevant |
| **Database** | Source of truth for drug interactions | Real FDA interactions, static throughout episode |
| **Reward** | Learning signal | Asymmetric severity penalties, base score for identification |
| **Task design** | Curriculum learning | Carefully curated medication lists (no phantom pairs) |
| **Grading** | Evaluate agent performance | Penalty-weighted score normalized to [0, 1] |
| **LLM Interface** | Agent-environment interaction | JSON-constrained output, structured prompting |

---

## 📚 Dependencies

```python
fastapi >= 0.100.0       # API framework
uvicorn >= 0.23.0       # ASGI server
pydantic >= 2.0         # Data validation
openai >= 1.0.0         # LLM client
huggingface_hub         # HuggingFace deployment
python >= 3.10          # Required for type hints
```

---

## 🚀 Deployment

- **Docker**: Self-contained image with FastAPI server
- **HuggingFace Spaces**: OpenEnv-compatible deployment via `hf_deploy.py`
- **Local dev**: `uvicorn drug_interaction_env.server.app:app --reload`

