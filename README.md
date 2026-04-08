# Pharmaceutical Drug Interaction Checker (OpenEnv Specification)
This project implements a Pharmaceutical Drug Interaction Checker designed to test the clinical reasoning and safety evaluations of Large Language Models (LLMs). Built strictly adhering to the [OpenEnv](https://github.com/openenv/openenv-core) framework specification, it provides an isolated, stateful environment where an AI agent acts as a clinical pharmacist reviewing patient medication lists.
## 🏗️ Architecture Overview
The codebase is split into two completely decoupled layers to strictly separate the environment from the agent:
1. **Environment Server (`drug_interaction_env/`)**:
   - A FastAPI application implementing the OpenEnv `/reset`, `/step`, and `/state` interface endpoints.
   - Houses the ground truth clinical rules (`drug_database.py`) and patient scenarios (`patients.py`).
   - Maintains the state machine, evaluates the agent's actions securely on the backend, and calculates step-by-step reward/penalties based on the accuracy of flagged interactions.
2. **Inference Agent (`inference.py`)**:
   - A client-side Python script leveraging the `openai` SDK to act as the agent.
   - Retrieves observations (patient profiles) from the environment and uses an LLM to identify interactions.
   - Logs its trajectory conforming strictly to the `[START]`, `[STEP]`, and `[END]` evaluation format.
   - Invokes `grader.py` at the conclusion of an episode to compute the final, normalized metric score (`[0.0, 1.0]`).
## ⚙️ Requirements & Setup
This project uses standard Python packaging. All environment dependencies are defined inside `drug_interaction_env/server/requirements.txt`.
### Step 1: Install Dependencies
Create and activate a virtual environment, then install the dependencies:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate
pip install -r drug_interaction_env/server/requirements.txt
```
### Step 2: Start the Environment Server
Before the agent can evaluate a patient, the OpenEnv FastAPI server must be running. We recommend running this in a separate terminal window:
```bash
cd drug_interaction_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
This boots the `/reset` and `/step` HTTP interfaces on `http://localhost:8000`.
## 🚀 Running the LLM Evaluation Loop
The agent relies on Hugging Face Serverless APIs (which support strictly compliant OpenAI interfaces) pointing at endpoints such as `Llama-3.1-8B-Instruct`.
Ensure you configure the variables **in a separate terminal window** (with the virtual environment activated) before starting inference.
**Windows (PowerShell):**
```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN = "your_actual_huggingface_token"
$env:ENV_BASE_URL = "http://localhost:8000"
python inference.py
```
**Linux/MacOS (Bash):**
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_actual_huggingface_token"
export ENV_BASE_URL="http://localhost:8000"
python inference.py
```
## 📊 Evaluation & Grader Logic
The environment ships with 3 carefully curated `task_level`s representing escalating difficulty:
- **Easy**: 6 drugs, 1 interaction.
- **Medium**: 10 drugs, 3 interactions.
- **Hard**: 15 drugs, 5 interactions.
### Scoring Schema:
Agents receive positive rewards for correctly identifying an interaction's severity (+w) and suggested action (+w). They face compounding penalties for:
- Claiming non-existent pairs (Hallucinations).
- Re-reporting pairs they have already identified.
- Terminating the episode while leaving interactions undetected in the patient's system.
Upon receiving the `{"action_type": "DONE"}` trigger or exceeding the step budget, the environment's `grader.py` produces the normalized final mathematical `episode_score`.
## 🐳 Docker (Target Deployment)
To deploy the environment server directly via Docker:
```bash
docker build -t drug-interaction-env -f drug_interaction_env/server/Dockerfile drug_interaction_env/
docker run -p 8000:8000 drug-interaction-env
```
Once the container boots, `inference.py` can be executed locally as outlined above.
## ✅ Validator Status
The codebase has been executed against the `openenv validate` pipeline and has returned `[OK] : Ready for multi-mode deployment`.
