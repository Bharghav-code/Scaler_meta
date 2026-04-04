---
title: Pharmaceutical Drug Interaction Checker
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Pharmaceutical Drug Interaction Checker — OpenEnv

This is a clinical pharmacist simulator built on the OpenEnv framework. It exposes a patient profile and a medication list to an LLM agent, evaluating its ability to correctly identify dangerous drug-drug interactions, classify severity, and recommend clinical actions.

## 🚀 Running Locally

Build the Docker Container:
```bash
docker build -t drug-interaction-env .
docker run -p 8000:8000 drug-interaction-env
```

## 🌐 Deploying to Hugging Face Spaces

This repository is strictly structured for an immediate Hugging Face Docker Space deployment via OpenEnv.

1. Ensure you have the Hugging Face CLI installed:
   ```bash
   pip install -U "huggingface_hub[cli]"
   ```
2. Login to your Hugging Face account (Requires Hugging Face Account Token from Settings > Access Tokens):
   ```bash
   huggingface-cli login
   ```
3. Use the OpenEnv CLI to push:
   ```bash
   openenv push --enable-interface
   ```

*(Alternatively, you can manually create a new Hugging Face Space on huggingface.co, select "Docker" as the space SDK, and push this entire repository directly to it!)*
