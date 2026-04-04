from huggingface_hub import HfApi
import os

print("Starting upload to Hugging Face...")
api = HfApi()

try:
    api.upload_folder(
        folder_path=os.path.abspath("c:/Users/Laxmikant Joshi/Desktop/Scaler_meta/drug_interaction_env"),
        repo_id="HiberNET/drug-interaction-checker",
        repo_type="space"
    )
    print("\nSUCCESS! All files uploaded securely to the space.")
except Exception as e:
    print(f"\nERROR: {e}")
