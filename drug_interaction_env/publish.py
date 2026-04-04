import os
from huggingface_hub import HfApi

print("--- Hugging Face Secure Deploy ---")
print("We need to authenticate with a WRITE token to preserve your folder structure.")
print("If you didn't check the 'Write' permission when making your token earlier, go to huggingface.co/settings/tokens and make a new one with WRITE checked!")

token = input("\nPaste your Hugging Face WRITE token here: ").strip()

api = HfApi(token=token)

print("\nUploading files...")
try:
    # Use absolute path to guarantee it grabs everything securely
    target_path = os.path.abspath("c:/Users/Laxmikant Joshi/Desktop/Scaler_meta/drug_interaction_env")
    
    api.upload_folder(
        folder_path=target_path,
        repo_id="HiberNET/drug-interaction-checker",
        repo_type="space"
    )
    print("\n✅ SUCCESS! All folders and files pushed perfectly. Check your Space to watch it build!")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
