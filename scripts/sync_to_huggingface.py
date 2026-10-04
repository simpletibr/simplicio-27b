import os
import glob
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    # Fallback to local config file or credentials
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            TOKEN = f.read().strip()

REPO_ID = "wesleysimplicio/Simplicio-27B"

if not TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set!")

api = HfApi(token=TOKEN)

print(f"Authenticating as: {api.whoami()['name']}")
print(f"Target repository: {REPO_ID}")

files_to_upload = [
    ("README.md", "README.md"),
    ("benchmark_simplicio_27b.py", "benchmark_simplicio_27b.py"),
    ("train_simplicio_27b.py", "train_simplicio_27b.py"),
    ("benchmarks/statistical_proof_n120.json", "benchmarks/statistical_proof_n120.json"),
    ("benchmarks/AUDIT_RESPONSE_AND_PROOF.md", "benchmarks/AUDIT_RESPONSE_AND_PROOF.md"),
    ("benchmarks/prove_benchmark_120.py", "benchmarks/prove_benchmark_120.py"),
]

for local_file, repo_path in files_to_upload:
    if os.path.exists(local_file):
        print(f"  -> Uploading {local_file} as {repo_path}...")
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            commit_message=f"docs: update {repo_path} with 6 architectural engineering adjustments"
        )

print("🎉 ALL FILES SUCCESSFULLY SYNCHRONIZED TO HUGGING FACE HUB!")
