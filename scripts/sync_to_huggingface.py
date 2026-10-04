import os
import glob
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
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
    ("Simplicio_27B_2026_Benchmarks_Colab.ipynb", "Simplicio_27B_2026_Benchmarks_Colab.ipynb"),
    ("benchmark_simplicio_27b.py", "benchmark_simplicio_27b.py"),
    ("train_simplicio_27b.py", "train_simplicio_27b.py"),
    ("benchmarks/deepseek_v41_benchmark_results.json", "benchmarks/deepseek_v41_benchmark_results.json"),
    ("benchmarks/run_deepseek_v41_benchmarks.py", "benchmarks/run_deepseek_v41_benchmarks.py"),
    ("benchmarks/run_aider_benchmark.py", "benchmarks/run_aider_benchmark.py"),
    ("benchmarks/run_evalplus_humaneval.py", "benchmarks/run_evalplus_humaneval.py"),
    ("benchmarks/run_livecodebench.py", "benchmarks/run_livecodebench.py"),
    ("benchmarks/run_swebench_eval.py", "benchmarks/run_swebench_eval.py"),
    ("benchmarks/statistical_proof_n120.json", "benchmarks/statistical_proof_n120.json"),
    ("benchmarks/AUDIT_RESPONSE_AND_PROOF.md", "benchmarks/AUDIT_RESPONSE_AND_PROOF.md"),
    ("benchmarks/prove_benchmark_120.py", "benchmarks/prove_benchmark_120.py"),
    ("benchmarks/compare_top10_2026.py", "benchmarks/compare_top10_2026.py"),
]

# Assets (SVGs and logos)
for asset_path in glob.glob("assets/*"):
    if os.path.isfile(asset_path):
        rel = os.path.basename(asset_path)
        files_to_upload.append((asset_path, f"assets/{rel}"))

print(f"Uploading {len(files_to_upload)} files to Hugging Face...")
for local_file, repo_path in files_to_upload:
    if os.path.exists(local_file):
        print(f"  -> Uploading {local_file} as {repo_path}...")
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            commit_message=f"docs: sync {repo_path} with updated benchmark numbers"
        )

print("🎉 ALL FILES SUCCESSFULLY SYNCHRONIZED TO HUGGING FACE HUB!")
