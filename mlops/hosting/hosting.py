from pathlib import Path
import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Add it as a GitHub Actions secret.")

api = HfApi(token=HF_TOKEN)

# ---- Configure your target Space/repo ----
REPO_ID = "haneuris1/Bank-Customer-Churn"   # <-- change if needed
REPO_TYPE = "space"                         # or "model" / "dataset"
PATH_IN_REPO = ""                           # optional subfolder in the repo

# Ensure the repo exists (no-op if it already exists)
create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True, token=HF_TOKEN)

# Resolve deployment folder relative to this file:
# hosting.py is in mlops/hosting → go up to mlops/ and then into deployment/
mlops_dir = Path(__file__).resolve().parents[1]
folder_path = (mlops_dir / "deployment").as_posix()

print(f"Uploading from: {folder_path}")
if not os.path.isdir(folder_path):
    raise FileNotFoundError(f"Provided path is not a directory: {folder_path}")

# Upload
api.upload_folder(
    folder_path=folder_path,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    path_in_repo=PATH_IN_REPO,
    commit_message="Update Space files from GitHub Actions",
)
