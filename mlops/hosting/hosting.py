from pathlib import Path
import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Add it as a GitHub Actions secret.")

# ---- target repo (Space) ----
REPO_ID = "haneuris1/Bank-Customer-Churn"   # change if needed
REPO_TYPE = "space"                         # "space" | "model" | "dataset"
SPACE_SDK = "gradio"                        # required for spaces: gradio|streamlit|docker|static
PATH_IN_REPO = ""                           # optional subfolder path inside the repo

api = HfApi(token=HF_TOKEN)

# Create the Space if it doesn't exist (no-op if it already exists)
create_repo(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    private=False,
    exist_ok=True,          # <-- avoids errors if it already exists
    token=HF_TOKEN,
    space_sdk=SPACE_SDK if REPO_TYPE == "space" else None,
)

# Resolve deployment folder relative to this file:
# hosting.py is at mlops/hosting -> go up to mlops and into deployment
mlops_dir = Path(__file__).resolve().parents[1]
folder_path = (mlops_dir / "deployment").as_posix()

print(f"Uploading from: {folder_path}")
if not os.path.isdir(folder_path):
    raise FileNotFoundError(f"Provided path is not a directory: {folder_path}")

api.upload_folder(
    folder_path=folder_path,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    path_in_repo=PATH_IN_REPO,
    commit_message="Update Space files from GitHub Actions",
)
