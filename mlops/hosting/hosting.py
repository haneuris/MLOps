from pathlib import Path
import os
from huggingface_hub import HfApi, create_repo, HfHubHTTPError

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Add it as a GitHub Actions secret.")

REPO_ID = "haneuris1/Bank-Customer-Churn"   # <- your Space path
REPO_TYPE = "space"                         # we're creating a Space
SPACE_SDK = "gradio"                        # or: "streamlit" | "docker" | "static"
PATH_IN_REPO = ""                           # optional subfolder within the Space

api = HfApi(token=HF_TOKEN)

# Ensure the Space exists with the right SDK (create only if missing)
try:
    info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
except HfHubHTTPError as e:
    if e.response is not None and e.response.status_code == 404:
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            exist_ok=True,
            token=HF_TOKEN,
            space_sdk=SPACE_SDK,      # <-- REQUIRED for spaces
        )
    else:
        raise

# Resolve deployment folder relative to this file (mlops/hosting/hosting.py)
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
