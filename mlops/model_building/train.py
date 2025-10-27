from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os, joblib

# -------- paths (absolute, robust) --------
THIS_DIR = Path(__file__).resolve().parent                 # mlops/model_building
ARTIFACTS_DIR = THIS_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILENAME = "best_churn_model_v1.joblib"              # <- single canonical name
MODEL_PATH = ARTIFACTS_DIR / MODEL_FILENAME               # mlops/model_building/artifacts/best_churn_model_v1.joblib

# Save model
joblib.dump(best_model, MODEL_PATH)

# -------- upload to Hub --------
api = HfApi(token=os.getenv("HF_TOKEN"))

# (A) Upload to a MODEL repo (keeps models separate)
create_repo(repo_id="haneuris1/churn-model", repo_type="model", private=False, exist_ok=True)
api.upload_file(
    path_or_fileobj=str(MODEL_PATH),                       # absolute path
    path_in_repo=MODEL_FILENAME,                           # name in the model repo
    repo_id="haneuris1/churn-model",
    repo_type="model",
)

# (Optional B) ALSO push to your Space so the app can load locally without downloading
# create_repo(repo_id="haneuris1/Bank-Customer-Churn", repo_type="space", private=False, exist_ok=True, space_sdk="streamlit")
# api.upload_file(
#     path_or_fileobj=str(MODEL_PATH),
#     path_in_repo=f"src/models/{MODEL_FILENAME}",          # Space path
#     repo_id="haneuris1/Bank-Customer-Churn",
#     repo_type="space",
# )
