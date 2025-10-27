# =========================
# Save & upload the model
# =========================
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os, joblib

# 1) Save the trained/best model (make sure this is AFTER grid_search.fit)
best_model = grid_search.best_estimator_

# Robust paths (relative to this file)
THIS_DIR = Path(__file__).resolve().parent                 # mlops/model_building
ARTIFACTS_DIR = THIS_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILENAME = "best_churn_model_v1.joblib"
MODEL_PATH = ARTIFACTS_DIR / MODEL_FILENAME

# Save
joblib.dump(best_model, MODEL_PATH)

# 2) Upload to your HF model repo
api = HfApi(token=os.getenv("HF_TOKEN"))

create_repo(
    repo_id="haneuris1/churn-model",
    repo_type="model",
    private=False,
    exist_ok=True,            # no error if it already exists
)

api.upload_file(
    path_or_fileobj=str(MODEL_PATH),   # absolute path
    path_in_repo=MODEL_FILENAME,       # name inside the repo
    repo_id="haneuris1/churn-model",
    repo_type="model",
    commit_message="Upload trained churn model",
)
