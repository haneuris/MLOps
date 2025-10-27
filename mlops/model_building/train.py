# =========================
# train.py (end-to-end)
# =========================
import os
from pathlib import Path

# Data / ML
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib

# HF Hub
from huggingface_hub import HfApi, create_repo

# -------------------------
# 1) Load data
# -------------------------
Xtrain_path = "hf://datasets/haneuris1/bank-customer-churn/Xtrain.csv"
Xtest_path  = "hf://datasets/haneuris1/bank-customer-churn/Xtest.csv"
ytrain_path = "hf://datasets/haneuris1/bank-customer-churn/ytrain.csv"
ytest_path  = "hf://datasets/haneuris1/bank-customer-churn/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze("columns")  # ensure Series
ytest  = pd.read_csv(ytest_path).squeeze("columns")   # ensure Series

# -------------------------
# 2) Preprocess + model
# -------------------------
numeric_features = [
    "CreditScore","Age","Tenure","Balance",
    "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"
]
categorical_features = ["Geography"]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Handle class imbalance
# (ratio of negatives to positives)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -------------------------
# 3) Hyperparameter grid
# -------------------------
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.6],
}

# -------------------------
# 4) GridSearchCV + fit
# -------------------------
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

# -------------------------
# 5) Best model
# -------------------------
best_model = grid_search.best_estimator_

# -------------------------
# 6) Evaluate
# -------------------------
classification_threshold = 0.45

y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

print("=== TRAIN REPORT ===")
print(classification_report(ytrain, y_pred_train))
print("\n=== TEST REPORT ===")
print(classification_report(ytest, y_pred_test))

# -------------------------
# 7) Save artifact
# -------------------------
THIS_DIR = Path(__file__).resolve().parent                 # mlops/model_building
ARTIFACTS_DIR = THIS_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILENAME = "best_churn_model_v1.joblib"
MODEL_PATH = ARTIFACTS_DIR / MODEL_FILENAME

joblib.dump(best_model, MODEL_PATH)

# -------------------------
# 8) Upload to HF model repo
# -------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

create_repo(
    repo_id="haneuris1/churn-model",
    repo_type="model",
    private=False,
    exist_ok=True,  # no error if exists
)

api.upload_file(
    path_or_fileobj=str(MODEL_PATH),   # absolute path
    path_in_repo=MODEL_FILENAME,       # name inside repo
    repo_id="haneuris1/churn-model",
    repo_type="model",
    commit_message="Upload trained churn model",
)
