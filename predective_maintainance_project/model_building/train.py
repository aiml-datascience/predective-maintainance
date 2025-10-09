# Data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model training, tuning, evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Model serialization and system
import joblib
import os

# Hugging Face API
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow
import mlflow

# ------------------------------
# Load HF token
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except ModuleNotFoundError:
    HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("predectivemlops")

# Dataset paths
Xtrain_path = "hf://datasets/sasipriyank/predectivemlops/Xtrain.csv"
Xtest_path  = "hf://datasets/sasipriyank/predectivemlops/Xtest.csv"
ytrain_path = "hf://datasets/sasipriyank/predectivemlops/ytrain.csv"
ytest_path  = "hf://datasets/sasipriyank/predectivemlops/ytest.csv"

# ------------------------------
# Load CSV with header and safe conversion
Xtrain = pd.read_csv(Xtrain_path, header=0)
Xtest  = pd.read_csv(Xtest_path, header=0)
ytrain = pd.read_csv(ytrain_path, header=0).iloc[:,0]
ytest  = pd.read_csv(ytest_path, header=0).iloc[:,0]

numeric_features = [
    'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 
    'lub oil temp', 'Coolant temp', 'Engine rpm'
]

# Convert numeric columns safely
for col in numeric_features:
    Xtrain[col] = pd.to_numeric(Xtrain[col], errors='coerce')
    Xtest[col]  = pd.to_numeric(Xtest[col], errors='coerce')

# Drop rows with NaN (if conversion fails)
Xtrain.dropna(inplace=True)
Xtest.dropna(inplace=True)
ytrain = ytrain[Xtrain.index].astype(int)
ytest  = ytest[Xtest.index].astype(int)

print("✅ Data loaded and converted successfully")

# ------------------------------
# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing
preprocessor = make_column_transformer((StandardScaler(), numeric_features))

# XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ------------------------------
# Train & log with MLflow
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    # Predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:,1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test_proba = best_model.predict_proba(Xtest)[:,1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save locally
    model_path = "best_predective_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✅ Model saved at {model_path}")

# ------------------------------
# Upload to Hugging Face
repo_id = "sasipriyank/predectivemodel"
repo_type = "model"
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type
)
print("✅ Model uploaded to Hugging Face successfully")
