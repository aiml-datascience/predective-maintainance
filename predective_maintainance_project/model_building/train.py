# predective_maintainance_project/model_building/train.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --------------------------
# Hugging Face token
# --------------------------
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    print("✅ Loaded HF token from Colab userdata")
except ModuleNotFoundError:
    HF_TOKEN = os.getenv("HF_TOKEN")
    print("✅ Loaded HF token from environment")

api = HfApi(token=HF_TOKEN)

# --------------------------
# MLflow setup
# --------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("predectivemlops")

# --------------------------
# Features
# --------------------------
numeric_features = [
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp',
    'Engine rpm'
]
target_column = 'Engine Condition'

# --------------------------
# Robust CSV loader
# --------------------------
def load_csv_safe(path, numeric_features, target=None):
    """
    Load CSV, remove any duplicate header row, strip whitespace from column names, convert numeric columns.
    """
    df = pd.read_csv(path, header=0)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Detect if first row is still the header (string in numeric column)
    first_row = df.iloc[0]
    for col in numeric_features:
        if col not in df.columns:
            continue
        try:
            float(first_row[col])
        except ValueError:
            df = df.iloc[1:].reset_index(drop=True)
            break

    # Convert numeric columns to float
    df[numeric_features] = df[numeric_features].astype(float)

    if target:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in CSV columns: {df.columns.tolist()}")
        y = df[target].astype(int)
        X = df.drop(columns=[target])
        return X, y

    return df

# --------------------------
# Load datasets
# --------------------------
Xtrain_path = "hf://datasets/sasipriyank/predectivemlops/Xtrain.csv"
Xtest_path = "hf://datasets/sasipriyank/predectivemlops/Xtest.csv"

Xtrain, ytrain = load_csv_safe(Xtrain_path, numeric_features, target=target_column)
Xtest, ytest = load_csv_safe(Xtest_path, numeric_features, target=target_column)

# --------------------------
# Handle class imbalance
# --------------------------
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# --------------------------
# Preprocessing and pipeline
# --------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# --------------------------
# Hyperparameter grid
# --------------------------
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__reg_lambda': [0.4, 0.5],
}

# --------------------------
# Train + Hyperparameter Tuning
# --------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    # Predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })

    # Save model locally
    model_path = "best_predective_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✅ Model saved at {model_path}")

    # --------------------------
    # Upload to Hugging Face
    # --------------------------
    repo_id = "sasipriyank/predectivemodel"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✅ Repo '{repo_id}' exists. Using it.")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"✅ Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type
    )
    print("✅ Model uploaded to Hugging Face Hub successfully.")
