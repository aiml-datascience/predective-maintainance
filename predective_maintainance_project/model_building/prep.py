# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Handle Hugging Face token from Colab or environment
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    print("✅ Loaded HF token from Colab userdata")
except ModuleNotFoundError:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        print("✅ Loaded HF token from environment variable")
    else:
        print("⚠️ No HF token found — please set HF_TOKEN as an environment variable.")


api = HfApi(token=HF_TOKEN)
DATASET_PATH = "hf://datasets/sasipriyank/predectivemlops/engine_data.csv" # Corrected dataset path
engine_dataset = pd.read_csv(DATASET_PATH) # Updated variable name
print("Dataset loaded successfully.")
# Define the target variable for the classification task
target = 'Engine Condition'

# List of numerical features in the dataset
numeric_features = [
    'Lub oil pressure',               # Lub oil pressure
    'Fuel pressure',           # Fuel pressure
    'Coolant pressure', # Coolant Pressure
    'lub oil temp',    # Lub oil temp
    'Coolant temp',    # coolant Temp.
    'Engine rpm'    # Engine Rpm
 ]

# Define predictor matrix (X) using selected numeric and categorical features
X = engine_dataset[numeric_features] # Updated variable name

# Define target variable
y = engine_dataset[target] # Updated variable name


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sasipriyank/predectivemlops",
        repo_type="dataset",
    )
