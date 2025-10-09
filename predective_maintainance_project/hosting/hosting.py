from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

repo_id = "sasipriyank/predective-mlops"
repo_type = "space"  # Use "model" if it's a model repo

# Ensure repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' exists.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

# Now upload the folder
api.upload_folder(
    folder_path="predective_maintainance_project/deployment",
    path_in_repo=".",
    repo_id=repo_id,
    repo_type=repo_type
)
print("✅ Folder uploaded successfully")
