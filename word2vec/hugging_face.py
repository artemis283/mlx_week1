from huggingface_hub import create_repo, HfApi

# Replace with your Hugging Face username and the dataset name you want
repo_id = "artemisweb/test-dataset"

create_repo(repo_id, repo_type="dataset")

api = HfApi()

# Upload a file from your current directory (e.g., "mydata.txt")
api.upload_file(
    path_or_fileobj="text8_plus_hn2.txt",       # Local file path
    path_in_repo="mydata.txt",          # How it will appear in the repo
    repo_id=repo_id,
    repo_type="dataset"
)
