# !pip install huggingface_hub transformers

from huggingface_hub import login, HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Login to your account
# You will be prompted for your token, or use login("your_token_here")
# login()

# 2. Configuration
local_model_path = "../checkpoints_bertweet_large/checkpoint-52000"
repo_id = "RangerX/Bertweet_large"  # Change this to your desired name and username

# 3. Load the model and tokenizer from your local path
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 4. Push to Hub
# This creates the repo automatically if it doesn't exist
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print(f"Model successfully uploaded to: https://huggingface.co/{repo_id}")