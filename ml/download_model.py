"""Download a small GGUF model for testing"""
from huggingface_hub import hf_hub_download
import os

# Create downloads directory
os.makedirs("models/downloads", exist_ok=True)

print("Downloading TinyLlama-1.1B GGUF model (~670MB)...")
print("This may take a few minutes...")

model_path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    local_dir="./models/downloads",
)

print(f"Model downloaded to: {model_path}")
print("Done!")
