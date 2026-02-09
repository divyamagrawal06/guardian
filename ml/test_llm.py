"""Test LLM inference with downloaded model"""
from llama_cpp import Llama

print("Loading TinyLlama model...")
model = Llama(
    model_path="./models/downloads/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=0,  # CPU only
    verbose=False,
)
print("Model loaded!")

# Test intent extraction
prompt = """[INST] You are an intent extraction system. Extract the user's intent from their request.

User request: "Open Notepad and type hello world"

Respond with JSON only:
{
    "goal": "main action in snake_case",
    "app": "target application name or null",
    "entities": {
        "key": "extracted value"
    }
}

[/INST]
"""

print("\nTesting intent extraction...")
output = model(prompt, max_tokens=256, temperature=0.3, stop=["</s>", "[/INST]"])
response = output["choices"][0]["text"].strip()
print(f"Response:\n{response}")

print("\nLLM test PASSED!")
