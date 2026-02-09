"""
ATLAS MCP Agent — Configuration
================================

Loads settings from environment variables or .env file.
Supports three LLM backends:
  1. Gemini API (Google's Generative AI — primary)
  2. OpenAI-compatible API (OpenAI, Groq, Together, local vLLM/Ollama, etc.)
  3. Local llama.cpp model (fully offline)

Set LLM_BACKEND=gemini, openai, or llama in your .env file.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv

# Load .env from the mcp folder itself
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


class GeminiConfig(BaseModel):
    """Config for Google Gemini API backend."""
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    temperature: float = 0.3
    max_output_tokens: int = 8192


class OpenAIConfig(BaseModel):
    """Config for OpenAI-compatible API backend."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", None))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = 0.3
    max_tokens: int = 4096


class LlamaConfig(BaseModel):
    """Config for local llama-cpp-python backend."""
    model_path: str = Field(
        default_factory=lambda: os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path(__file__).parent / "models" / "model.gguf")
        )
    )
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    temperature: float = 0.3
    max_tokens: int = 4096


class MCPConfig(BaseModel):
    """Master config for the MCP agent."""
    llm_backend: Literal["gemini", "openai", "llama"] = Field(
        default_factory=lambda: os.getenv("LLM_BACKEND", "gemini")
    )
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    llama: LlamaConfig = Field(default_factory=LlamaConfig)
    
    # Playwright MCP server settings
    playwright_headless: bool = Field(
        default_factory=lambda: os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() == "true"
    )
    
    # Agent settings
    max_steps: int = 30          # Max actions before giving up
    max_retries: int = 3         # Retries per failed action
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )


def get_config() -> MCPConfig:
    """Load and return the config (reads .env on first call)."""
    return MCPConfig()
