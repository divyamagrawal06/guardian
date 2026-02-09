"""
ATLAS ML Pipeline - Configuration Settings
==========================================

Central configuration for all pipeline components.
Supports both local models (llama.cpp) and Ollama.
"""

from pydantic import BaseModel
from typing import Optional, Literal
import os


class OCRConfig(BaseModel):
    """EasyOCR configuration."""
    use_gpu: bool = True  # GPU enabled for speed (was False for CPU fallback)
    lang: str = "en"
    min_confidence: float = 0.3  # Minimum confidence threshold


class OllamaConfig(BaseModel):
    """Ollama server configuration."""
    base_url: str = "http://localhost:11434"
    timeout: int = 120  # Request timeout in seconds


class VLMConfig(BaseModel):
    """Vision-Language Model configuration."""
    backend: Literal["ollama", "transformers"] = "ollama"
    
    # Ollama settings
    ollama_model: str = "llava:7b"  # or "llava:13b", "bakllava"
    
    # Transformers settings (fallback)
    hf_model_name: str = "llava-hf/llava-1.5-7b-hf"
    quantization: str = "4bit"
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.2


class LLMConfig(BaseModel):
    """Local LLM configuration."""
    backend: Literal["ollama", "llama_cpp"] = "ollama"
    
    # Ollama settings
    ollama_model: str = "mistral:7b"  # or "phi3", "llama3", etc.
    
    # llama.cpp settings (fallback)
    model_path: str = "./models/downloads/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = use all GPU layers (was 0 for CPU only)
    
    # Generation settings
    temperature: float = 0.3
    max_tokens: int = 1024


class ScreenConfig(BaseModel):
    """Screen capture and interaction configuration."""
    capture_monitor: int = 1  # Primary monitor
    dpi_scale: float = 1.0  # Will be auto-detected
    screenshot_format: str = "png"
    mouse_move_duration: float = 0.1  # Seconds
    typing_interval: float = 0.02  # Seconds between keystrokes


class VerificationConfig(BaseModel):
    """Visual verification configuration."""
    max_retries: int = 3
    verification_delay: float = 0.5  # Wait before verification screenshot
    confidence_threshold: float = 0.7
    iou_threshold: float = 0.5  # For bounding box fusion


class MemoryConfig(BaseModel):
    """Memory/persistence configuration."""
    enabled: bool = True
    db_path: str = "./data/memory.db"
    max_patterns: int = 1000


class PipelineConfig(BaseModel):
    """Master pipeline configuration."""
    ollama: OllamaConfig = OllamaConfig()
    ocr: OCRConfig = OCRConfig()
    vlm: VLMConfig = VLMConfig()
    llm: LLMConfig = LLMConfig()
    screen: ScreenConfig = ScreenConfig()
    verification: VerificationConfig = VerificationConfig()
    memory: MemoryConfig = MemoryConfig()
    
    # Global settings
    debug_mode: bool = True
    log_level: str = "INFO"
    save_screenshots: bool = True
    screenshots_dir: str = "./data/screenshots"


def load_config() -> PipelineConfig:
    """Load configuration from environment or defaults."""
    # TODO: Load from .env or config file
    return PipelineConfig()


# Global config instance
config = load_config()
