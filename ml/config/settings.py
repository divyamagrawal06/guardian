"""
ATLAS ML Pipeline - Configuration Settings
==========================================

Central configuration for all pipeline components.
"""

from pydantic import BaseModel
from typing import Optional
import os


class OCRConfig(BaseModel):
    """PaddleOCR configuration."""
    use_gpu: bool = True
    lang: str = "en"
    use_angle_cls: bool = True
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    rec_batch_num: int = 6


class VLMConfig(BaseModel):
    """LLaVA Vision-Language Model configuration."""
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    quantization: str = "4bit"  # Options: "4bit", "8bit", "none"
    max_new_tokens: int = 512
    temperature: float = 0.2
    device: str = "cuda"


class LLMConfig(BaseModel):
    """Local LLM (Mistral/Phi) configuration."""
    model_path: str = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    n_ctx: int = 4096  # Context window
    n_gpu_layers: int = -1  # -1 = use all GPU layers
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
