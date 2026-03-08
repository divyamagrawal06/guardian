"""
ATLAS ML Pipeline - Model Loaders
=================================

This module handles loading and initializing all ML models:
- PaddleOCR for text detection
- LLaVA for visual understanding
- Gemini for reasoning and planning

Imports are lazy to avoid pulling in heavy deps (paddleocr, transformers)
when only the LLM is needed (e.g. backend server startup).
"""


def __getattr__(name: str):
    if name == "OCRModel":
        from .ocr_model import OCRModel
        return OCRModel
    if name == "VLMModel":
        from .vlm_model import VLMModel
        return VLMModel
    if name == "LLMModel":
        from .llm_model import LLMModel
        return LLMModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OCRModel", "VLMModel", "LLMModel"]
