"""
ATLAS ML Pipeline - Perception Engine
======================================
Orchestrates OCR, VLM, and fusion for complete screen understanding.
PIPELINE STEPS 3-6 combined.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np
from loguru import logger

from models import OCRModel, VLMModel
from perception.screen_capture import ScreenCapture, ScreenFrame
from perception.bbox_fusion import BoundingBoxFusion, FusedElement


@dataclass
class PerceptionResult:
    """Complete perception result for one screen capture."""
    frame: ScreenFrame
    ocr_results: List[Dict[str, Any]]
    vlm_regions: List[Dict[str, Any]]
    fused_elements: List[FusedElement]
    screen_description: str
    
    def get_element_by_role(self, role: str) -> Optional[FusedElement]:
        for e in self.fused_elements:
            if e.role == role:
                return e
        return None
    
    def get_element_by_text(self, text: str, fuzzy: bool = True) -> Optional[FusedElement]:
        for e in self.fused_elements:
            if e.text:
                if fuzzy and text.lower() in e.text.lower():
                    return e
                elif e.text.lower() == text.lower():
                    return e
        return None


class PerceptionEngine:
    """
    Complete perception pipeline combining screen capture, OCR, VLM, and fusion.
    
    Usage:
        engine = PerceptionEngine()
        result = engine.perceive()
        element = result.get_element_by_text("Search")
    """
    
    def __init__(self):
        self.screen_capture = ScreenCapture()
        self.ocr = OCRModel()
        self.vlm = VLMModel()
        self.fusion = BoundingBoxFusion()
        self._initialized = False
        
    def initialize(self) -> None:
        """Load all models. Call once at startup."""
        if self._initialized:
            return
        logger.info("Initializing perception engine...")
        self.ocr.load()
        self.vlm.load()
        self._initialized = True
        logger.info("Perception engine ready")
        
    def perceive(self, monitor: Optional[int] = None, use_vlm: bool = True) -> PerceptionResult:
        """
        Run complete perception pipeline.
        
        Args:
            monitor: Monitor to capture (None for default)
            use_vlm: Whether to run VLM (can skip for speed)
        """
        if not self._initialized:
            self.initialize()
        
        # Step 3: Screen capture
        frame = self.screen_capture.grab(monitor)
        logger.debug(f"Captured {frame.width}x{frame.height}")
        
        # Step 4: OCR
        ocr_raw = self.ocr.detect(frame.image)
        ocr_results = [{"text": r.text, "bbox": r.bbox, "bbox_rect": r.bbox_rect, 
                        "confidence": r.confidence} for r in ocr_raw]
        
        # Step 5: VLM (optional)
        vlm_regions = []
        screen_description = ""
        if use_vlm:
            vlm_raw = self.vlm.detect_ui_elements(frame.image)
            vlm_regions = [{"role": r.role, "description": r.description,
                           "bbox_normalized": r.bbox_normalized, "confidence": r.confidence} for r in vlm_raw]
            screen_description = self.vlm.describe_screen(frame.image)
        
        # Step 6: Fusion
        fused = self.fusion.fuse(ocr_results, vlm_regions, frame.width, frame.height, frame.image)
        
        return PerceptionResult(
            frame=frame, ocr_results=ocr_results, vlm_regions=vlm_regions,
            fused_elements=fused, screen_description=screen_description
        )
    
    def quick_perceive(self, monitor: Optional[int] = None) -> PerceptionResult:
        """Fast perception using only OCR (no VLM)."""
        return self.perceive(monitor, use_vlm=False)
