"""
ATLAS ML Pipeline - OCR Model (PaddleOCR)
=========================================

Handles text detection and recognition from screenshots.
Returns text content with precise bounding box coordinates.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger

from config import OCRConfig, config


@dataclass
class OCRResult:
    """Single OCR detection result."""
    text: str
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] polygon
    confidence: float
    
    @property
    def bbox_rect(self) -> Tuple[int, int, int, int]:
        """Convert polygon to rectangle (x1, y1, x2, y2)."""
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return (min(xs), min(ys), max(xs), max(ys))
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox_rect
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class OCRModel:
    """
    PaddleOCR wrapper for text detection and recognition.
    
    Usage:
        ocr = OCRModel()
        results = ocr.detect(screenshot_array)
        for result in results:
            print(f"Text: {result.text}, Box: {result.bbox_rect}")
    """
    
    def __init__(self, ocr_config: Optional[OCRConfig] = None):
        self.config = ocr_config or config.ocr
        self._model = None
        
    def load(self) -> None:
        """Initialize the PaddleOCR model."""
        try:
            from paddleocr import PaddleOCR
            
            self._model = PaddleOCR(
                use_gpu=self.config.use_gpu,
                lang=self.config.lang,
                use_angle_cls=self.config.use_angle_cls,
                det_db_thresh=self.config.det_db_thresh,
                det_db_box_thresh=self.config.det_db_box_thresh,
                rec_batch_num=self.config.rec_batch_num,
                show_log=False,
            )
            logger.info("PaddleOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[OCRResult]:
        """
        Run OCR on an image.
        
        Args:
            image: Screenshot as numpy array (H, W, C) in RGB format
            
        Returns:
            List of OCRResult with text and bounding boxes
        """
        if self._model is None:
            self.load()
        
        results = []
        
        try:
            # PaddleOCR expects BGR, convert if needed
            ocr_output = self._model.ocr(image, cls=self.config.use_angle_cls)
            
            if ocr_output is None or len(ocr_output) == 0:
                return results
            
            # Parse results - format: [[[box], (text, conf)], ...]
            for line in ocr_output[0]:
                if line is None:
                    continue
                    
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = line[1][0]
                confidence = line[1][1]
                
                results.append(OCRResult(
                    text=text,
                    bbox=[[int(p[0]), int(p[1])] for p in bbox],
                    confidence=confidence
                ))
                
            logger.debug(f"OCR detected {len(results)} text regions")
            
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            
        return results
    
    def find_text(self, image: np.ndarray, target_text: str, 
                  fuzzy: bool = False) -> Optional[OCRResult]:
        """
        Find specific text in image.
        
        Args:
            image: Screenshot array
            target_text: Text to find
            fuzzy: If True, use partial matching
            
        Returns:
            OCRResult if found, None otherwise
        """
        results = self.detect(image)
        
        for result in results:
            if fuzzy:
                if target_text.lower() in result.text.lower():
                    return result
            else:
                if result.text.lower() == target_text.lower():
                    return result
                    
        return None
