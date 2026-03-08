"""
ATLAS ML Pipeline - OCR Model (EasyOCR)
=======================================

Handles text detection and recognition from screenshots.
Returns text content with precise bounding box coordinates.

Switched from PaddleOCR to EasyOCR for Windows compatibility.
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
    
    @property
    def width(self) -> int:
        """Width of bounding box."""
        x1, y1, x2, y2 = self.bbox_rect
        return x2 - x1
    
    @property
    def height(self) -> int:
        """Height of bounding box."""
        x1, y1, x2, y2 = self.bbox_rect
        return y2 - y1
    
    def to_normalized(self, screen_width: int, screen_height: int) -> Tuple[float, float, float, float]:
        """Convert bbox to normalized coordinates (0-1 range)."""
        x1, y1, x2, y2 = self.bbox_rect
        return (
            x1 / screen_width,
            y1 / screen_height,
            x2 / screen_width,
            y2 / screen_height
        )


class OCRModel:
    """
    EasyOCR wrapper for text detection and recognition.
    
    Usage:
        ocr = OCRModel()
        results = ocr.detect(screenshot_array)
        for result in results:
            print(f"Text: {result.text}, Box: {result.bbox_rect}")
    """
    
    def __init__(self, ocr_config: Optional[OCRConfig] = None):
        self.config = ocr_config or config.ocr
        self._reader = None
        
    def load(self) -> None:
        """Initialize the EasyOCR reader."""
        try:
            import easyocr
            
            # EasyOCR uses GPU if available, fallback to CPU
            use_gpu = self.config.use_gpu
            
            self._reader = easyocr.Reader(
                [self.config.lang],
                gpu=use_gpu,
                verbose=False,
            )
            logger.info(f"EasyOCR loaded (GPU: {use_gpu}, Lang: {self.config.lang})")
            
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            raise
    
    def detect(self, image: np.ndarray, min_confidence: float = 0.3) -> List[OCRResult]:
        """
        Run OCR on an image.
        
        Args:
            image: Screenshot as numpy array (H, W, C) in RGB format
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            List of OCRResult with text and bounding boxes
        """
        if self._reader is None:
            self.load()
        
        results = []
        
        try:
            # EasyOCR returns: [[bbox, text, confidence], ...]
            # bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            ocr_output = self._reader.readtext(image)
            
            for detection in ocr_output:
                bbox = detection[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                text = detection[1]
                confidence = detection[2]
                
                # Filter by confidence
                if confidence < min_confidence:
                    continue
                
                # Convert bbox points to integers
                bbox_int = [[int(p[0]), int(p[1])] for p in bbox]
                
                results.append(OCRResult(
                    text=text,
                    bbox=bbox_int,
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
        
        target_lower = target_text.lower()
        
        for result in results:
            text_lower = result.text.lower()
            
            if fuzzy:
                if target_lower in text_lower:
                    return result
            else:
                if text_lower == target_lower:
                    return result
                    
        return None
    
    def find_all_matching(self, image: np.ndarray, target_text: str,
                          fuzzy: bool = True) -> List[OCRResult]:
        """
        Find all text regions matching target.
        
        Args:
            image: Screenshot array
            target_text: Text pattern to find
            fuzzy: If True, use partial matching
            
        Returns:
            List of matching OCRResult
        """
        results = self.detect(image)
        matches = []
        
        target_lower = target_text.lower()
        
        for result in results:
            text_lower = result.text.lower()
            
            if fuzzy:
                if target_lower in text_lower:
                    matches.append(result)
            else:
                if text_lower == target_lower:
                    matches.append(result)
                    
        return matches
    
    def get_text_near_point(self, image: np.ndarray, x: int, y: int, 
                            radius: int = 100) -> List[OCRResult]:
        """
        Find text regions near a specific point.
        
        Args:
            image: Screenshot array
            x, y: Point coordinates
            radius: Search radius in pixels
            
        Returns:
            List of OCRResult within radius, sorted by distance
        """
        results = self.detect(image)
        nearby = []
        
        for result in results:
            cx, cy = result.center
            distance = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            
            if distance <= radius:
                nearby.append((distance, result))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[0])
        return [r for _, r in nearby]
    
    def extract_all_text(self, image: np.ndarray) -> str:
        """
        Extract all text from image as a single string.
        
        Args:
            image: Screenshot array
            
        Returns:
            All detected text joined with spaces
        """
        results = self.detect(image)
        
        # Sort by position (top to bottom, left to right)
        sorted_results = sorted(results, key=lambda r: (r.bbox_rect[1], r.bbox_rect[0]))
        
        return " ".join(r.text for r in sorted_results)
