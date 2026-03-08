"""
ATLAS ML Pipeline - Vision-Language Model (LLaVA)
=================================================

Handles visual understanding of screenshots:
- UI element detection (buttons, inputs, icons)
- Layout analysis
- Semantic descriptions of screen regions
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image
from loguru import logger

from config import VLMConfig, config


@dataclass
class UIRegion:
    """Detected UI region with semantic info."""
    role: str  # e.g., "button", "input_field", "icon", "panel"
    description: str
    bbox_normalized: List[float]  # [x1, y1, x2, y2] normalized 0-1
    confidence: float
    
    def to_absolute(self, width: int, height: int) -> tuple:
        """Convert normalized bbox to absolute pixels."""
        x1 = int(self.bbox_normalized[0] * width)
        y1 = int(self.bbox_normalized[1] * height)
        x2 = int(self.bbox_normalized[2] * width)
        y2 = int(self.bbox_normalized[3] * height)
        return (x1, y1, x2, y2)
    
    @property
    def center_normalized(self) -> tuple:
        """Get normalized center point."""
        x = (self.bbox_normalized[0] + self.bbox_normalized[2]) / 2
        y = (self.bbox_normalized[1] + self.bbox_normalized[3]) / 2
        return (x, y)


class VLMModel:
    """
    LLaVA Vision-Language Model wrapper.
    
    Used for:
    - Identifying interactive UI elements
    - Understanding screen layout and structure
    - Providing semantic descriptions of visual content
    
    Usage:
        vlm = VLMModel()
        regions = vlm.detect_ui_elements(screenshot)
        description = vlm.describe_screen(screenshot)
    """
    
    def __init__(self, vlm_config: Optional[VLMConfig] = None):
        self.config = vlm_config or config.vlm
        self._model = None
        self._processor = None
        
    def load(self) -> None:
        """Initialize the VLM model."""
        if self.config.backend == "ollama":
            # Ollama manages models internally
            logger.info(f"VLM using Ollama backend: {self.config.ollama_model}")
            self._model = "ollama"  # unique marker
            return

        # Fallback to transformers (or others) could go here...
        logger.warning(f"Backend {self.config.backend} not fully implemented in load()")
        self._model = None
            
    def _query(self, image: Image.Image, prompt: str) -> str:
        """Run a query against the VLM."""
        if self._model is None:
            self.load()
            
        if self.config.backend == "ollama":
            import requests
            import base64
            from io import BytesIO
            
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            logger.debug(f"Ollama VLM Prompt: {prompt}")
            
            try:
                response = requests.post(
                    f"{config.ollama.base_url}/api/generate",
                    json={
                        "model": self.config.ollama_model,
                        "prompt": prompt,
                        "images": [img_str],
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_new_tokens
                        }
                    },
                    timeout=config.ollama.timeout
                )
                response.raise_for_status()
                result = response.json()["response"].strip()
                logger.debug(f"Ollama VLM Output: {result}")
                return result
            except Exception as e:
                logger.error(f"Ollama VLM query failed: {e}")
                return ""
        
        return ""
    
    def detect_ui_elements(self, image: np.ndarray) -> List[UIRegion]:
        """
        Detect interactive UI elements in screenshot.
        
        Args:
            image: Screenshot as numpy array
            
        Returns:
            List of UIRegion with detected elements
        """
        pil_image = Image.fromarray(image)
        
        prompt = """Analyze this screenshot and identify all interactive UI elements.
For each element, provide:
1. Role (button, input_field, icon, checkbox, dropdown, link, tab, panel)
2. Brief description
3. Approximate location as normalized coordinates [x1, y1, x2, y2] where values are 0-1

Format as JSON list:
[{"role": "...", "description": "...", "bbox": [x1, y1, x2, y2], "confidence": 0.9}]

Focus on clickable and interactive elements."""

        try:
            response = self._query(pil_image, prompt)
            regions = self._parse_ui_response(response)
            logger.debug(f"VLM detected {len(regions)} UI regions")
            return regions
        except Exception as e:
            logger.error(f"VLM UI detection failed: {e}")
            return []
    
    def describe_screen(self, image: np.ndarray) -> str:
        """Get a general description of what's on screen."""
        pil_image = Image.fromarray(image)
        
        prompt = """Describe what application and screen is shown in this screenshot.
Include:
- Application name (if identifiable)
- Current view/page
- Main visible content
- Overall layout structure

Be concise but specific."""

        try:
            return self._query(pil_image, prompt)
        except Exception as e:
            logger.error(f"VLM describe failed: {e}")
            return ""
    
    def find_element(self, image: np.ndarray, target_description: str) -> Optional[UIRegion]:
        """
        Find a specific UI element by description.
        
        Args:
            image: Screenshot array
            target_description: Natural language description of element
            
        Returns:
            UIRegion if found, None otherwise
        """
        pil_image = Image.fromarray(image)
        
        prompt = f"""Find this UI element in the screenshot: "{target_description}"

If found, provide:
- Role (button, input_field, icon, etc.)
- Exact description
- Location as normalized bbox [x1, y1, x2, y2] (values 0-1)
- Confidence (0-1)

Format as JSON: {{"role": "...", "description": "...", "bbox": [...], "confidence": ...}}

If not found, respond with: {{"found": false}}"""

        try:
            response = self._query(pil_image, prompt)
            return self._parse_single_element(response)
        except Exception as e:
            logger.error(f"VLM find element failed: {e}")
            return None
    
    def _parse_ui_response(self, response: str) -> List[UIRegion]:
        """Parse VLM response into UIRegion objects."""
        import json
        
        regions = []
        
        try:
            # Try to extract JSON from response
            # Handle case where response has extra text
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                for item in data:
                    regions.append(UIRegion(
                        role=item.get("role", "unknown"),
                        description=item.get("description", ""),
                        bbox_normalized=item.get("bbox", [0, 0, 1, 1]),
                        confidence=item.get("confidence", 0.5)
                    ))
        except json.JSONDecodeError:
            logger.warning("Failed to parse VLM JSON response")
            
        return regions
    
    def _parse_single_element(self, response: str) -> Optional[UIRegion]:
        """Parse single element response."""
        import json
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                
                if data.get("found") == False:
                    return None
                    
                return UIRegion(
                    role=data.get("role", "unknown"),
                    description=data.get("description", ""),
                    bbox_normalized=data.get("bbox", [0, 0, 1, 1]),
                    confidence=data.get("confidence", 0.5)
                )
        except json.JSONDecodeError:
            pass
            
        return None
