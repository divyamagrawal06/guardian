"""
ATLAS ML Pipeline - Screen Capture
===================================

Handles screenshot capture and coordinate frame management.

PIPELINE STEP 3
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger

from config import ScreenConfig, config


@dataclass
class ScreenFrame:
    """Captured screen with metadata."""
    image: np.ndarray  # RGB image array
    width: int
    height: int
    monitor: int
    timestamp: float
    dpi_scale: float = 1.0
    
    def to_absolute(self, x_norm: float, y_norm: float) -> Tuple[int, int]:
        """Convert normalized coordinates to absolute pixels."""
        x = int(x_norm * self.width * self.dpi_scale)
        y = int(y_norm * self.height * self.dpi_scale)
        return (x, y)
    
    def to_normalized(self, x: int, y: int) -> Tuple[float, float]:
        """Convert absolute pixels to normalized coordinates."""
        x_norm = x / (self.width * self.dpi_scale)
        y_norm = y / (self.height * self.dpi_scale)
        return (x_norm, y_norm)


class ScreenCapture:
    """
    Screen capture utility using mss.
    
    Captures screenshots and manages coordinate frames.
    
    Usage:
        capture = ScreenCapture()
        frame = capture.grab()
        x, y = frame.to_absolute(0.5, 0.5)  # Center of screen
    """
    
    def __init__(self, screen_config: Optional[ScreenConfig] = None):
        self.config = screen_config or config.screen
        self._mss = None
        self._monitor_info = None
        
    def _init_mss(self) -> None:
        """Initialize mss screen capture."""
        import mss
        self._mss = mss.mss()
        self._monitor_info = self._mss.monitors
        logger.debug(f"Screen capture initialized. Monitors: {len(self._monitor_info) - 1}")
        
    def grab(self, monitor: Optional[int] = None) -> ScreenFrame:
        """
        Capture screenshot of specified monitor.
        
        Args:
            monitor: Monitor number (1-based) or None for configured default
            
        Returns:
            ScreenFrame with image and metadata
        """
        if self._mss is None:
            self._init_mss()
            
        import time
        
        mon = monitor or self.config.capture_monitor
        
        try:
            # Get monitor geometry
            mon_info = self._mss.monitors[mon]
            
            # Capture screenshot
            screenshot = self._mss.grab(mon_info)
            
            # Convert to numpy RGB array
            image = np.array(screenshot)
            # mss returns BGRA, convert to RGB
            image = image[:, :, [2, 1, 0]]
            
            frame = ScreenFrame(
                image=image,
                width=mon_info["width"],
                height=mon_info["height"],
                monitor=mon,
                timestamp=time.time(),
                dpi_scale=self.config.dpi_scale
            )
            
            logger.debug(f"Captured screen: {frame.width}x{frame.height}")
            return frame
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            raise
    
    def get_monitor_info(self) -> list:
        """Get information about all monitors."""
        if self._mss is None:
            self._init_mss()
        return self._monitor_info[1:]  # Skip the "all monitors" entry
    
    def detect_dpi_scale(self) -> float:
        """
        Detect current DPI scaling.
        
        TODO: Implement proper DPI detection (CRITICAL-001)
        """
        try:
            import ctypes
            
            # Get DPI awareness
            awareness = ctypes.c_int()
            ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
            
            # Get DPI for primary monitor
            dc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, dc)
            
            scale = dpi / 96.0  # 96 DPI is 100% scaling
            logger.info(f"Detected DPI scale: {scale}")
            return scale
            
        except Exception as e:
            logger.warning(f"DPI detection failed, using 1.0: {e}")
            return 1.0
    
    def save_screenshot(self, frame: ScreenFrame, path: str) -> None:
        """Save screenshot to file for debugging."""
        from PIL import Image
        
        img = Image.fromarray(frame.image)
        img.save(path)
        logger.debug(f"Screenshot saved: {path}")
