"""
ATLAS ML Pipeline - Verification
=================================
Visual verification after actions.
PIPELINE STEP 10
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import time
from loguru import logger

from perception.perception_engine import PerceptionEngine, PerceptionResult
from models.llm_model import PlannedAction
from config import config


@dataclass
class VerificationResult:
    """Result of visual verification."""
    passed: bool
    confidence: float
    reason: str
    before_state: Optional[Dict] = None
    after_state: Optional[Dict] = None


class Verifier:
    """
    Verifies that actions had their intended effect via visual comparison.
    
    Usage:
        verifier = Verifier(perception_engine)
        result = verifier.verify(action, before_perception, after_perception)
    """
    
    def __init__(self, perception: PerceptionEngine):
        self.perception = perception
        self.delay = config.verification.verification_delay
        self.confidence_threshold = config.verification.confidence_threshold
        
    def verify(self, action: PlannedAction, 
               before: PerceptionResult,
               after: Optional[PerceptionResult] = None) -> VerificationResult:
        """
        Verify an action had its intended effect.
        
        Args:
            action: The action that was executed
            before: Perception before action
            after: Perception after action (captured if not provided)
        """
        # Wait for UI to update
        time.sleep(self.delay)
        
        # Capture new state if not provided
        if after is None:
            after = self.perception.quick_perceive()
        
        # Verify based on action type
        if action.action_type == "click":
            return self._verify_click(action, before, after)
        elif action.action_type == "type":
            return self._verify_type(action, before, after)
        elif action.action_type == "key":
            return self._verify_key(action, before, after)
        else:
            # For scroll/wait, just check screen changed
            return self._verify_generic(before, after)
    
    def _verify_click(self, action: PlannedAction, 
                      before: PerceptionResult, 
                      after: PerceptionResult) -> VerificationResult:
        """Verify click action."""
        # Check for visual changes that indicate success
        changes = []
        
        # Look for new elements (dropdown, dialog, etc.)
        before_texts = {e.text for e in before.fused_elements if e.text}
        after_texts = {e.text for e in after.fused_elements if e.text}
        new_texts = after_texts - before_texts
        if new_texts:
            changes.append(f"New text appeared: {list(new_texts)[:3]}")
        
        # Check if clicked element is now selected/focused
        # (This would need more sophisticated visual diff)
        
        # Simple heuristic: if OCR results changed, something happened
        before_count = len(before.ocr_results)
        after_count = len(after.ocr_results)
        if abs(after_count - before_count) > 2:
            changes.append(f"Text count changed: {before_count} -> {after_count}")
        
        passed = len(changes) > 0
        return VerificationResult(
            passed=passed,
            confidence=0.7 if passed else 0.3,
            reason="; ".join(changes) if changes else "No visible change detected"
        )
    
    def _verify_type(self, action: PlannedAction,
                     before: PerceptionResult,
                     after: PerceptionResult) -> VerificationResult:
        """Verify text was typed."""
        typed_text = action.text or ""
        
        # Look for the typed text in OCR results
        for result in after.ocr_results:
            if typed_text.lower() in result.get("text", "").lower():
                return VerificationResult(
                    passed=True, confidence=0.9,
                    reason=f"Typed text '{typed_text}' found on screen"
                )
        
        # Partial match
        for result in after.ocr_results:
            ocr_text = result.get("text", "").lower()
            if any(word in ocr_text for word in typed_text.lower().split()):
                return VerificationResult(
                    passed=True, confidence=0.6,
                    reason=f"Partial text match found"
                )
        
        return VerificationResult(
            passed=False, confidence=0.3,
            reason=f"Typed text not found on screen"
        )
    
    def _verify_key(self, action: PlannedAction,
                    before: PerceptionResult,
                    after: PerceptionResult) -> VerificationResult:
        """Verify key press."""
        key = action.key or ""
        
        # For Enter, check if screen changed significantly
        if key in ["enter", "return"]:
            before_texts = {e.text for e in before.fused_elements if e.text}
            after_texts = {e.text for e in after.fused_elements if e.text}
            if before_texts != after_texts:
                return VerificationResult(passed=True, confidence=0.7, 
                                          reason="Screen changed after Enter")
        
        # For Tab, check if focus moved
        if key == "tab":
            return VerificationResult(passed=True, confidence=0.5,
                                      reason="Tab pressed (focus change not verified)")
        
        return self._verify_generic(before, after)
    
    def _verify_generic(self, before: PerceptionResult,
                        after: PerceptionResult) -> VerificationResult:
        """Generic verification: did anything change?"""
        before_count = len(before.ocr_results)
        after_count = len(after.ocr_results)
        
        if before_count != after_count:
            return VerificationResult(passed=True, confidence=0.5,
                                      reason="Screen content changed")
        
        return VerificationResult(passed=True, confidence=0.4,
                                  reason="Action executed (no change verification)")
