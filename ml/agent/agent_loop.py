"""
ATLAS ML Pipeline - Agent Loop
===============================
Main agent loop: PERCEIVE → UNDERSTAND → PLAN → ACT → VERIFY → repeat
"""

from typing import Optional, List, Dict, Any
import time
from loguru import logger

from models import LLMModel
from perception import PerceptionEngine
from actions import ActionExecutor, ClickAction, TypeAction, KeyAction, ScrollAction, WaitAction
from agent.state import AgentState, AgentStatus
from agent.verification import Verifier
from memory import Memory
from memory.memory import PatternRecord
from config import config


class AgentLoop:
    """
    Main agent control loop implementing the core pipeline.
    
    Usage:
        agent = AgentLoop()
        agent.run("Open Notepad and type hello")
    """
    
    def __init__(self):
        self.llm = LLMModel()
        self.perception = PerceptionEngine(enable_vlm=config.vlm.use_vlm)
        self.executor = ActionExecutor()
        self.verifier = Verifier(self.perception)
        self.memory = Memory()
        self.state = AgentState()
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize all models."""
        if self._initialized:
            return
        logger.info("Initializing agent...")
        self.llm.load()
        # Initialize perception (starts with OCR)
        self.perception.initialize_ocr()
        if config.vlm.use_vlm:
            self.perception.initialize_vlm()
             
        self._initialized = True
        logger.info("Agent ready")
        
    def run(self, user_prompt: str) -> bool:
        """
        Execute a user task.
        
        Args:
            user_prompt: Natural language instruction
            
        Returns:
            True if task completed successfully
        """
        if not self._initialized:
            self.initialize()
            
        self.state.reset()
        self.state.status = AgentStatus.RUNNING
        
        try:
            # STEP 1: Extract intent
            logger.info(f"Processing: {user_prompt}")
            self.state.intent = self.llm.extract_intent(user_prompt)
            logger.info(f"Intent: {self.state.intent.goal}")
            
            # STEP 2: Create task plan
            self.state.task_steps = self.llm.create_task_plan(self.state.intent)
            if not self.state.task_steps:
                logger.error("Failed to create task plan")
                return False
                
            logger.info(f"Plan: {len(self.state.task_steps)} steps")
            for s in self.state.task_steps:
                logger.info(f"  {s.step_number}. {s.description}")
            
            # Main loop
            while not self.state.is_complete and self.state.status == AgentStatus.RUNNING:
                success = self._execute_step()
                
                if not success:
                    recovery = self._handle_failure()
                    if recovery == "retry":
                        continue
                    elif recovery == "skip":
                        self.state.retry_count = 0
                        self.state.advance_step()
                        continue
                    else:  # abort
                        self.state.status = AgentStatus.FAILED
                        logger.error("Task aborted after failure")
                        return False
                
                # Store successful pattern in memory
                self._store_success_pattern()
                
                # Reset retry count on success
                self.state.retry_count = 0
                self.state.advance_step()
            
            self.state.status = AgentStatus.COMPLETED
            logger.info("Task completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self.state.status = AgentStatus.FAILED
            self.state.last_error = str(e)
            return False
    
    def _execute_step(self) -> bool:
        """Execute one step of the task."""
        step = self.state.current_step
        if not step:
            return True
            
        logger.info(f"Executing Step {step.step_number}: {step.description}")
        
        # STEP 3-6: Perceive (OCR-only for speed and reliability)
        # VLM detect_ui_elements returns unreliable bboxes and takes ~60s per call.
        # OCR-only path is fast (~5s) and gives accurate text+bbox data.
        self.state.perception = self.perception.quick_perceive()
        
        # Log detected elements count
        num_elements = len(self.state.perception.fused_elements)
        logger.debug(f"Perceived {num_elements} UI elements")
        
        elements = [e.to_dict() for e in self.state.perception.fused_elements]
        
        # STEP 7: Plan action
        planned = self.llm.plan_action(
            screen_description=self.state.perception.screen_description or f"Screen with {num_elements} elements",
            available_elements=elements,
            current_step=step,
            history=self.state.get_recent_actions()
        )
        self.state.current_action = planned
        logger.info(f"Planned action: {planned.action_type} on {planned.target_text or planned.target_role or 'screen'}")
        
        # Boost confidence from memory
        if self.state.intent and self.state.intent.app:
            boost = self.memory.get_confidence_boost(
                self.state.intent.app, 
                planned.target_role or "",
                planned.target_text
            )
            if boost > 0:
                planned.confidence = min(1.0, planned.confidence + boost)
                logger.debug(f"Memory confidence boost: +{boost:.2f} -> {planned.confidence:.2f}")
        
        if planned.confidence < 0.3:
            logger.warning(f"Low confidence action: {planned.confidence}")
        
        # STEP 8: Resolve coordinates
        action = self._resolve_action(planned)
        if action is None:
            logger.error(f"Failed to resolve action coordinates for {planned.target_text or planned.target_role}")
            return False
        
        # Store pre-action perception
        before = self.state.perception
        
        # STEP 9: Execute
        logger.info(f"Executing: {action.describe()}")
        success = self.executor.execute(action)
        
        if not success:
            logger.error("Execution failed at OS level")
            self.state.record_action(planned, False, False, "Execution failed")
            return False
        
        # STEP 10: Verify
        verification = self.verifier.verify(planned, before)
        
        logger.info(f"Verification: {'PASSED' if verification.passed else 'FAILED'} ({verification.confidence:.2f}) - {verification.reason}")
        
        self.state.record_action(planned, True, verification.passed, 
                                 None if verification.passed else verification.reason)
        
        if not verification.passed:
            return False
            
        return True
    
    def _resolve_action(self, planned):
        """Convert planned action to executable action with coordinates."""
        frame = self.state.perception.frame
        
        # Find target element
        target = None
        
        # 1. Try exact text match
        if planned.target_text:
            target = self.state.perception.get_element_by_text(planned.target_text, fuzzy=False)
            if not target:
                # 2. Try fuzzy text match
                target = self.state.perception.get_element_by_text(planned.target_text, fuzzy=True)
        
        # 3. Try role match if no text target or text match failed
        if not target and planned.target_role:
            target = self.state.perception.get_element_by_role(planned.target_role)
        
        # 4. If multiple candidates, use LLM to rank them
        if not target and planned.target_text:
            candidates = self.state.perception.get_all_by_text(planned.target_text, fuzzy=True)
            if len(candidates) > 1:
                ranked = self.llm.rank_candidates(
                    [c.to_dict() for c in candidates],
                    planned.target_text
                )
                if ranked and ranked[0].get("relevance_score", 0) > 0.5:
                    # Use the top-ranked candidate
                    target = candidates[0]  # Already sorted by relevance
                    logger.debug(f"LLM ranked candidate: {target.text}")
        
        # Debug log result
        if target:
            logger.debug(f"Resolved target: {target.text or target.role} at {target.center}")
        elif planned.target_text or planned.target_role:
            logger.warning(f"Target not found: text='{planned.target_text}', role='{planned.target_role}'")
        
        if planned.action_type == "click":
            if target:
                cx, cy = target.center
                x = int(cx * frame.width)
                y = int(cy * frame.height)
                return ClickAction(x=x, y=y, confidence=planned.confidence)
            logger.error("Click validation failed: No target element found")
            return None
            
        elif planned.action_type == "type":
            return TypeAction(text=planned.text or "", confidence=planned.confidence)
            
        elif planned.action_type == "key":
            return KeyAction(key=planned.key or "", confidence=planned.confidence)
            
        elif planned.action_type == "scroll":
            x, y = frame.width // 2, frame.height // 2
            if target:
                cx, cy = target.center
                x, y = int(cx * frame.width), int(cy * frame.height)
            return ScrollAction(x=x, y=y, direction=planned.direction or "down", 
                                confidence=planned.confidence)
        
        elif planned.action_type == "wait":
            return WaitAction(duration=1.0)
        
        return None
    
    def _handle_failure(self) -> str:
        """
        Handle a failed step using LLM-guided error recovery.
        
        Returns:
            "retry" - retry the same step
            "skip" - skip to next step
            "abort" - stop the task
        """
        if not self.state.can_retry:
            logger.error("Max retries exceeded")
            return "abort"
        
        self.state.retry_count += 1
        self.state.status = AgentStatus.RETRYING
        
        step = self.state.current_step
        last_action = self.state.current_action
        last_error = self.state.last_error or "Verification failed"
        
        # Get the last action record's error if available
        if self.state.action_history:
            last_record = self.state.action_history[-1]
            if last_record.error:
                last_error = last_record.error
        
        logger.warning(f"Step {step.step_number} failed (attempt {self.state.retry_count}): {last_error}")
        
        # Ask LLM for recovery strategy
        try:
            screen_desc = ""
            if self.state.perception:
                screen_desc = self.state.perception.screen_description or f"Screen with {len(self.state.perception.fused_elements)} elements"
            
            recovery = self.llm.handle_error(
                error_description=last_error,
                screen_state=screen_desc,
                last_action=last_action,
                retry_count=self.state.retry_count
            )
            
            strategy = recovery.get("strategy", "retry")
            reason = recovery.get("reason", "")
            logger.info(f"LLM recovery strategy: {strategy} - {reason}")
            
            if strategy == "abort":
                return "abort"
            elif strategy == "skip":
                return "skip"
            else:
                # retry or replan — both result in re-executing the step
                time.sleep(1.0)
                self.state.status = AgentStatus.RUNNING
                return "retry"
                
        except Exception as e:
            logger.warning(f"LLM error recovery failed, defaulting to retry: {e}")
            time.sleep(1.0)
            self.state.status = AgentStatus.RUNNING
            return "retry"
    
    def _store_success_pattern(self) -> None:
        """Store successful action pattern in memory for future confidence boosting."""
        try:
            action = self.state.current_action
            if not action or not self.state.intent:
                return
            
            app_name = self.state.intent.app or "unknown"
            self.memory.store(PatternRecord(
                app_name=app_name,
                element_role=action.target_role or "unknown",
                element_text=action.target_text,
                bbox_relative=list(self.state.perception.fused_elements[0].bbox_normalized) if self.state.perception and self.state.perception.fused_elements else [0, 0, 0, 0],
                action_type=action.action_type
            ))
        except Exception as e:
            logger.debug(f"Failed to store pattern (non-critical): {e}")
    
    def stop(self) -> None:
        """Stop the agent."""
        self.state.status = AgentStatus.IDLE
        self.memory.close()
        logger.info("Agent stopped")
