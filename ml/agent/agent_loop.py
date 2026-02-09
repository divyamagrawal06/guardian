"""
ATLAS ML Pipeline - Agent Loop
===============================
Main agent loop: PERCEIVE → UNDERSTAND → PLAN → ACT → VERIFY → repeat
"""

from typing import Optional
from loguru import logger

from models import LLMModel
from perception import PerceptionEngine
from actions import ActionExecutor, ClickAction, TypeAction, KeyAction, ScrollAction
from agent.state import AgentState, AgentStatus
from agent.verification import Verifier
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
        self.perception = PerceptionEngine()
        self.executor = ActionExecutor()
        self.verifier = Verifier(self.perception)
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
                    if self.state.can_retry:
                        self.state.retry_count += 1
                        self.state.status = AgentStatus.RETRYING
                        logger.warning(f"Retrying step {self.state.current_step.step_number} (attempt {self.state.retry_count})")
                        # Add a small delay and potentially re-perceive/re-plan in next iteration
                        import time
                        time.sleep(1.0)
                        # Reset status to RUNNING to continue loop
                        self.state.status = AgentStatus.RUNNING
                        continue
                    else:
                        self.state.status = AgentStatus.FAILED
                        logger.error("Max retries exceeded")
                        return False
                
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
        
        # STEP 3-6: Perceive
        # Use quick_perceive by default for speed, unless VLM is strictly required
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
        
        if planned.confidence < 0.3:
            logger.warning(f"Low confidence action: {planned.confidence}")
        
        # STEP 8: Resolve coordinates
        action = self._resolve_action(planned)
        if action is None:
            # If action resolution failed, it might be because the element wasn't found.
            # We could try to fallback or just fail.
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
        # Verify will capture the 'after' state automatically if we don't pass it
        verification = self.verifier.verify(planned, before)
        
        logger.info(f"Verification: {'PASSED' if verification.passed else 'FAILED'} ({verification.confidence:.2f}) - {verification.reason}")
        
        self.state.record_action(planned, True, verification.passed, 
                                 None if verification.passed else verification.reason)
        
        if not verification.passed:
            # If verification failed, we return False to trigger retry logic in run()
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
             from actions import WaitAction
             return WaitAction(duration=1.0)
        
        return None
    
    def stop(self) -> None:
        """Stop the agent."""
        self.state.status = AgentStatus.IDLE
        logger.info("Agent stopped")
