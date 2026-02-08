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
        self.perception.initialize()
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
            logger.info(f"Plan: {len(self.state.task_steps)} steps")
            
            # Main loop
            while not self.state.is_complete and self.state.status == AgentStatus.RUNNING:
                success = self._execute_step()
                
                if not success:
                    if self.state.can_retry:
                        self.state.retry_count += 1
                        self.state.status = AgentStatus.RETRYING
                        logger.warning(f"Retrying (attempt {self.state.retry_count})")
                        continue
                    else:
                        self.state.status = AgentStatus.FAILED
                        logger.error("Max retries exceeded")
                        return False
                
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
            
        logger.info(f"Step {step.step_number}: {step.description}")
        
        # STEP 3-6: Perceive
        self.state.perception = self.perception.perceive()
        elements = [e.to_dict() for e in self.state.perception.fused_elements]
        
        # STEP 7: Plan action
        planned = self.llm.plan_action(
            screen_description=self.state.perception.screen_description,
            available_elements=elements,
            current_step=step,
            history=self.state.get_recent_actions()
        )
        self.state.current_action = planned
        
        if planned.confidence < 0.3:
            logger.warning(f"Low confidence action: {planned.confidence}")
        
        # STEP 8: Resolve coordinates
        action = self._resolve_action(planned)
        if action is None:
            logger.error("Failed to resolve action coordinates")
            return False
        
        # Store pre-action perception
        before = self.state.perception
        
        # STEP 9: Execute
        success = self.executor.execute(action)
        
        if not success:
            self.state.record_action(planned, False, False, "Execution failed")
            return False
        
        # STEP 10: Verify
        verification = self.verifier.verify(planned, before)
        
        self.state.record_action(planned, True, verification.passed, 
                                 None if verification.passed else verification.reason)
        
        if not verification.passed:
            logger.warning(f"Verification failed: {verification.reason}")
            return False
            
        return True
    
    def _resolve_action(self, planned):
        """Convert planned action to executable action with coordinates."""
        frame = self.state.perception.frame
        
        # Find target element
        target = None
        for element in self.state.perception.fused_elements:
            if planned.target_role and element.role == planned.target_role:
                target = element
                break
            if planned.target_text and element.text:
                if planned.target_text.lower() in element.text.lower():
                    target = element
                    break
        
        if planned.action_type == "click":
            if target:
                cx, cy = target.center
                x = int(cx * frame.width)
                y = int(cy * frame.height)
                return ClickAction(x=x, y=y, confidence=planned.confidence)
            logger.warning("No target found for click")
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
        
        return None
    
    def stop(self) -> None:
        """Stop the agent."""
        self.state.status = AgentStatus.IDLE
        logger.info("Agent stopped")
