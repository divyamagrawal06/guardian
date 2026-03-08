"""
ATLAS ML Pipeline - Agent Loop
===============================
Main agent loop: PERCEIVE → UNDERSTAND → PLAN → ACT → VERIFY → repeat
"""

import time
from typing import Optional, Callable, List
from loguru import logger

from models import LLMModel
from perception import PerceptionEngine
from perception.ui_automation import UIAutomationPerception, UIElement
from actions import ActionExecutor, ClickAction, TypeAction, KeyAction, ScrollAction, WaitAction, OpenAction
from agent.state import AgentState, AgentStatus
from agent.verification import Verifier
from memory.memory import Memory, PatternRecord
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
        self.memory = Memory()
        self.ui_auto = UIAutomationPerception(max_depth=8, max_elements=120)
        self._initialized = False
        self._on_progress: Optional[Callable] = None
        self._ui_elements: List[UIElement] = []  # latest UIA elements
        
    def initialize(self) -> None:
        """Initialize all models (graceful-degrade if vision deps are missing)."""
        if self._initialized:
            return
        logger.info("Initializing agent...")
        self._emit("init", "started", "Loading models...")
        self.llm.load()
        try:
            self.perception.initialize()
        except Exception as e:
            logger.warning(f"Perception init failed (will use screenshot-only mode): {e}")
        self._initialized = True
        self._emit("init", "completed", "Agent ready")
        logger.info("Agent ready")
    
    def _emit(self, step: str, status: str, detail: str = "") -> None:
        """Send progress update to callback if set."""
        if self._on_progress:
            try:
                self._on_progress(step, status, detail)
            except Exception:
                pass
        
    def run(self, user_prompt: str, on_progress: Optional[Callable] = None) -> bool:
        """
        Execute a user task.
        
        Args:
            user_prompt: Natural language instruction
            on_progress: Optional callback(step, status, detail) for progress updates
            
        Returns:
            True if task completed successfully
        """
        self._on_progress = on_progress
        
        if not self._initialized:
            self.initialize()
            
        self.state.reset()
        self.state.status = AgentStatus.RUNNING
        
        try:
            # STEP 1: Extract intent
            self._emit("intent", "started", f"Understanding: {user_prompt}")
            logger.info(f"Processing: {user_prompt}")
            self.state.intent = self.llm.extract_intent(user_prompt)
            self._emit("intent", "completed", f"Goal: {self.state.intent.goal}")
            logger.info(f"Intent: {self.state.intent.goal}")
            
            # STEP 2: Create task plan
            self._emit("planning", "started", "Creating task plan...")
            self.state.task_steps = self.llm.create_task_plan(self.state.intent)
            step_descriptions = [s.description for s in self.state.task_steps]
            self._emit("planning", "completed", f"{len(self.state.task_steps)} steps: {step_descriptions}")
            logger.info(f"Plan: {len(self.state.task_steps)} steps")
            
            # Main loop
            while not self.state.is_complete and self.state.status == AgentStatus.RUNNING:
                step = self.state.current_step
                self._emit("step", "started", 
                          f"Step {step.step_number}: {step.description}" if step else "")
                
                success = self._execute_step()
                
                if not success:
                    if self.state.can_retry:
                        self.state.retry_count += 1
                        self._emit("step", "retrying", 
                                  f"Retry {self.state.retry_count}/{self.state.max_retries}")
                        logger.warning(f"Retrying (attempt {self.state.retry_count})")
                        self.state.status = AgentStatus.RUNNING
                        continue
                    else:
                        self.state.status = AgentStatus.FAILED
                        self._emit("task", "failed", "Max retries exceeded")
                        logger.error("Max retries exceeded")
                        return False
                
                self._emit("step", "completed", 
                          f"Step {step.step_number} done" if step else "")
                self.state.advance_step()
            
            self.state.status = AgentStatus.COMPLETED
            self._emit("task", "completed", "Task completed successfully")
            logger.info("Task completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self.state.status = AgentStatus.FAILED
            self.state.last_error = str(e)
            self._emit("task", "failed", str(e))
            return False
        finally:
            self._on_progress = None
    
    def _execute_step(self) -> bool:
        """Execute one step of the task."""
        step = self.state.current_step
        if not step:
            return True
            
        logger.info(f"Step {step.step_number}: {step.description}")
        
        # ── PERCEIVE via Windows UI Automation (primary) ──
        elements = []
        screen_description = "Screen analysis unavailable"
        try:
            self._emit("perceive", "started", "Reading UI Automation tree...")
            self._ui_elements = self.ui_auto.get_elements()
            elements = [e.to_dict() for e in self._ui_elements]
            screen_description = self.ui_auto.describe_screen(self._ui_elements)
            self._emit("perceive", "completed", f"Found {len(elements)} UI elements")
        except Exception as e:
            logger.warning(f"UI Automation perception failed: {e}")
            # Fallback: try PaddleOCR/VLM perception
            try:
                self.state.perception = self.perception.perceive()
                elements = [el.to_dict() for el in self.state.perception.fused_elements]
                screen_description = self.state.perception.screen_description
                self._emit("perceive", "completed", f"Fallback perception: {len(elements)} elements")
            except Exception as e2:
                logger.warning(f"All perception failed, direct-action mode: {e2}")
                self._emit("perceive", "completed", "Using direct-action mode (no vision)")
        
        # Check memory for known patterns
        app_name = self.state.intent.app or "unknown"
        
        # Plan action
        self._emit("action", "planning", "Deciding next action...")
        planned = self.llm.plan_action(
            screen_description=screen_description,
            available_elements=elements,
            current_step=step,
            history=self.state.get_recent_actions(),
            user_prompt=self.state.intent.raw_prompt or self.state.intent.goal or ""
        )
        self.state.current_action = planned
        
        if planned.confidence < 0.3:
            logger.warning(f"Low confidence action: {planned.confidence}")
        
        # Resolve coordinates (or direct action for type/key)
        action = self._resolve_action(planned)
        if action is None:
            logger.error("Failed to resolve action coordinates")
            self._emit("action", "failed", "Could not find target element")
            return False
        
        # Execute
        self._emit("action", "executing", f"{action.describe()}")
        success = self.executor.execute(action)
        
        if not success:
            self.state.record_action(planned, False, False, "Execution failed")
            self._emit("action", "failed", "Execution failed")
            return False
        
        # Brief delay to let the UI settle after action
        time.sleep(1.5)
        
        # Lightweight verification (just check window title, skip full tree re-read)
        try:
            new_window = self.ui_auto.get_window_title()
            self.state.record_action(planned, True, True, None)
            self._emit("verify", "completed", f"Window: {new_window}")
        except Exception:
            self.state.record_action(planned, True, True, None)
            self._emit("verify", "completed", "Action executed")
        
        # Store successful patterns in memory
        if planned.target_text:
            try:
                self.memory.store(PatternRecord(
                    app_name=app_name,
                    element_role=planned.target_role or "unknown",
                    element_text=planned.target_text,
                    bbox_relative=[0, 0, 0, 0],
                    action_type=planned.action_type,
                ))
            except Exception:
                pass

        return True
    
    def _resolve_action(self, planned):
        """Convert planned action to executable action with coordinates."""
        # For type/key/wait actions, we don't need coordinates
        if planned.action_type == "type":
            return TypeAction(text=planned.text or "", confidence=planned.confidence)
        elif planned.action_type == "key":
            return KeyAction(key=planned.key or "", confidence=planned.confidence)
        elif planned.action_type == "wait":
            duration = 3.0  # default wait
            if planned.text and planned.text.replace('.','',1).isdigit():
                duration = float(planned.text)
            return WaitAction(duration=duration, confidence=planned.confidence)
        elif planned.action_type == "open":
            # LLM might put app/URL in text OR target_text
            target = planned.text or planned.target_text or ""
            return OpenAction(target=target, confidence=planned.confidence)
        
        # ── For click/scroll, resolve target from UI Automation elements ──
        target: Optional[UIElement] = None

        # 1. Try UIAutomation elements (already absolute screen-pixel coords)
        if self._ui_elements:
            # Exact role + text match
            for el in self._ui_elements:
                if planned.target_role and el.role.lower() == planned.target_role.lower():
                    if planned.target_text and el.name and planned.target_text.lower() in el.name.lower():
                        target = el
                        break
            # Role-only match
            if target is None and planned.target_role:
                for el in self._ui_elements:
                    if el.role.lower() == planned.target_role.lower():
                        target = el
                        break
            # Text match (any role)
            if target is None and planned.target_text:
                for el in self._ui_elements:
                    if el.name and planned.target_text.lower() in el.name.lower():
                        target = el
                        break
            # Fuzzy word match
            if target is None and planned.target_text:
                words = planned.target_text.lower().split()
                for el in self._ui_elements:
                    if el.name and any(w in el.name.lower() for w in words):
                        target = el
                        break

        # 2. Fallback: old perception engine elements
        if target is None and self.state.perception:
            frame = getattr(self.state.perception, 'frame', None)
            dpi_scale = frame.dpi_scale if frame else 1.0
            fused = self.state.perception.fused_elements or []
            for element in fused:
                if planned.target_text and element.text:
                    if planned.target_text.lower() in element.text.lower():
                        cx, cy = element.center
                        x = int((cx * frame.width) / dpi_scale) if frame else int(cx)
                        y = int((cy * frame.height) / dpi_scale) if frame else int(cy)
                        if planned.action_type == "click":
                            return ClickAction(x=x, y=y, confidence=planned.confidence)
                        elif planned.action_type == "scroll":
                            return ScrollAction(x=x, y=y, direction=planned.direction or "down",
                                               confidence=planned.confidence)

        if planned.action_type == "click":
            if target:
                x, y = target.center  # absolute screen pixels from UIA
                logger.info(f"Click target: '{target.name}' ({target.role}) @ ({x}, {y})")
                return ClickAction(x=x, y=y, confidence=planned.confidence)
            logger.warning("No target found for click — returning None")
            return None
            
        elif planned.action_type == "scroll":
            x, y = 960, 540  # default center
            if target:
                x, y = target.center
            return ScrollAction(x=x, y=y, direction=planned.direction or "down", 
                               confidence=planned.confidence)
        
        return None
    
    def stop(self) -> None:
        """Stop the agent."""
        self.state.status = AgentStatus.IDLE
        logger.info("Agent stopped")
