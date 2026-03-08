"""
ATLAS ML Pipeline - Action Executor
====================================
Executes actions via OS-level input (PyAutoGUI).
PIPELINE STEP 9
"""

import pyautogui
import time
from typing import Optional
from loguru import logger

from actions.actions import Action, ClickAction, TypeAction, KeyAction, ScrollAction, WaitAction
from config import config


# Safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small pause between actions


class ActionExecutor:
    """
    Executes atomic actions using PyAutoGUI.
    
    Usage:
        executor = ActionExecutor()
        executor.execute(ClickAction(x=100, y=200))
    """
    
    def __init__(self):
        self.move_duration = config.screen.mouse_move_duration
        self.type_interval = config.screen.typing_interval
        self.last_action = None
        self.last_action_time = 0
        
    def execute(self, action: Action) -> bool:
        """
        Execute an action.
        
        Returns:
            True if action executed without error (NOT verification of success)
        """
        logger.info(f"Executing: {action.describe()}")
        
        try:
            if isinstance(action, ClickAction):
                return self._execute_click(action)
            elif isinstance(action, TypeAction):
                return self._execute_type(action)
            elif isinstance(action, KeyAction):
                return self._execute_key(action)
            elif isinstance(action, ScrollAction):
                return self._execute_scroll(action)
            elif isinstance(action, WaitAction):
                return self._execute_wait(action)
            else:
                logger.error(f"Unknown action type: {type(action)}")
                return False
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
        finally:
            self.last_action = action
            self.last_action_time = time.time()
    
    def _execute_click(self, action: ClickAction) -> bool:
        """Execute mouse click."""
        pyautogui.moveTo(action.x, action.y, duration=self.move_duration)
        pyautogui.click(x=action.x, y=action.y, clicks=action.clicks, button=action.button)
        logger.debug(f"Clicked at ({action.x}, {action.y})")
        return True
    
    def _execute_type(self, action: TypeAction) -> bool:
        """Execute keyboard typing."""
        pyautogui.write(action.text, interval=action.interval or self.type_interval)
        logger.debug(f"Typed {len(action.text)} characters")
        return True
    
    def _execute_key(self, action: KeyAction) -> bool:
        """Execute key press."""
        # Handle hotkeys like "ctrl+c"
        if "+" in action.key:
            keys = action.key.split("+")
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(action.key)
        logger.debug(f"Pressed key: {action.key}")
        return True
    
    def _execute_scroll(self, action: ScrollAction) -> bool:
        """Execute scroll."""
        pyautogui.moveTo(action.x, action.y, duration=self.move_duration)
        
        if action.direction in ["up", "down"]:
            amount = action.amount if action.direction == "up" else -action.amount
            pyautogui.scroll(amount, x=action.x, y=action.y)
        else:
            amount = action.amount if action.direction == "right" else -action.amount
            pyautogui.hscroll(amount, x=action.x, y=action.y)
            
        logger.debug(f"Scrolled {action.direction}")
        return True
    
    def _execute_wait(self, action: WaitAction) -> bool:
        """Execute wait."""
        time.sleep(action.duration)
        return True
    
    def move_to(self, x: int, y: int) -> None:
        """Move mouse without clicking."""
        pyautogui.moveTo(x, y, duration=self.move_duration)
    
    def get_mouse_position(self) -> tuple:
        """Get current mouse position."""
        return pyautogui.position()
