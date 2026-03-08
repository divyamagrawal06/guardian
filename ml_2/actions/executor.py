"""
ATLAS ML Pipeline - Action Executor
====================================
Executes actions via OS-level input (PyAutoGUI).
PIPELINE STEP 9
"""

import pyautogui
import subprocess
import time
import os
from typing import Optional
from loguru import logger

from actions.actions import Action, ClickAction, TypeAction, KeyAction, ScrollAction, WaitAction, OpenAction
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
            elif isinstance(action, OpenAction):
                return self._execute_open(action)
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
        """Type text via clipboard paste (reliable across all apps and keyboard layouts)."""
        text = action.text
        if not text:
            return True
        try:
            self._clipboard_type(text)
            logger.debug(f"Typed (clipboard paste) {len(text)} chars")
        except Exception as e:
            logger.warning(f"Clipboard paste failed, falling back to pyautogui: {e}")
            pyautogui.write(text, interval=action.interval or self.type_interval)
            logger.debug(f"Typed (key-by-key) {len(text)} chars")
        return True
    
    def _execute_key(self, action: KeyAction) -> bool:
        """Execute key press with normalized key names."""
        key = action.key.lower().strip()
        _alias = {
            'windows': 'win', 'cmd': 'win', 'command': 'win',
            'control': 'ctrl', 'return': 'enter', 'esc': 'escape',
            'delete': 'del',
        }
        if '+' in key:
            keys = [_alias.get(k.strip(), k.strip()) for k in key.split('+')]
            pyautogui.hotkey(*keys)
        else:
            key = _alias.get(key, key)
            pyautogui.press(key)
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
        duration = min(action.duration, 15.0)
        time.sleep(duration)
        return True
    
    def _execute_open(self, action: OpenAction) -> bool:
        """Open an application or URL using the OS shell."""
        target = action.target.strip()
        if not target:
            logger.error("Open action: no target specified")
            return False
        # Normalize URLs
        if target.startswith('www.'):
            target = f'https://{target}'
        
        logger.info(f"Opening: {target}")
        
        # Try os.startfile first (most reliable on Windows)
        try:
            os.startfile(target)
            logger.info(f"Opened via os.startfile: {target}")
            return True
        except Exception as e1:
            logger.warning(f"os.startfile failed: {e1}")
        
        # Fallback: subprocess start command
        try:
            subprocess.Popen(f'start "" "{target}"', shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Opened via start command: {target}")
            return True
        except Exception as e2:
            logger.error(f"Both open methods failed for '{target}': {e2}")
            return False
    
    def _clipboard_type(self, text: str) -> None:
        """Copy text to clipboard and paste via Ctrl+V."""
        try:
            import win32clipboard
            import win32con
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
        except ImportError:
            # pywin32 not installed — fall back to PowerShell
            p = subprocess.Popen(
                ['powershell', '-NoProfile', '-Command', 'Set-Clipboard -Value $input'],
                stdin=subprocess.PIPE,
            )
            p.communicate(text.encode('utf-8'))
        time.sleep(0.05)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.1)
    
    def get_mouse_position(self) -> tuple:
        """Get current mouse position."""
        return pyautogui.position()
