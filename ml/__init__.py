"""
ATLAS - Vision-Driven Desktop Agent
====================================

A closed-loop autonomous agent that operates any application
using only visual perception (screen pixels).

Core Loop: PERCEIVE → UNDERSTAND → PLAN → ACT → VERIFY → repeat

Usage:
    from ml import AgentLoop
    
    agent = AgentLoop()
    agent.run("Open Notepad and type hello world")
"""

from agent import AgentLoop, AgentState
from models import OCRModel, VLMModel, LLMModel
from perception import PerceptionEngine, ScreenCapture, BoundingBoxFusion
from actions import ActionExecutor
from memory import Memory
from config import config, load_config

__version__ = "0.1.0"

__all__ = [
    "AgentLoop",
    "AgentState",
    "OCRModel",
    "VLMModel", 
    "LLMModel",
    "PerceptionEngine",
    "ScreenCapture",
    "BoundingBoxFusion",
    "ActionExecutor",
    "Memory",
    "config",
    "load_config",
]
