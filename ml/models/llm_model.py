"""
ATLAS ML Pipeline - Local LLM (Mistral/Phi)
===========================================

Handles reasoning and planning:
- Intent extraction from user prompts
- Task decomposition into steps
- Action planning
- Error recovery decisions
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json
from loguru import logger

from config import LLMConfig, config


@dataclass
class Intent:
    """Structured intent from user prompt."""
    goal: str
    app: Optional[str]
    entities: Dict[str, Any]
    raw_prompt: str


@dataclass
class TaskStep:
    """Single step in a task plan."""
    step_number: int
    action: str
    description: str
    target: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class PlannedAction:
    """Single atomic action to execute."""
    action_type: str  # "click", "type", "key", "scroll", "wait"
    target_role: Optional[str] = None
    target_text: Optional[str] = None
    text: Optional[str] = None  # For typing
    key: Optional[str] = None  # For key press
    direction: Optional[str] = None  # For scroll
    confidence: float = 0.0


class LLMModel:
    """
    Local LLM wrapper using llama.cpp.
    
    Handles all reasoning tasks:
    - Intent interpretation
    - Task planning
    - Action selection
    - Error recovery
    
    Usage:
        llm = LLMModel()
        intent = llm.extract_intent("Open notepad and write hello")
        steps = llm.create_task_plan(intent)
        action = llm.plan_action(current_state, next_step)
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.config = llm_config or config.llm
        self._model = None
        
    def load(self) -> None:
        """Initialize the local LLM via llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            self._model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=False,
            )
            
            logger.info(f"LLM loaded: {self.config.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
    
    def _generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate completion from prompt."""
        if self._model is None:
            self.load()
            
        output = self._model(
            prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            stop=["</s>", "[/INST]", "\n\n\n"],
        )
        
        return output["choices"][0]["text"].strip()
    
    def extract_intent(self, user_prompt: str) -> Intent:
        """
        Extract structured intent from user's natural language prompt.
        
        PIPELINE STEP 1
        """
        prompt = f"""[INST] You are an intent extraction system. Extract the user's intent from their request.

User request: "{user_prompt}"

Respond with JSON only:
{{
    "goal": "main action in snake_case",
    "app": "target application name or null",
    "entities": {{
        "key": "extracted value"
    }}
}}

Extract WHAT the user wants, not HOW to do it. [/INST]
"""

        try:
            response = self._generate(prompt)
            data = self._parse_json(response)
            
            return Intent(
                goal=data.get("goal", "unknown"),
                app=data.get("app"),
                entities=data.get("entities", {}),
                raw_prompt=user_prompt
            )
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            return Intent(
                goal="unknown",
                app=None,
                entities={},
                raw_prompt=user_prompt
            )
    
    def create_task_plan(self, intent: Intent) -> List[TaskStep]:
        """
        Convert intent into high-level task steps.
        
        PIPELINE STEP 2
        """
        prompt = f"""[INST] You are a task planning system. Create a step-by-step plan for this task.

Intent:
- Goal: {intent.goal}
- App: {intent.app or "unspecified"}
- Details: {json.dumps(intent.entities)}

Create abstract steps (no coordinates, no UI specifics).

Respond with JSON array only:
[
    {{"step": 1, "action": "verb phrase", "description": "what to do", "target": "element name or null"}},
    ...
]

Keep steps atomic and verifiable. [/INST]
"""

        try:
            response = self._generate(prompt)
            data = self._parse_json(response)
            
            steps = []
            for item in data:
                steps.append(TaskStep(
                    step_number=item.get("step", len(steps) + 1),
                    action=item.get("action", ""),
                    description=item.get("description", ""),
                    target=item.get("target"),
                    parameters=item.get("parameters")
                ))
            
            return steps
            
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            return []
    
    def plan_action(
        self, 
        screen_description: str,
        available_elements: List[Dict[str, Any]],
        current_step: TaskStep,
        history: Optional[List[str]] = None
    ) -> PlannedAction:
        """
        Decide the next atomic action based on screen state.
        
        PIPELINE STEP 7
        """
        elements_str = json.dumps(available_elements, indent=2)
        history_str = "\n".join(history) if history else "None"
        
        prompt = f"""[INST] You are an action planning system. Decide ONE atomic action to perform.

Current task step: {current_step.action} - {current_step.description}

Screen state: {screen_description}

Available UI elements:
{elements_str}

Previous actions: {history_str}

Choose ONE action. Respond with JSON only:
{{
    "action_type": "click|type|key|scroll|wait",
    "target_role": "role of element to interact with or null",
    "target_text": "visible text on element or null",
    "text": "text to type or null",
    "key": "key to press or null",
    "direction": "up|down|left|right or null",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

[/INST]
"""

        try:
            response = self._generate(prompt)
            data = self._parse_json(response)
            
            return PlannedAction(
                action_type=data.get("action_type", "wait"),
                target_role=data.get("target_role"),
                target_text=data.get("target_text"),
                text=data.get("text"),
                key=data.get("key"),
                direction=data.get("direction"),
                confidence=data.get("confidence", 0.5)
            )
            
        except Exception as e:
            logger.error(f"Action planning failed: {e}")
            return PlannedAction(action_type="wait", confidence=0.0)
    
    def handle_error(
        self,
        error_description: str,
        screen_state: str,
        last_action: PlannedAction,
        retry_count: int
    ) -> Dict[str, Any]:
        """
        Decide how to handle an error/failed verification.
        
        Returns recovery strategy.
        """
        prompt = f"""[INST] You are an error recovery system. Decide how to recover from this failure.

Error: {error_description}
Last action attempted: {last_action.action_type} on {last_action.target_role or last_action.target_text}
Current screen: {screen_state}
Retry count: {retry_count}

Choose a recovery strategy. Respond with JSON only:
{{
    "strategy": "retry|replan|skip|abort",
    "reason": "explanation",
    "modification": "what to change if retrying"
}}

[/INST]
"""

        try:
            response = self._generate(prompt)
            return self._parse_json(response)
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            return {"strategy": "abort", "reason": str(e)}
    
    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        target_description: str
    ) -> List[Dict[str, Any]]:
        """
        Rank UI element candidates by relevance to target.
        
        Used in PIPELINE STEP 6 for bounding box fusion.
        """
        prompt = f"""[INST] Rank these UI elements by how well they match: "{target_description}"

Candidates:
{json.dumps(candidates, indent=2)}

Respond with JSON array of indices sorted by relevance (most relevant first):
{{"ranking": [index1, index2, ...], "scores": [score1, score2, ...]}}

Scores should be 0-1. [/INST]
"""

        try:
            response = self._generate(prompt)
            data = self._parse_json(response)
            
            ranking = data.get("ranking", list(range(len(candidates))))
            scores = data.get("scores", [0.5] * len(candidates))
            
            # Reorder candidates
            ranked = []
            for idx, score in zip(ranking, scores):
                if idx < len(candidates):
                    candidate = candidates[idx].copy()
                    candidate["relevance_score"] = score
                    ranked.append(candidate)
                    
            return ranked
            
        except Exception as e:
            logger.error(f"Candidate ranking failed: {e}")
            return candidates
    
    def _parse_json(self, text: str) -> Any:
        """Extract and parse JSON from LLM response."""
        # Find JSON in response
        start = text.find('{') if '{' in text else text.find('[')
        end = max(text.rfind('}'), text.rfind(']')) + 1
        
        if start != -1 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
            
        raise ValueError("No valid JSON found in response")
