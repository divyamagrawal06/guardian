"""
ATLAS ML Pipeline - LLM via Gemini API
=======================================

Handles reasoning and planning using Google Gemini:
- Intent extraction from user prompts
- Task decomposition into steps
- Action planning
- Error recovery decisions
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json
import time as _time
from loguru import logger

from config import LLMConfig, config

# Global rate limiter: minimum seconds between API calls
_MIN_CALL_INTERVAL = 1.0
_last_call_time = 0.0


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
    Gemini API wrapper for reasoning and planning.
    
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
        """Initialize the Gemini API client."""
        try:
            import google.generativeai as genai
            
            api_key = self.config.api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set. Add it to .env or environment.")
            
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.config.model_name)
            
            logger.info(f"Gemini LLM ready: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def _generate(self, prompt: str, max_tokens: Optional[int] = None, json_mode: bool = True) -> str:
        """Generate completion from Gemini with retry on rate limit."""
        import google.generativeai as genai
        global _last_call_time
        
        if self._model is None:
            self.load()
        
        gen_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=max_tokens or self.config.max_tokens,
        )
        if json_mode:
            gen_config.response_mime_type = "application/json"
        
        last_err = None
        for attempt in range(10):
            # Global rate limiter: ensure minimum gap between calls
            now = _time.time()
            elapsed = now - _last_call_time
            if elapsed < _MIN_CALL_INTERVAL:
                _time.sleep(_MIN_CALL_INTERVAL - elapsed)
            
            try:
                _last_call_time = _time.time()
                response = self._model.generate_content(
                    prompt,
                    generation_config=gen_config,
                )
                return response.text.strip()
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str or "Resource exhausted" in err_str:
                    wait = min(5 * (attempt + 1), 60)  # 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
                    logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt+1}/10)")
                    _time.sleep(wait)
                    continue
                raise
        raise last_err
    
    def extract_intent(self, user_prompt: str) -> Intent:
        """
        Extract structured intent from user's natural language prompt.
        
        PIPELINE STEP 1
        """
        prompt = f"""You are an intent extraction system for a desktop automation agent. 
Extract the user's intent from their request.

User request: "{user_prompt}"

Respond with this JSON schema:
{{
    "goal": "main action in snake_case (e.g. open_app, type_text, search_web)",
    "app": "target application name or null",
    "entities": {{
        "key": "extracted value pairs relevant to the task"
    }}
}}

Extract WHAT the user wants, not HOW to do it."""

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
            # Fallback: use the raw prompt as intent
            return Intent(
                goal=user_prompt.strip().replace(" ", "_")[:50],
                app=None,
                entities={"raw": user_prompt},
                raw_prompt=user_prompt
            )
    
    def create_task_plan(self, intent: Intent) -> List[TaskStep]:
        """
        Convert intent into high-level task steps.
        
        PIPELINE STEP 2
        """
        prompt = f"""You are a task planning system for a desktop automation agent that controls a Windows PC.
Create a step-by-step plan for this task.

The agent CAN:
- Open applications by name (e.g. "notepad", "chrome", "calc", "explorer") or URLs directly
- Press keyboard shortcuts (Ctrl+L, Tab, Enter, Ctrl+C, Ctrl+V, Alt+F4)
- Type text into the currently focused field
- Click on UI elements in the Windows accessibility tree (buttons, links, text fields, menus)
- Scroll
- Wait for UI to load

Original user request: "{intent.raw_prompt}"
Intent:
- Goal: {intent.goal}
- App: {intent.app or "unspecified"}
- Details: {json.dumps(intent.entities)}

IMPORTANT RULES:
1. To launch an application, use a step like "Open notepad" or "Open chrome". The agent can launch them directly.
2. To open a website, use "Open https://youtube.com". The agent opens it in the default browser.
3. ALTERNATIVE: You can also use Win+R to open apps/URLs — press Win+R, wait 1 second, type the name, press Enter.
4. After every open/launch step, ALWAYS add a "Wait 3 seconds" step so the app/page can load.
5. To type text, make sure the target field is focused first (click it or Tab to it), then add a "Type" step.
6. Keep each step atomic — one action per step.
7. Include exact text, URLs, and email addresses from the user's original request.
8. For browser search: after the page loads, click the search bar, type the query, press Enter.
9. Use keyboard shortcuts when reliable: Ctrl+L (address bar), Tab (next field), Enter (confirm).

Respond with a JSON array:
[
    {{"step": 1, "action": "verb phrase", "description": "what to do", "target": null}},
    ...
]

Keep steps atomic and specific. Include exact text/emails/URLs from the user's request."""

        try:
            response = self._generate(prompt)
            data = self._parse_json(response)
            
            if not isinstance(data, list):
                data = data.get("steps", data.get("plan", []))
            
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
        history: Optional[List[str]] = None,
        user_prompt: str = ""
    ) -> PlannedAction:
        """
        Decide the next atomic action based on screen state.
        
        PIPELINE STEP 7
        """
        # Truncate elements list if too long (save tokens)
        elements_display = available_elements[:30]
        elements_str = json.dumps(elements_display, indent=2)
        history_str = "\n".join(history) if history else "None"
        
        prompt = f"""You are an action planning system for a desktop automation agent on Windows.
Based on the current screen state, decide ONE atomic action to perform.

ORIGINAL USER REQUEST: {user_prompt}

Current task step: {current_step.action} - {current_step.description}

Screen state: {screen_description}

Available UI elements (from the Windows accessibility tree — each has role, name, center pixel coordinates, and enabled state):
{elements_str}

Previous actions taken: {history_str}

Choose exactly ONE action. Respond with ONLY a compact JSON object (no explanation, no markdown):
{{
    "action_type": "click" or "type" or "key" or "scroll" or "wait" or "open",
    "target_role": "role of element (e.g. Button, Edit, Hyperlink) or null",
    "target_text": "exact 'name' field of the target element or null",
    "text": "for 'type': text to type. For 'open': app name or URL. Otherwise null.",
    "key": "key to press (e.g. enter, tab, ctrl+c, win+r) or null",
    "direction": "up/down/left/right or null",
    "confidence": 0.0 to 1.0
}}

CRITICAL RULES:
- Use "open" to launch apps or URLs directly.
  Examples: {{"action_type":"open","text":"notepad"}}, {{"action_type":"open","text":"https://youtube.com"}}
- ALTERNATIVELY you can use "key" with "win+r" to open the Run dialog, then "type" and "key" enter. Both approaches work.
- Use "click" ONLY when the target element exists in the available elements list above.
  Match by "name" field. The agent will click the element's center pixel coordinates.
- For keyboard shortcuts, use action_type "key" (e.g. "enter", "tab", "ctrl+c", "ctrl+l").
- For typing text into a focused field, use action_type "type".
- After "open" or "click" on a navigation element, the next action should be "wait" (2-3 seconds).
- If no UI elements match what you need, prefer "key" or "wait" actions.
- key format: "enter", "tab", "escape", "ctrl+c", "ctrl+l", "alt+f4", "win+r"
"""

        try:
            response = self._generate(prompt, max_tokens=8192)
            data = self._parse_json(response)
            
            return PlannedAction(
                action_type=data.get("action_type", "wait"),
                target_role=data.get("target_role"),
                target_text=data.get("target_text"),
                text=data.get("text"),
                key=data.get("key"),
                direction=data.get("direction"),
                confidence=float(data.get("confidence", 0.5))
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
        prompt = f"""You are an error recovery system for a desktop automation agent.

Error: {error_description}
Last action attempted: {last_action.action_type} on {last_action.target_role or last_action.target_text}
Current screen: {screen_state}
Retry count: {retry_count}

Choose a recovery strategy. Respond with JSON:
{{
    "strategy": "retry" or "replan" or "skip" or "abort",
    "reason": "explanation",
    "modification": "what to change if retrying"
}}"""

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
        prompt = f"""Rank these UI elements by how well they match: "{target_description}"

Candidates:
{json.dumps(candidates[:15], indent=2)}

Respond with JSON:
{{"ranking": [index1, index2, ...], "scores": [score1, score2, ...]}}

Indices refer to position in the candidates array. Scores should be 0-1."""

        try:
            response = self._generate(prompt)
            data = self._parse_json(response)
            
            ranking = data.get("ranking", list(range(len(candidates))))
            scores = data.get("scores", [0.5] * len(candidates))
            
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
        """Extract and parse JSON from LLM response, handling malformed output."""
        import re
        
        # Try direct parse first (Gemini JSON mode usually returns clean JSON)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Fallback: find JSON in response
        start = text.find('{') if '{' in text else text.find('[')
        end = max(text.rfind('}'), text.rfind(']')) + 1
        
        if start != -1 and end > start:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # Fix trailing commas (common LLM issue)
            cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        # Last resort: try to repair truncated JSON by closing brackets/braces
        if start != -1:
            json_str = text[start:]
            # Remove incomplete trailing string values: ,"key": "...(truncated)
            json_str = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', json_str)
            # Remove trailing comma and incomplete key-value pairs
            json_str = re.sub(r',\s*"[^"]*"?\s*:?\s*$', '', json_str)
            json_str = re.sub(r',\s*$', '', json_str)
            # Count and close unclosed braces/brackets
            open_braces = json_str.count('{') - json_str.count('}')
            open_brackets = json_str.count('[') - json_str.count(']')
            json_str += '}' * max(0, open_braces) + ']' * max(0, open_brackets)
            # Fix trailing commas again
            cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            
        raise ValueError(f"No valid JSON found in response: {text[:200]}")
