"""
ATLAS MCP Agent — Modular LLM Backend
======================================

Provides a unified interface for LLM calls with two backends:
  1. OpenAI-compatible API (works with OpenAI, Groq, Together, Ollama, etc.)
  2. Local llama-cpp-python (fully offline)

Usage:
    from llm_backend import create_llm
    llm = create_llm(config)
    response = llm.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from rich.console import Console

from config import MCPConfig, OpenAIConfig, LlamaConfig

console = Console()


class LLMBackend(ABC):
    """Abstract LLM interface — swap backends without changing agent code."""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.
        
        Returns dict with:
            - role: "assistant"
            - content: str or None
            - tool_calls: list of tool call dicts (if tools were provided)
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible API backend (works with any OpenAI-shaped endpoint)."""
    
    def __init__(self, cfg: OpenAIConfig):
        self.cfg = cfg
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.cfg.api_key}
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            self._client = OpenAI(**kwargs)
        return self._client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        client = self._get_client()
        
        kwargs = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature or self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        
        result = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": None,
        }
        
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        
        return result
    
    def name(self) -> str:
        base = self.cfg.base_url or "api.openai.com"
        return f"OpenAI ({self.cfg.model} @ {base})"


class LlamaBackend(LLMBackend):
    """Local llama-cpp-python backend for fully offline use."""
    
    def __init__(self, cfg: LlamaConfig):
        self.cfg = cfg
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Install it with: pip install llama-cpp-python"
                )
            console.print(f"[dim]Loading local model: {self.cfg.model_path}[/dim]")
            self._model = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_gpu_layers=self.cfg.n_gpu_layers,
                verbose=False,
                chat_format="chatml",  # Works with most instruct models
            )
            console.print("[green]✓ Local model loaded[/green]")
        return self._model
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        model = self._get_model()
        
        # If tools are provided, inject them into the system prompt
        # since llama.cpp chat doesn't natively support tool_choice
        if tools:
            messages = self._inject_tools_into_prompt(messages, tools)
        
        response = model.create_chat_completion(
            messages=messages,
            temperature=temperature or self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        
        msg = response["choices"][0]["message"]
        
        result = {
            "role": "assistant",
            "content": msg.get("content", ""),
            "tool_calls": None,
        }
        
        # Try to parse tool calls from the response text
        if tools and result["content"]:
            parsed = self._try_parse_tool_call(result["content"])
            if parsed:
                result["tool_calls"] = [parsed]
                result["content"] = None
        
        return result
    
    def _inject_tools_into_prompt(
        self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Add tool descriptions to the system prompt for models without native tool support."""
        tool_desc = "You have access to these tools. To call a tool, respond with EXACTLY this JSON format and nothing else:\n"
        tool_desc += '{"tool": "<tool_name>", "arguments": {<args>}}\n\n'
        tool_desc += "Available tools:\n"
        for t in tools:
            fn = t["function"]
            tool_desc += f"\n### {fn['name']}\n{fn.get('description', '')}\n"
            if "parameters" in fn:
                props = fn["parameters"].get("properties", {})
                required = fn["parameters"].get("required", [])
                for pname, pinfo in props.items():
                    req_mark = " (REQUIRED)" if pname in required else ""
                    tool_desc += f"  - {pname}: {pinfo.get('description', pinfo.get('type', 'any'))}{req_mark}\n"
        
        # Prepend to system message or create one
        messages = list(messages)  # copy
        if messages and messages[0]["role"] == "system":
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n\n" + tool_desc,
            }
        else:
            messages.insert(0, {"role": "system", "content": tool_desc})
        
        return messages
    
    def _try_parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract a tool call JSON from the model's text output."""
        import re
        # Look for JSON blocks
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*"tool"[^{}]*\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if "tool" in data:
                        return {
                            "id": "local_call_0",
                            "function": {
                                "name": data["tool"],
                                "arguments": json.dumps(data.get("arguments", {})),
                            },
                        }
                except json.JSONDecodeError:
                    continue
        return None
    
    def name(self) -> str:
        return f"Llama.cpp ({self.cfg.model_path})"


def create_llm(config: MCPConfig) -> LLMBackend:
    """Factory: create the right LLM backend based on config."""
    if config.llm_backend == "openai":
        if not config.openai.api_key:
            raise ValueError(
                "LLM_BACKEND=openai but OPENAI_API_KEY is not set.\n"
                "Set it in ml/mcp/.env or as an environment variable."
            )
        return OpenAIBackend(config.openai)
    elif config.llm_backend == "llama":
        return LlamaBackend(config.llama)
    else:
        raise ValueError(f"Unknown LLM backend: {config.llm_backend}")
