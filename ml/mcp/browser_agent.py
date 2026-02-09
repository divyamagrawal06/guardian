"""
ATLAS MCP Agent — Browser Agent
================================

End-to-end autonomous browser agent that:
  1. Launches a Playwright MCP server (browser automation tools)
  2. Feeds available tools to the LLM
  3. Runs an agentic loop: LLM decides actions → executes via MCP → observes → repeats
  4. Completes the user's task or reports failure

This is fully self-contained — no dependencies on the rest of the ATLAS pipeline.
"""

from __future__ import annotations
import asyncio
import json
import sys
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCPConfig, get_config
from llm_backend import LLMBackend, create_llm

console = Console()


# ─── Playwright MCP Server Parameters ─────────────────────────────────────────

def get_playwright_server_params(headless: bool = False) -> StdioServerParameters:
    """Get the params to launch @playwright/mcp as a subprocess."""
    args = ["@playwright/mcp@latest"]
    if headless:
        args.append("--headless")
    return StdioServerParameters(
        command="npx",
        args=args,
    )


# ─── Tool Schema Conversion ───────────────────────────────────────────────────

def mcp_tools_to_openai_tools(mcp_tools: list) -> List[Dict[str, Any]]:
    """Convert MCP tool schemas to OpenAI function-calling format."""
    openai_tools = []
    for tool in mcp_tools:
        fn = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
            },
        }
        if tool.inputSchema:
            # Clean up the schema — remove unsupported keys for OpenAI
            schema = dict(tool.inputSchema)
            schema.pop("additionalProperties", None)
            schema.pop("$schema", None)
            fn["function"]["parameters"] = schema
        else:
            fn["function"]["parameters"] = {"type": "object", "properties": {}}
        openai_tools.append(fn)
    return openai_tools


# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are ATLAS, an autonomous browser agent. You complete tasks by controlling a web browser.

RULES:
1. You can use browser tools to navigate, click, type, and interact with web pages.
2. ALWAYS start by using browser_snapshot to see the current page state.
3. Use the element references (ref="...") from snapshots to interact with elements.
4. After each action, take a new snapshot to verify what happened.
5. Think step-by-step. Break complex tasks into simple actions.
6. If something fails, try an alternative approach.
7. When the task is COMPLETE, respond with the final result in plain text (no tool call).
8. NEVER fabricate information — only report what you actually see on the page.
9. Keep interactions minimal and efficient.

IMPORTANT: Element references from browser_snapshot are the primary way to target elements.
Use browser_click with the "ref" parameter, and browser_type with "ref" and "text" parameters.
"""


# ─── Agent Loop ────────────────────────────────────────────────────────────────

class BrowserAgent:
    """
    Autonomous browser agent powered by MCP + LLM.
    
    Connects to Playwright MCP server, discovers tools, then runs
    an LLM-driven action loop to complete the user's task.
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or get_config()
        self.llm: LLMBackend = create_llm(self.config)
        self.messages: List[Dict[str, str]] = []
        self.step_count = 0
    
    async def run(self, task: str) -> str:
        """
        Execute a task end-to-end.
        
        Launches the Playwright MCP server, connects, discovers tools,
        and runs the agentic loop until completion or max steps.
        
        Args:
            task: Natural language task description
            
        Returns:
            Final result message from the agent
        """
        console.print(Panel(f"[bold cyan]Task:[/bold cyan] {task}", title="🧠 ATLAS MCP Agent"))
        console.print(f"[dim]LLM Backend: {self.llm.name()}[/dim]")
        console.print(f"[dim]Headless: {self.config.playwright_headless}[/dim]\n")
        
        server_params = get_playwright_server_params(self.config.playwright_headless)
        
        console.print("[yellow]Starting Playwright MCP server...[/yellow]")
        
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the MCP connection
                await session.initialize()
                console.print("[green]✓ Connected to Playwright MCP server[/green]")
                
                # Discover available tools
                tools_result = await session.list_tools()
                mcp_tools = tools_result.tools
                openai_tools = mcp_tools_to_openai_tools(mcp_tools)
                
                self._print_available_tools(mcp_tools)
                
                # Build initial messages
                self.messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ]
                
                # ── Agentic Loop ──
                result = await self._agent_loop(session, openai_tools)
                
                console.print(Panel(
                    f"[bold green]{result}[/bold green]",
                    title="✅ Task Complete",
                ))
                return result
    
    async def _agent_loop(
        self,
        session: ClientSession,
        tools: List[Dict[str, Any]],
    ) -> str:
        """
        Core agent loop:
          1. Send messages + tools to LLM
          2. If LLM returns tool calls → execute them via MCP → add results → repeat
          3. If LLM returns text (no tool calls) → task is done
        """
        while self.step_count < self.config.max_steps:
            self.step_count += 1
            
            console.print(f"\n[bold]─── Step {self.step_count}/{self.config.max_steps} ───[/bold]")
            
            # Call LLM
            try:
                response = self.llm.chat(self.messages, tools=tools)
            except Exception as e:
                console.print(f"[red]LLM error: {e}[/red]")
                return f"Error: LLM call failed — {e}"
            
            # Case 1: LLM wants to call tools
            if response.get("tool_calls"):
                # Add assistant message with tool calls to history
                self.messages.append({
                    "role": "assistant",
                    "content": response.get("content"),
                    "tool_calls": response["tool_calls"],
                })
                
                # Execute each tool call via MCP
                for tc in response["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args_raw = tc["function"]["arguments"]
                    
                    # Parse arguments
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except json.JSONDecodeError:
                        fn_args = {}
                    
                    console.print(f"  [cyan]→ {fn_name}[/cyan]({_format_args(fn_args)})")
                    
                    # Execute via MCP
                    try:
                        mcp_result = await session.call_tool(fn_name, fn_args)
                        result_text = _extract_mcp_result(mcp_result)
                        
                        # Truncate very long results for display
                        display_text = result_text[:500] + "..." if len(result_text) > 500 else result_text
                        console.print(f"  [green]✓[/green] [dim]{display_text}[/dim]")
                        
                    except Exception as e:
                        result_text = f"Error: {e}"
                        console.print(f"  [red]✗ {result_text}[/red]")
                    
                    # Add tool result to message history
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                    })
            
            # Case 2: LLM returns plain text → task is done
            else:
                final_text = response.get("content", "Task completed.")
                console.print(f"\n  [bold green]Agent:[/bold green] {final_text}")
                return final_text
        
        return f"Stopped after {self.config.max_steps} steps (max limit reached)."
    
    def _print_available_tools(self, tools: list) -> None:
        """Pretty-print discovered MCP tools."""
        table = Table(title="Available Browser Tools", show_lines=False)
        table.add_column("Tool", style="cyan", min_width=30)
        table.add_column("Description", style="dim")
        for t in tools:
            desc = (t.description or "")[:80]
            table.add_row(t.name, desc)
        console.print(table)
        console.print(f"[dim]Total: {len(tools)} tools[/dim]\n")


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _format_args(args: dict) -> str:
    """Format tool arguments for console display."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 60:
            val = val[:60] + "…"
        parts.append(f"{k}={val}")
    return ", ".join(parts)


def _extract_mcp_result(result) -> str:
    """Extract text from MCP tool result."""
    if hasattr(result, "content"):
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif hasattr(item, "data"):
                parts.append(f"[binary data: {len(item.data)} bytes]")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)


# ─── Standalone runner ─────────────────────────────────────────────────────────

async def run_agent(task: str, config: Optional[MCPConfig] = None) -> str:
    """Run the browser agent on a task. Convenience function."""
    agent = BrowserAgent(config)
    return await agent.run(task)


if __name__ == "__main__":
    # Quick test
    task = " ".join(sys.argv[1:]) or "Go to google.com and tell me the page title"
    asyncio.run(run_agent(task))
