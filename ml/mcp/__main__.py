"""
ATLAS MCP Agent — CLI Entry Point
===================================

Run with:
    cd ml/mcp
    python -m mcp "Go to google.com and search for ATLAS AI"

Or interactive mode:
    python -m mcp

Or directly:
    python __main__.py "your task here"
"""

import asyncio
import sys
import os

# Ensure this package's directory is on the path (for isolated running)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config import get_config
from browser_agent import BrowserAgent

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]ATLAS MCP Agent[/bold cyan]\n"
        "[dim]Autonomous browser automation via MCP + LLM[/dim]\n\n"
        "Commands:\n"
        "  [green]Type a task[/green]  — Execute it in the browser\n"
        "  [green]config[/green]      — Show current configuration\n"
        "  [green]quit / exit[/green] — Exit\n",
        title="🌐",
        border_style="cyan",
    ))


def print_config(cfg):
    console.print(f"  [cyan]LLM Backend:[/cyan]  {cfg.llm_backend}")
    if cfg.llm_backend == "openai":
        console.print(f"  [cyan]Model:[/cyan]        {cfg.openai.model}")
        console.print(f"  [cyan]Base URL:[/cyan]     {cfg.openai.base_url or 'https://api.openai.com/v1'}")
        key_preview = cfg.openai.api_key[:8] + "..." if cfg.openai.api_key else "(not set)"
        console.print(f"  [cyan]API Key:[/cyan]      {key_preview}")
    else:
        console.print(f"  [cyan]Model Path:[/cyan]   {cfg.llama.model_path}")
    console.print(f"  [cyan]Headless:[/cyan]     {cfg.playwright_headless}")
    console.print(f"  [cyan]Max Steps:[/cyan]    {cfg.max_steps}")
    console.print(f"  [cyan]Debug:[/cyan]        {cfg.debug}")


async def interactive_mode():
    """Interactive REPL — enter tasks one at a time."""
    print_banner()
    
    cfg = get_config()
    print_config(cfg)
    console.print()
    
    while True:
        try:
            task = Prompt.ask("\n[bold cyan]Task[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        
        task = task.strip()
        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if task.lower() == "config":
            cfg = get_config()
            print_config(cfg)
            continue
        
        try:
            agent = BrowserAgent(cfg)
            await agent.run(task)
        except KeyboardInterrupt:
            console.print("\n[yellow]Task cancelled.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if cfg.debug:
                console.print_exception()


async def single_task_mode(task: str):
    """Execute a single task and exit."""
    cfg = get_config()
    agent = BrowserAgent(cfg)
    await agent.run(task)


def main():
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        asyncio.run(single_task_mode(task))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
