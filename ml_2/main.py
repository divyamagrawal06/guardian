"""
ATLAS ML Pipeline - Main Entry Point
=====================================

Quick start:
    python main.py "Open Notepad and type hello"
    
Interactive mode:
    python main.py --interactive
"""

import argparse
from loguru import logger

from agent import AgentLoop


def main():
    parser = argparse.ArgumentParser(description="ATLAS Vision-Driven Desktop Agent")
    parser.add_argument("prompt", nargs="?", help="Task to execute")
    parser.add_argument("--interactive", "-i", action="store_true", 
                        help="Interactive mode")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logger.add("logs/agent_{time}.log", level="DEBUG", rotation="10 MB")
    
    agent = AgentLoop()
    
    if args.interactive:
        print("ATLAS Agent - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                prompt = input(">>> ").strip()
                if prompt.lower() in ["quit", "exit", "q"]:
                    break
                if not prompt:
                    continue
                    
                success = agent.run(prompt)
                print(f"Result: {'Success' if success else 'Failed'}\n")
                
            except KeyboardInterrupt:
                print("\nInterrupted")
                break
                
    elif args.prompt:
        success = agent.run(args.prompt)
        exit(0 if success else 1)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
