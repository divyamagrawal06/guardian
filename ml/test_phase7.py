"""
Phase 7 Test - End-to-End Testing (Simple Task)
===============================================
Tests the full system with a real LLM on a simple task.
Goal: Open Start Menu, Type 'notepad', Open Notepad.
"""
import sys
import os
import time
import pyautogui

# Add ml directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("PHASE 7: END-TO-END TEST (REAL LLM)")
print("=" * 60)

try:
    from agent.agent_loop import AgentLoop
    from agent.state import AgentStatus
    print("[0] Imports successful")
except ImportError as e:
    print(f"FAILED: {e}")
    sys.exit(1)

# Initialize Agent
print("\n[1] Initializing AgentLoop (Loading Real Models)...")
agent = AgentLoop()

# Verify model path (only for local GGUF)
if agent.llm.config.backend != "ollama":
    model_path = agent.llm.config.model_path
    if not os.path.exists(model_path):
        print(f"FAILED: Model not found at {model_path}")
        print("Please download the model first (run ml/download_model.py)")
        sys.exit(1)
    print(f"    Using local model: {model_path}")
else:
    print(f"    Using Ollama model: {agent.llm.config.ollama_model}")

# Run the Agent
PROMPT = "Open Notepad and type 3 lines about VIT Vellore"
print(f"\n[2] Running Agent with prompt: '{PROMPT}'...")

start_time = time.time()
success = agent.run(PROMPT)
end_time = time.time()

print(f"\n[3] Agent finished in {end_time - start_time:.2f}s")
print(f"    Success: {success}")
print(f"    Final Status: {agent.state.status}")
print(f"    History: {len(agent.state.action_history)} actions recorded")

# Print action history
print("\nAction History:")
for i, record in enumerate(agent.state.action_history):
    status = "VERIFIED" if record.verification_passed else "FAILED"
    print(f"  {i+1}. {record.action.action_type} -> {status} ({record.error or 'OK'})")
    if hasattr(record.action, 'reasoning') and record.action.reasoning:
         print(f"     Reasoning: {record.action.reasoning}")

print("\nWaiting 5 seconds for visual verification...")
time.sleep(5)

if success and agent.state.status == AgentStatus.COMPLETED:
    print("\n" + "=" * 60)
    print("PHASE 7 TEST PASSED! VERIFY SCREENSHOTS.")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("PHASE 7 TEST FAILED - CHECK LOGS")
    print("=" * 60)
    if agent.state.last_error:
        print(f"Last Error: {agent.state.last_error}")
