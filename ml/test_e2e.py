"""
ATLAS ML Pipeline - End-to-End Test
=====================================
Tests the full pipeline: prompt -> intent -> plan -> perceive -> act -> verify

Uses a simple safe task: Open Notepad via Windows Search.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("END-TO-END PIPELINE TEST")
print("=" * 60)

# Step 1: Import and construct
print("\n[1] Importing AgentLoop...")
from agent.agent_loop import AgentLoop
from agent.state import AgentStatus

agent = AgentLoop()
print(f"    LLM backend: {agent.llm.config.backend} ({agent.llm.config.ollama_model})")
print(f"    VLM enabled: {agent.perception._enable_vlm}")
print(f"    Memory enabled: {agent.memory.enabled}")

# Step 2: Initialize
print("\n[2] Initializing all models...")
t0 = time.time()
agent.initialize()
print(f"    Initialized in {time.time() - t0:.1f}s")

# Step 3: Run a simple task
PROMPT = "Open Notepad"
print(f"\n[3] Running task: '{PROMPT}'")
print("    (This will use Win+S search, type 'notepad', press Enter)")
print("    Watch your screen - the agent will control mouse/keyboard!")
print()

t0 = time.time()
success = agent.run(PROMPT)
elapsed = time.time() - t0

# Step 4: Report results
print(f"\n[4] Results:")
print(f"    Success: {success}")
print(f"    Status: {agent.state.status}")
print(f"    Duration: {elapsed:.1f}s")
print(f"    Steps planned: {len(agent.state.task_steps)}")
print(f"    Actions executed: {len(agent.state.action_history)}")

if agent.state.task_steps:
    print(f"\n    Plan:")
    for s in agent.state.task_steps:
        print(f"      {s.step_number}. {s.action}: {s.description}")

if agent.state.action_history:
    print(f"\n    Action History:")
    for i, record in enumerate(agent.state.action_history):
        status = "VERIFIED" if record.verification_passed else "FAILED"
        err = f" ({record.error})" if record.error else ""
        print(f"      {i+1}. {record.action.action_type} -> {status}{err}")

if agent.state.last_error:
    print(f"\n    Last Error: {agent.state.last_error}")

# Cleanup
agent.stop()

print("\n" + "=" * 60)
if success:
    print("END-TO-END TEST PASSED")
else:
    print("END-TO-END TEST COMPLETED (check results above)")
print("=" * 60)
