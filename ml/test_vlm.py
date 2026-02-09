"""
VLM (LLaVA) Test - GPU Required
===============================
Tests Vision-Language Model which was deferred in Phase 2.
Now that the dedicated GPU is ON, we can test this.
"""
import sys
import os
import time

# Add ml directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("VLM (LLaVA) TEST - GPU ENABLED")
print("=" * 60)

# Check GPU availability first
print("\n[0] Checking GPU availability...")
import torch
print(f"    PyTorch version: {torch.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    CUDA version: {torch.version.cuda}")
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("    WARNING: CUDA not available. VLM test may fail or be very slow.")

# Test 1: Import VLM Model
print("\n[1] Testing VLM Model import...")
try:
    from models.vlm_model import VLMModel, UIRegion
    print("    Import successful")
except ImportError as e:
    print(f"    FAILED: {e}")
    sys.exit(1)

# Test 2: Load VLM Model
print("\n[2] Loading VLM Model (this may take a while on first run)...")
try:
    vlm = VLMModel()
    start = time.time()
    vlm.load()
    load_time = time.time() - start
    print(f"    VLM loaded in {load_time:.2f}s")
except Exception as e:
    print(f"    FAILED: {e}")
    print("\n    Possible issues:")
    print("    - Not enough GPU memory (need ~4GB for 4-bit LLaVA)")
    print("    - bitsandbytes not installed for Windows")
    print("    - Model not downloaded yet")
    sys.exit(1)

# Test 3: Screen capture for VLM test
print("\n[3] Capturing screen for VLM test...")
from perception.screen_capture import ScreenCapture
capture = ScreenCapture()
frame = capture.grab()
print(f"    Captured: {frame.width}x{frame.height}")

# Test 4: Describe Screen
print("\n[4] Testing describe_screen()...")
start = time.time()
description = vlm.describe_screen(frame.image)
describe_time = time.time() - start
print(f"    Completed in {describe_time:.2f}s")
print(f"    Description:\n    {description[:200]}...")

# Test 5: Detect UI Elements
print("\n[5] Testing detect_ui_elements()...")
start = time.time()
regions = vlm.detect_ui_elements(frame.image)
detect_time = time.time() - start
print(f"    Detected {len(regions)} UI elements in {detect_time:.2f}s")

if regions:
    print("\n    Sample detected elements:")
    for i, r in enumerate(regions[:5]):
        print(f"      {i+1}. [{r.role}] '{r.description[:40]}' conf={r.confidence:.2f}")

# Test 6: Find specific element
print("\n[6] Testing find_element()...")
for target in ["button", "search", "file", "menu"]:
    start = time.time()
    result = vlm.find_element(frame.image, target)
    find_time = time.time() - start
    if result:
        print(f"    Found '{target}': [{result.role}] at center {result.center_normalized} ({find_time:.2f}s)")
        break
    else:
        print(f"    '{target}' not found ({find_time:.2f}s)")

# Summary
print("\n" + "=" * 60)
print("VLM TEST RESULTS")
print("=" * 60)
print(f"  GPU Available:      {'YES' if torch.cuda.is_available() else 'NO'}")
print(f"  VLM Load Time:      {load_time:.2f}s")
print(f"  Describe Screen:    {describe_time:.2f}s")
print(f"  Detect UI:          {detect_time:.2f}s ({len(regions)} elements)")
print("=" * 60)
print("VLM TESTS PASSED!")
print("=" * 60)
