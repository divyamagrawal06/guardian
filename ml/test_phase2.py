"""Phase 2 Test - Model Testing with EasyOCR"""
import mss
import numpy as np

print("=" * 50)
print("PHASE 2: MODEL TESTING")
print("=" * 50)

# Test 1: Screen Capture
print("\n[1] Testing mss screen capture...")
sct = mss.mss()
monitor = sct.monitors[1]
screenshot = sct.grab(monitor)
img = np.array(screenshot)[:, :, :3]  # Remove alpha
print(f"    OK - Screenshot captured: {img.shape}")

# Test 2: EasyOCR
print("\n[2] Testing EasyOCR...")
import easyocr
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("    OK - EasyOCR model loaded")

print("    Running OCR on screenshot...")
results = reader.readtext(img)

if results:
    print(f"    OK - Detected {len(results)} text regions!")
    print("\n    Sample detections:")
    for i, (bbox, text, conf) in enumerate(results[:5]):
        print(f"      {i+1}. \"{text[:35]}\" (conf: {conf:.2f})")
else:
    print("    No text detected")

# Test 3: llama-cpp-python
print("\n[3] Testing llama-cpp-python import...")
from llama_cpp import Llama
print("    OK - llama-cpp-python imported successfully")
print("    Note: Model file needed for full test")

# Test 4: PyAutoGUI
print("\n[4] Testing PyAutoGUI...")
import pyautogui
pos = pyautogui.position()
size = pyautogui.size()
print(f"    OK - Screen size: {size}")
print(f"    OK - Mouse position: {pos}")

print("\n" + "=" * 50)
print("PHASE 2 TESTS COMPLETED!")
print("=" * 50)
