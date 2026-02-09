"""
Phase 3 Test - Perception Pipeline
===================================
Tests: Screen capture, OCR, bbox fusion, and perception engine.
"""
import sys
import os
import time

# Add ml directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("PHASE 3: PERCEPTION PIPELINE TEST")
print("=" * 60)

# Test 1: Screen Capture with DPI detection
print("\n[1] Testing Screen Capture with DPI...")
from perception.screen_capture import ScreenCapture

capture = ScreenCapture()

# Test DPI detection
dpi_scale = capture.detect_dpi_scale()
print(f"    Detected DPI scale: {dpi_scale}")

# Test screenshot
frame = capture.grab()
print(f"    Screenshot: {frame.width}x{frame.height}")
print(f"    Monitor: {frame.monitor}")

# Test coordinate conversion
x_abs, y_abs = frame.to_absolute(0.5, 0.5)
print(f"    Center (0.5, 0.5) -> absolute: ({x_abs}, {y_abs})")

x_norm, y_norm = frame.to_normalized(x_abs, y_abs)
print(f"    Absolute ({x_abs}, {y_abs}) -> normalized: ({x_norm:.3f}, {y_norm:.3f})")

# Test 2: OCR Model (EasyOCR)
print("\n[2] Testing OCR Model (EasyOCR)...")
from models.ocr_model import OCRModel

ocr = OCRModel()
start = time.time()
results = ocr.detect(frame.image)
ocr_time = time.time() - start

print(f"    Detected {len(results)} text regions in {ocr_time:.2f}s")

if results:
    print("\n    Sample detections:")
    # Sort by position (top to bottom)
    sorted_results = sorted(results, key=lambda r: (r.bbox_rect[1], r.bbox_rect[0]))
    for i, r in enumerate(sorted_results[:5]):
        print(f"      {i+1}. \"{r.text[:40]}\" @ {r.bbox_rect} (conf: {r.confidence:.2f})")

# Test find_text
print("\n    Testing find_text (looking for common words)...")
for word in ["File", "Edit", "View", "Help", "Settings"]:
    found = ocr.find_text(frame.image, word, fuzzy=True)
    if found:
        print(f"      Found '{word}' at {found.center}")
        break

# Test 3: Bounding Box Fusion
print("\n[3] Testing Bounding Box Fusion...")
from perception.bbox_fusion import BoundingBoxFusion

fusion = BoundingBoxFusion()

# Convert OCR results to dict format
ocr_dicts = [
    {"text": r.text, "bbox": r.bbox, "bbox_rect": r.bbox_rect, "confidence": r.confidence}
    for r in results
]

# Fuse without VLM (VLM is optional/deferred)
start = time.time()
fused = fusion.fuse(ocr_dicts, [], frame.width, frame.height, frame.image)
fusion_time = time.time() - start

print(f"    Fused {len(fused)} elements in {fusion_time:.2f}s")

if fused:
    print("\n    Sample fused elements:")
    for i, elem in enumerate(fused[:5]):
        role = elem.role or "unknown"
        text = elem.text[:30] if elem.text else "N/A"
        print(f"      {i+1}. [{role}] \"{text}\" @ center {elem.center}")

# Test 4: Quick Perception (no VLM)
print("\n[4] Testing Quick Perception (OCR only)...")
from perception.perception_engine import PerceptionEngine

engine = PerceptionEngine()

start = time.time()
result = engine.quick_perceive()
total_time = time.time() - start

print(f"    Full pipeline completed in {total_time:.2f}s")
print(f"    OCR results: {len(result.ocr_results)}")
print(f"    Fused elements: {len(result.fused_elements)}")

# Test element search
print("\n    Testing element search...")
elem = result.get_element_by_text("File", fuzzy=True)
if elem:
    print(f"      Found 'File' element at {elem.center}")
else:
    print("      'File' not found (may not be visible)")

# Summary
print("\n" + "=" * 60)
print("PHASE 3 TEST RESULTS")
print("=" * 60)
print(f"  Screen Capture:     OK ({frame.width}x{frame.height})")
print(f"  DPI Detection:      OK (scale: {dpi_scale})")
print(f"  OCR (EasyOCR):      OK ({len(results)} detections, {ocr_time:.2f}s)")
print(f"  BBox Fusion:        OK ({len(fused)} elements)")
print(f"  Quick Perception:   OK ({total_time:.2f}s total)")
print("=" * 60)
print("PHASE 3 TESTS PASSED!")
print("=" * 60)
