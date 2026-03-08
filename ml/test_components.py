"""
ATLAS ML Pipeline - Component Tests
=====================================
Tests each component individually to verify they work before running end-to-end.

Components tested:
1. Config loading
2. OCR Model (EasyOCR)
3. Screen Capture (mss)
4. BBox Fusion
5. Perception Engine
6. Action definitions + executor (safe - no real clicks)
7. LLM Model (Ollama)
8. VLM Model (Ollama)
9. Verification logic
10. Memory (SQLite)
11. Agent State
12. Agent Loop (import + init only)
"""

import sys
import os
import time
import traceback

# Add ml directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = {}

def test(name):
    """Decorator to register and run a test."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print(f"{'='*60}")
            try:
                fn()
                RESULTS[name] = "PASS"
                print(f"  >> PASS")
            except Exception as e:
                RESULTS[name] = f"FAIL: {e}"
                print(f"  >> FAIL: {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ============================================================
# 1. Config
# ============================================================
@test("Config Loading")
def test_config():
    from config import config, PipelineConfig
    assert isinstance(config, PipelineConfig), "config is not PipelineConfig"
    print(f"  LLM backend: {config.llm.backend}")
    print(f"  LLM model: {config.llm.ollama_model}")
    print(f"  VLM backend: {config.vlm.backend}")
    print(f"  VLM model: {config.vlm.ollama_model}")
    print(f"  VLM enabled: {config.vlm.use_vlm}")
    print(f"  OCR lang: {config.ocr.lang}")
    print(f"  Screen monitor: {config.screen.capture_monitor}")

test_config()


# ============================================================
# 2. Screen Capture
# ============================================================
@test("Screen Capture (mss)")
def test_screen_capture():
    from perception.screen_capture import ScreenCapture, ScreenFrame
    cap = ScreenCapture()
    frame = cap.grab()
    assert isinstance(frame, ScreenFrame), "Not a ScreenFrame"
    assert frame.width > 0 and frame.height > 0, f"Invalid size: {frame.width}x{frame.height}"
    assert frame.image is not None, "Image is None"
    assert frame.image.shape[0] == frame.height, f"Height mismatch: {frame.image.shape[0]} vs {frame.height}"
    assert frame.image.shape[1] == frame.width, f"Width mismatch: {frame.image.shape[1]} vs {frame.width}"
    print(f"  Resolution: {frame.width}x{frame.height}")
    print(f"  DPI scale: {frame.dpi_scale}")
    print(f"  Image shape: {frame.image.shape}")
    
    # Test coordinate conversion
    ax, ay = frame.to_absolute(0.5, 0.5)
    nx, ny = frame.to_normalized(ax, ay)
    assert abs(nx - 0.5) < 0.01, f"Normalized X roundtrip failed: {nx}"
    assert abs(ny - 0.5) < 0.01, f"Normalized Y roundtrip failed: {ny}"
    print(f"  Coordinate roundtrip: OK")

test_screen_capture()


# ============================================================
# 3. OCR Model
# ============================================================
@test("OCR Model (EasyOCR)")
def test_ocr():
    from models.ocr_model import OCRModel
    from perception.screen_capture import ScreenCapture
    
    ocr = OCRModel()
    ocr.load()
    
    cap = ScreenCapture()
    frame = cap.grab()
    
    t0 = time.time()
    results = ocr.detect(frame.image)
    elapsed = time.time() - t0
    
    assert isinstance(results, list), "Results not a list"
    print(f"  Detected {len(results)} text regions in {elapsed:.1f}s")
    if results:
        sample = results[0]
        print(f"  Sample: text='{sample.text}', confidence={sample.confidence:.2f}, bbox={sample.bbox_rect}")
    assert len(results) > 0, "No text detected on screen (is something visible?)"

test_ocr()


# ============================================================
# 4. BBox Fusion
# ============================================================
@test("BBox Fusion")
def test_fusion():
    from perception.bbox_fusion import BoundingBoxFusion, FusedElement
    
    fusion = BoundingBoxFusion()
    
    # Fake OCR results
    ocr_results = [
        {"text": "File", "bbox": [[10,10],[50,10],[50,30],[10,30]], "bbox_rect": (10,10,50,30), "confidence": 0.95},
        {"text": "Edit", "bbox": [[60,10],[100,10],[100,30],[60,30]], "bbox_rect": (60,10,100,30), "confidence": 0.90},
        {"text": "OK", "bbox": [[400,500],[440,500],[440,520],[400,520]], "bbox_rect": (400,500,440,520), "confidence": 0.88},
    ]
    vlm_regions = []
    
    fused = fusion.fuse(ocr_results, vlm_regions, 1920, 1080, None)
    assert isinstance(fused, list), "Fused not a list"
    assert len(fused) >= 3, f"Expected >= 3 fused elements, got {len(fused)}"
    
    for e in fused:
        assert isinstance(e, FusedElement), "Not a FusedElement"
        assert len(e.bbox_normalized) == 4, "bbox_normalized wrong length"
    
    print(f"  Fused {len(fused)} elements from {len(ocr_results)} OCR results")
    for e in fused:
        print(f"    role={e.role}, text={e.text}, conf={e.confidence:.2f}")

test_fusion()


# ============================================================
# 5. Perception Engine (quick_perceive)
# ============================================================
@test("Perception Engine (quick_perceive)")
def test_perception():
    from perception.perception_engine import PerceptionEngine, PerceptionResult
    
    engine = PerceptionEngine(enable_vlm=False)
    engine.initialize_ocr()
    
    t0 = time.time()
    result = engine.quick_perceive()
    elapsed = time.time() - t0
    
    assert isinstance(result, PerceptionResult), "Not a PerceptionResult"
    assert result.frame is not None, "Frame is None"
    assert len(result.fused_elements) > 0, "No fused elements"
    print(f"  Perceived {len(result.fused_elements)} elements in {elapsed:.1f}s")
    
    # Test element search
    found = result.get_element_by_text("File", fuzzy=True)
    if found:
        print(f"  Found 'File' element: role={found.role}, center={found.center}")
    else:
        print(f"  'File' not found (may not be visible on screen)")

test_perception()


# ============================================================
# 6. Action Definitions
# ============================================================
@test("Action Definitions")
def test_actions():
    from actions import ClickAction, TypeAction, KeyAction, ScrollAction, WaitAction
    
    click = ClickAction(x=100, y=200, confidence=0.9)
    assert click.describe() == "Click (left, 1x) at (100, 200)"
    
    type_a = TypeAction(text="hello world", confidence=0.8)
    assert "hello world" in type_a.describe()
    
    key_a = KeyAction(key="enter", confidence=0.7)
    assert "enter" in key_a.describe()
    
    scroll = ScrollAction(x=500, y=500, direction="down", amount=3, confidence=0.6)
    assert "down" in scroll.describe()
    
    wait = WaitAction(duration=1.0, confidence=0.5)
    assert "1.0" in wait.describe()
    
    print(f"  ClickAction: {click.describe()}")
    print(f"  TypeAction: {type_a.describe()}")
    print(f"  KeyAction: {key_a.describe()}")
    print(f"  ScrollAction: {scroll.describe()}")
    print(f"  WaitAction: {wait.describe()}")

test_actions()


# ============================================================
# 7. Action Executor (safe - WaitAction only)
# ============================================================
@test("Action Executor (safe WaitAction)")
def test_executor():
    from actions import ActionExecutor, WaitAction
    
    executor = ActionExecutor()
    wait = WaitAction(duration=0.1, confidence=1.0)
    
    t0 = time.time()
    result = executor.execute(wait)
    elapsed = time.time() - t0
    
    assert result is True, "WaitAction failed"
    assert elapsed >= 0.1, f"Wait too short: {elapsed:.3f}s"
    print(f"  WaitAction executed in {elapsed:.3f}s")
    
    # Test mouse position read (safe, no movement)
    pos = executor.get_mouse_position()
    print(f"  Mouse position: {pos}")

test_executor()


# ============================================================
# 8. LLM Model (Ollama)
# ============================================================
@test("LLM Model (Ollama - Intent Extraction)")
def test_llm_intent():
    from models.llm_model import LLMModel, Intent
    
    llm = LLMModel()
    llm.load()
    
    t0 = time.time()
    intent = llm.extract_intent("Open Notepad and type hello")
    elapsed = time.time() - t0
    
    assert isinstance(intent, Intent), "Not an Intent"
    assert intent.goal != "unknown", f"Intent extraction returned 'unknown': {intent}"
    print(f"  Intent extracted in {elapsed:.1f}s")
    print(f"    goal: {intent.goal}")
    print(f"    app: {intent.app}")
    print(f"    entities: {intent.entities}")

test_llm_intent()


@test("LLM Model (Ollama - Task Planning)")
def test_llm_plan():
    from models.llm_model import LLMModel, Intent, TaskStep
    
    llm = LLMModel()
    llm.load()
    
    intent = Intent(goal="open_notepad_and_type", app="Notepad", entities={"text": "hello"}, raw_prompt="Open Notepad and type hello")
    
    t0 = time.time()
    steps = llm.create_task_plan(intent)
    elapsed = time.time() - t0
    
    assert isinstance(steps, list), "Steps not a list"
    assert len(steps) > 0, "No steps generated"
    print(f"  Plan created in {elapsed:.1f}s: {len(steps)} steps")
    for s in steps:
        assert isinstance(s, TaskStep), "Not a TaskStep"
        print(f"    {s.step_number}. {s.action}: {s.description}")

test_llm_plan()


@test("LLM Model (Ollama - Action Planning)")
def test_llm_action():
    from models.llm_model import LLMModel, TaskStep, PlannedAction
    
    llm = LLMModel()
    llm.load()
    
    step = TaskStep(step_number=1, action="open search", description="Press Win+S to open Windows search", target=None)
    elements = [
        {"role": "button", "text": "Search", "bbox": [0.0, 0.95, 0.05, 1.0], "confidence": 0.9, "sources": ["ocr"]},
        {"role": "text", "text": "File", "bbox": [0.0, 0.0, 0.03, 0.02], "confidence": 0.85, "sources": ["ocr"]},
    ]
    
    t0 = time.time()
    action = llm.plan_action(
        screen_description="Windows desktop with taskbar at bottom",
        available_elements=elements,
        current_step=step,
        history=[]
    )
    elapsed = time.time() - t0
    
    assert isinstance(action, PlannedAction), "Not a PlannedAction"
    assert action.action_type in ["click", "type", "key", "scroll", "wait"], f"Invalid action type: {action.action_type}"
    print(f"  Action planned in {elapsed:.1f}s")
    print(f"    type: {action.action_type}")
    print(f"    target_text: {action.target_text}")
    print(f"    target_role: {action.target_role}")
    print(f"    text: {action.text}")
    print(f"    key: {action.key}")
    print(f"    confidence: {action.confidence}")

test_llm_action()


# ============================================================
# 9. VLM Model (Ollama)
# ============================================================
@test("VLM Model (Ollama - Screen Description)")
def test_vlm():
    from models.vlm_model import VLMModel
    from perception.screen_capture import ScreenCapture
    
    vlm = VLMModel()
    vlm.load()
    
    cap = ScreenCapture()
    frame = cap.grab()
    
    t0 = time.time()
    description = vlm.describe_screen(frame.image)
    elapsed = time.time() - t0
    
    assert isinstance(description, str), "Description not a string"
    assert len(description) > 10, f"Description too short: '{description}'"
    print(f"  Screen described in {elapsed:.1f}s")
    print(f"  Description: {description[:200]}...")

test_vlm()


@test("VLM Model (Ollama - UI Element Detection)")
def test_vlm_elements():
    from models.vlm_model import VLMModel, UIRegion
    from perception.screen_capture import ScreenCapture
    
    vlm = VLMModel()
    vlm.load()
    
    cap = ScreenCapture()
    frame = cap.grab()
    
    t0 = time.time()
    regions = vlm.detect_ui_elements(frame.image)
    elapsed = time.time() - t0
    
    assert isinstance(regions, list), "Regions not a list"
    print(f"  Detected {len(regions)} UI regions in {elapsed:.1f}s")
    for r in regions[:5]:
        assert isinstance(r, UIRegion), "Not a UIRegion"
        print(f"    role={r.role}, desc={r.description[:50]}, conf={r.confidence}")

test_vlm_elements()


# ============================================================
# 10. Verification Logic
# ============================================================
@test("Verification Logic (Visual Diff)")
def test_verification():
    from agent.verification import Verifier
    from perception.perception_engine import PerceptionEngine
    import numpy as np
    
    engine = PerceptionEngine(enable_vlm=False)
    engine.initialize_ocr()
    verifier = Verifier(engine)
    
    # Test visual diff with identical images
    img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img2 = img1.copy()
    
    mse, pct = verifier._calculate_visual_diff(img1, img2)
    assert mse == 0.0, f"MSE should be 0 for identical images, got {mse}"
    assert pct == 0.0, f"Percent should be 0 for identical images, got {pct}"
    print(f"  Identical images: MSE={mse:.2f}, changed={pct:.2f}%")
    
    # Test with different images
    img3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mse2, pct2 = verifier._calculate_visual_diff(img1, img3)
    assert mse2 > 0, "MSE should be > 0 for different images"
    print(f"  Different images: MSE={mse2:.2f}, changed={pct2:.2f}%")

test_verification()


# ============================================================
# 11. Memory (SQLite)
# ============================================================
@test("Memory (SQLite)")
def test_memory():
    from memory.memory import Memory, PatternRecord
    import tempfile, os
    
    # Use temp file for test
    tmp = os.path.join(tempfile.gettempdir(), "atlas_test_memory.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    
    mem = Memory(db_path=tmp)
    mem.enabled = True
    
    # Store a pattern
    pattern = PatternRecord(
        app_name="Notepad",
        element_role="button",
        element_text="File",
        bbox_relative=[0.0, 0.0, 0.03, 0.02],
        action_type="click"
    )
    mem.store(pattern)
    
    # Lookup
    results = mem.lookup("Notepad", role="button")
    assert len(results) >= 1, f"Expected >= 1 result, got {len(results)}"
    assert results[0].element_text == "File"
    print(f"  Stored and retrieved pattern: {results[0].element_text}")
    
    # Confidence boost
    boost = mem.get_confidence_boost("Notepad", "button", "File")
    assert boost > 0, f"Expected boost > 0, got {boost}"
    print(f"  Confidence boost: {boost:.3f}")
    
    # Store same pattern again (should increment success_count)
    mem.store(pattern)
    results2 = mem.lookup("Notepad", role="button")
    assert results2[0].success_count >= 2, f"Expected success_count >= 2, got {results2[0].success_count}"
    print(f"  Success count after 2nd store: {results2[0].success_count}")
    
    mem.close()
    os.remove(tmp)

test_memory()


# ============================================================
# 12. Agent State
# ============================================================
@test("Agent State")
def test_state():
    from agent.state import AgentState, AgentStatus
    from models.llm_model import PlannedAction
    
    state = AgentState()
    assert state.status == AgentStatus.IDLE
    assert state.is_complete is True  # No steps = complete
    assert state.can_retry is True
    
    state.status = AgentStatus.RUNNING
    assert state.status == AgentStatus.RUNNING
    
    # Record action
    action = PlannedAction(action_type="click", target_text="File", confidence=0.9)
    state.record_action(action, True, True)
    assert len(state.action_history) == 1
    
    recent = state.get_recent_actions()
    assert len(recent) == 1
    print(f"  Recent actions: {recent}")
    
    # Reset
    state.reset()
    assert state.status == AgentStatus.IDLE
    assert len(state.action_history) == 0
    print(f"  State reset: OK")

test_state()


# ============================================================
# 13. Agent Loop (import + construction only)
# ============================================================
@test("Agent Loop (import + construct)")
def test_agent_loop_import():
    from agent.agent_loop import AgentLoop
    
    agent = AgentLoop()
    assert agent.llm is not None
    assert agent.perception is not None
    assert agent.executor is not None
    assert agent.verifier is not None
    assert agent.memory is not None
    assert agent.state is not None
    assert agent._initialized is False
    print(f"  AgentLoop constructed successfully")
    print(f"  LLM backend: {agent.llm.config.backend}")
    print(f"  VLM enabled: {agent.perception._enable_vlm}")

test_agent_loop_import()


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("COMPONENT TEST SUMMARY")
print("=" * 60)

passed = 0
failed = 0
for name, result in RESULTS.items():
    status = "PASS" if result == "PASS" else "FAIL"
    icon = "+" if status == "PASS" else "X"
    print(f"  [{icon}] {name}: {result}")
    if status == "PASS":
        passed += 1
    else:
        failed += 1

print(f"\nTotal: {passed} passed, {failed} failed, {passed + failed} total")

if failed > 0:
    print("\nFAILED TESTS NEED FIXING BEFORE END-TO-END TEST")
    sys.exit(1)
else:
    print("\nALL COMPONENTS WORKING - READY FOR END-TO-END TEST")
    sys.exit(0)
