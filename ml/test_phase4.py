"""
Phase 4 Test - Action Execution
================================
Tests: PyAutoGUI executor, click accuracy, keyboard input, 
coordinate conversion, and safety mechanisms.
"""
import sys
import os
import time

# Add ml directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("PHASE 4: ACTION EXECUTION TEST")
print("=" * 60)

# Test 0: Import checks
print("\n[0] Importing modules...")
try:
    from actions.actions import Action, ClickAction, TypeAction, KeyAction, ScrollAction, WaitAction
    from actions.executor import ActionExecutor
    from perception.screen_capture import ScreenCapture
    import pyautogui
    print("    Imports successful")
except ImportError as e:
    print(f"    FAILED: {e}")
    sys.exit(1)

# Test 1: PyAutoGUI basic functionality
print("\n[1] Testing PyAutoGUI basics...")
screen_width, screen_height = pyautogui.size()
print(f"    Screen size: {screen_width}x{screen_height}")

mouse_x, mouse_y = pyautogui.position()
print(f"    Current mouse position: ({mouse_x}, {mouse_y})")

# Check failsafe is enabled
print(f"    Failsafe enabled: {pyautogui.FAILSAFE}")
print(f"    Pause between actions: {pyautogui.PAUSE}s")

# Test 2: ActionExecutor initialization
print("\n[2] Testing ActionExecutor initialization...")
executor = ActionExecutor()
print(f"    Move duration: {executor.move_duration}s")
print(f"    Type interval: {executor.type_interval}s")
print("    ActionExecutor created successfully")

# Test 3: Coordinate conversion (normalized -> absolute)
print("\n[3] Testing coordinate conversion (normalized -> absolute)...")
capture = ScreenCapture()
frame = capture.grab()
print(f"    Screen captured: {frame.width}x{frame.height}")

# Test center point conversion
test_coords = [
    (0.0, 0.0, "top-left"),
    (0.5, 0.5, "center"),
    (1.0, 1.0, "bottom-right"),
    (0.25, 0.75, "quarter")
]

print("    Coordinate conversions:")
for nx, ny, label in test_coords:
    abs_x, abs_y = frame.to_absolute(nx, ny)
    back_nx, back_ny = frame.to_normalized(abs_x, abs_y)
    print(f"      {label}: ({nx}, {ny}) -> ({abs_x}, {abs_y}) -> ({back_nx:.4f}, {back_ny:.4f})")
    # Verify round-trip accuracy
    assert abs(nx - back_nx) < 0.01, f"X coordinate mismatch: {nx} vs {back_nx}"
    assert abs(ny - back_ny) < 0.01, f"Y coordinate mismatch: {ny} vs {back_ny}"
print("    All coordinate conversions verified [OK]")

# Test 4: Action creation
print("\n[4] Testing Action creation...")
actions = [
    ClickAction(x=100, y=200, button="left", clicks=1),
    ClickAction(x=500, y=300, button="right", clicks=1),
    ClickAction(x=200, y=400, button="left", clicks=2),  # Double click
    TypeAction(text="Hello World"),
    KeyAction(key="enter"),
    KeyAction(key="ctrl+a"),
    ScrollAction(x=500, y=500, direction="down", amount=3),
    WaitAction(duration=0.5),
]

print("    Created actions:")
for i, action in enumerate(actions):
    print(f"      {i+1}. {action.describe()}")

# Test 5: Mouse position tracking (no actual clicks)
print("\n[5] Testing mouse movement (safe - just tracking)...")
original_pos = pyautogui.position()
print(f"    Original position: {original_pos}")

# Move to center of screen (non-destructive)
center_x, center_y = screen_width // 2, screen_height // 2
executor.move_to(center_x, center_y)
new_pos = pyautogui.position()
print(f"    Moved to center: {new_pos}")

# Verify position accuracy
pos_diff = abs(new_pos[0] - center_x) + abs(new_pos[1] - center_y)
print(f"    Position accuracy: {pos_diff}px difference")
assert pos_diff < 5, f"Mouse position inaccurate: expected ({center_x}, {center_y}), got {new_pos}"

# Move back to original
executor.move_to(original_pos[0], original_pos[1])
final_pos = pyautogui.position()
print(f"    Returned to: {final_pos}")

# Test 6: Wait action test
print("\n[6] Testing WaitAction...")
wait_action = WaitAction(duration=0.5)
start = time.time()
result = executor.execute(wait_action)
elapsed = time.time() - start
print(f"    Waited {elapsed:.2f}s (target: 0.5s)")
assert result == True, "WaitAction failed"
assert 0.4 < elapsed < 0.7, f"Wait duration incorrect: {elapsed}s"
print("    WaitAction works correctly [OK]")

# Test 7: Action logging verification
print("\n[7] Testing action logging...")
print(f"    Last action: {executor.last_action}")
print(f"    Last action time: {executor.last_action_time}")
assert executor.last_action is not None, "Action not logged"
print("    Action logging works [OK]")

# Summary
print("\n" + "=" * 60)
print("PHASE 4 TEST RESULTS")
print("=" * 60)
print(f"  PyAutoGUI:            OK (screen: {screen_width}x{screen_height})")
print(f"  ActionExecutor:       OK (initialized)")
print(f"  Coord conversion:     OK (round-trip verified)")
print(f"  Action creation:      OK ({len(actions)} actions)")
print(f"  Mouse tracking:       OK ({pos_diff}px accuracy)")
print(f"  Wait action:          OK ({elapsed:.2f}s)")
print(f"  Action logging:       OK")
print("=" * 60)
print("PHASE 4 SAFE TESTS PASSED!")
print("=" * 60)
