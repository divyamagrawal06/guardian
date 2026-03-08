"""Quick test of UI Automation perception."""
from perception.ui_automation import UIAutomationPerception

p = UIAutomationPerception(max_depth=15, max_elements=200)
elems = p.get_elements()
print(f"Found {len(elems)} elements")
for e in elems[:30]:
    label = e.name[:50] if e.name else e.value[:50] if e.value else "(no name)"
    print(f"  [{e.role:12s}] \"{label}\" @ {e.center}  enabled={e.is_enabled}")
