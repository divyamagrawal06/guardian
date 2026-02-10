"""
ATLAS ML Pipeline – Windows UI Automation Perception
=====================================================

Uses the Windows Accessibility (UIA) tree to enumerate *real* UI elements
with precise bounding-box coordinates, roles, text, and state.

This replaces PaddleOCR + VLM for element detection — it reads what the
OS already knows about the UI, which is faster and pixel-accurate.

PIPELINE STEPS 4-6 replacement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UIElement:
    """A single UI element from the accessibility tree."""
    role: str              # e.g. "Button", "Edit", "MenuItem", "Link"
    name: str              # visible / accessible name
    automation_id: str     # developer-assigned id (often empty)
    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom) in absolute px
    is_enabled: bool
    is_offscreen: bool
    value: str = ""        # text value for edit controls
    children_count: int = 0

    @property
    def center(self) -> Tuple[int, int]:
        """Absolute screen-pixel center of the element."""
        l, t, r, b = self.bbox
        return ((l + r) // 2, (t + b) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        """Compact dict for passing into LLM prompts."""
        return {
            "role": self.role,
            "name": self.name,
            "center": list(self.center),
            "bbox": list(self.bbox),
            "enabled": self.is_enabled,
            "value": self.value[:60] if self.value else "",
        }

    def __repr__(self) -> str:
        return f"<UIElement {self.role} '{self.name}' @ {self.center}>"


# ---------------------------------------------------------------------------
# Main perception class
# ---------------------------------------------------------------------------

_INTERESTING_ROLES = {
    "ButtonControl", "EditControl", "TextControl", "HyperlinkControl",
    "MenuItemControl", "ListItemControl", "TabItemControl", "CheckBoxControl",
    "RadioButtonControl", "ComboBoxControl", "TreeItemControl",
    "DocumentControl", "ImageControl", "ToolBarControl", "MenuBarControl",
    "GroupControl", "DataItemControl", "SliderControl", "SpinnerControl",
}

# Roles we always skip (reduces noise)
_SKIP_ROLES = {
    "ScrollBarControl", "ThumbControl", "SeparatorControl",
    "TitleBarControl", "HeaderControl", "HeaderItemControl",
}


class UIAutomationPerception:
    """
    Reads the Windows UI Automation tree for the foreground window and
    returns a flat list of interactive/visible elements.

    Usage:
        perc = UIAutomationPerception()
        elements = perc.get_elements()          # list[UIElement]
        description = perc.describe_screen()    # short text summary
    """

    def __init__(self, max_depth: int = 8, max_elements: int = 120):
        self.max_depth = max_depth
        self.max_elements = max_elements

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_elements(self, window_name: Optional[str] = None) -> List[UIElement]:
        """
        Return visible, interactive elements from the foreground window
        (or a window matching *window_name*).
        """
        import uiautomation as auto

        t0 = time.perf_counter()

        if window_name:
            root = auto.WindowControl(searchDepth=1, Name=window_name)
        else:
            root = auto.GetForegroundControl()
            # Walk up to the top-level window
            while root and root.ControlTypeName != "WindowControl":
                parent = root.GetParentControl()
                if parent is None or parent == root:
                    break
                root = parent

        if root is None:
            logger.warning("No foreground window found")
            return []

        elements: List[UIElement] = []
        self._walk(root, elements, depth=0)

        elapsed = time.perf_counter() - t0
        logger.debug(f"UI Automation found {len(elements)} elements in {elapsed:.2f}s")
        return elements

    def get_window_title(self) -> str:
        """Return the title of the current foreground window."""
        import uiautomation as auto
        win = auto.GetForegroundControl()
        return win.Name if win else ""

    def describe_screen(self, elements: Optional[List[UIElement]] = None) -> str:
        """Build a short text summary of the current screen."""
        if elements is None:
            elements = self.get_elements()

        import uiautomation as auto
        win_name = auto.GetForegroundControl().Name

        parts = [f"Active window: {win_name}"]
        parts.append(f"Found {len(elements)} interactive elements.")

        # List the first ~20 interesting ones
        shown = 0
        for e in elements:
            if shown >= 20:
                parts.append(f"  ... and {len(elements) - shown} more")
                break
            if e.name or e.value:
                label = e.name or e.value[:40]
                parts.append(f"  [{e.role}] \"{label}\" @ {e.center}")
                shown += 1

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal tree walk
    # ------------------------------------------------------------------

    def _walk(self, node, out: List[UIElement], depth: int) -> None:
        if depth > self.max_depth or len(out) >= self.max_elements:
            return

        try:
            ctrl_type = node.ControlTypeName or ""
        except Exception:
            return

        if ctrl_type in _SKIP_ROLES:
            return

        # Collect this node if it's interesting
        if ctrl_type in _INTERESTING_ROLES or depth <= 1:
            try:
                rect = node.BoundingRectangle
                # Skip elements with zero-size or completely offscreen
                if rect.width() > 0 and rect.height() > 0:
                    bbox = (rect.left, rect.top, rect.right, rect.bottom)
                    name = (node.Name or "").strip()
                    value = ""
                    try:
                        vp = node.GetValuePattern()
                        if vp:
                            value = vp.Value or ""
                    except Exception:
                        pass

                    elem = UIElement(
                        role=ctrl_type.replace("Control", ""),
                        name=name,
                        automation_id=node.AutomationId or "",
                        bbox=bbox,
                        is_enabled=node.IsEnabled,
                        is_offscreen=node.IsOffscreen,
                        value=value,
                    )
                    # Only add if it has some identity
                    if name or value or ctrl_type in ("EditControl", "ButtonControl"):
                        out.append(elem)
            except Exception:
                pass

        # Recurse into children
        try:
            children = node.GetChildren()
            if children:
                for child in children:
                    if len(out) >= self.max_elements:
                        break
                    self._walk(child, out, depth + 1)
        except Exception:
            pass
