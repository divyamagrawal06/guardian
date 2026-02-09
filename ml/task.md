# ATLAS ML Pipeline - Implementation Tasks

**Last Updated:** 2026-02-09 03:54 IST

---

## Phase 1: Model Integration ✅ COMPLETE (VERIFIED)

| Item | Status | Verification |
|------|--------|--------------|
| Project structure | ✅ | `ml/` with 7 subdirectories |
| `config/settings.py` | ✅ | Pydantic models for all config |
| `models/ocr_model.py` | ✅ | EasyOCR wrapper (updated) |
| `models/vlm_model.py` | ✅ | 281 lines, LLaVA wrapper |
| `models/llm_model.py` | ✅ | 337 lines, llama-cpp wrapper |
| `perception/` | ✅ | screen_capture, bbox_fusion, engine |
| `actions/` | ✅ | actions.py, executor.py |
| `agent/` | ✅ | agent_loop, state, verification |
| `memory/` | ✅ | memory.py (SQLite) |
| Dependencies installed | ✅ | All core packages working |

---

## Phase 2: Model Testing & Validation ✅ COMPLETE (VERIFIED)

| Item | Status | Verification |
|------|--------|--------------|
| **OCR (EasyOCR)** | ✅ | 196-226 text regions detected |
| **Screen capture (mss)** | ✅ | 1920x1080 captured |
| **PyAutoGUI** | ✅ | Mouse position + screen size working |
| **llama-cpp-python** | ✅ | Import + model load + inference working |
| **GGUF model downloaded** | ✅ | TinyLlama 1.1B (668MB) |
| VLM (LLaVA) | ⚠️ | Deferred - CUDA install cancelled |

---

## Phase 3: Perception Pipeline ✅ COMPLETE (VERIFIED)

| Item | Status | Verification |
|------|--------|--------------|
| **Update OCR to EasyOCR** | ✅ | `ocr_model.py` rewritten |
| **Screen capture** | ✅ | 1920x1080, DPI scale: 1.0 |
| **Coordinate normalization** | ✅ | to_absolute/to_normalized working |
| **DPI detection** | ✅ | Windows API detection implemented |
| **BBox fusion** | ✅ | 198-207 elements, 0.05s |
| **Perception engine** | ✅ | quick_perceive() works without VLM |
| **Element search** | ✅ | get_element_by_text() found "File" |
| **Full pipeline benchmark** | ✅ | ~19s (OCR is ~30s on CPU) |

### Phase 3 Notes:
- VLM is **optional** - perception engine works with OCR only
- OCR running on CPU (GPU enabled in config, but PyTorch is CPU-only)
- Full pipeline: capture (0.1s) → OCR (~30s) → fusion (0.05s)

---

## Phase 4: Action Execution ✅ COMPLETE (VERIFIED)

| Item | Status | Verification |
|------|--------|--------------|
| **PyAutoGUI executor** | ✅ | Mouse movement precise (0px error) |
| **Direct actions** | ✅ | Click, Type, Key, Scroll, Wait implemented |
| **Coordinate conversion** | ✅ | Normalized ↔ Absolute round-trip verified |
| **Screen resolution** | ✅ | 1920x1080 detected correctly |
| **Safety mechanisms** | ✅ | FAILSAFE=True, PAUSE=0.1s enabled |
| **Action logging** | ✅ | Last action tracked successfully |

---

## Phase 5: Visual Verification 🔄 NEXT UP
- [ ] Before/after screenshot comparison
- [ ] Change detection logic
- [ ] Retry mechanism on failure

---

## Phase 6: Agent Loop Integration
- [ ] Wire: PERCEIVE → PLAN → ACT → VERIFY
- [ ] Step advancement logic
- [ ] Task completion detection
- [ ] Error handling and recovery

---

## Phase 7: End-to-End Testing
- [ ] Simple task: "Open Notepad"
- [ ] Multi-step task: "Open Notepad and type hello"
- [ ] Error recovery test

---

## Phase 8: Critical Issues (from errors.txt)
- [x] CRITICAL-001: DPI/Scaling handling ✅ (detect_dpi_scale implemented)
- [ ] CRITICAL-002: Modal/Dialog detection  
- [ ] CRITICAL-003: App window state tracking
- [ ] CRITICAL-004: Multi-monitor support

---

## Dependencies Status

| Package | Version | Status | Notes |
|---------|---------|--------|-------|
| torch | 2.8.0 | ⚠️ | CPU version (CUDA install deferred) |
| easyocr | 1.7.2 | ✅ | **Primary OCR** |
| paddleocr | 2.8.1 | ⚠️ | Not used (oneDNN issues) |
| llama-cpp-python | 0.3.2 | ✅ | CPU version |
| mss | - | ✅ | Screen capture |
| pyautogui | - | ✅ | Mouse/keyboard |
| transformers | - | ✅ | For VLM (deferred) |
| numpy | 2.2.6 | ✅ | |
| opencv-python | 4.10.0 | ✅ | |

---

## Model Files

| Model | Location | Size | Status |
|-------|----------|------|--------|
| TinyLlama-1.1B | `models/downloads/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | 668 MB | ✅ |
| EasyOCR models | `~/.EasyOCR/` | ~100 MB | ✅ Auto-downloaded |

---

## Test Commands
```bash
cd ml
python test_phase3.py   # Perception pipeline (OCR + Fusion)
python test_phase4.py   # Action execution (Safe test)
python test_vlm.py      # VLM test (Requires CUDA)
```

---

## Current Focus
> **Next: Phase 5 — Visual Verification (Detecting changes after actions)**
