# ATLAS ML Pipeline
## Vision-Driven Desktop Agent

A closed-loop autonomous agent that operates any desktop application using **only visual perception** (screen pixels).

### Core Philosophy
> The screen is the only source of truth. No APIs, no hooks, no extensions.

### Core Loop
```
PERCEIVE → UNDERSTAND → PLAN → ACT → VERIFY → (repeat)
```

---

## 📁 Project Structure

```
ml/
├── config/             # Configuration management
│   ├── __init__.py
│   └── settings.py     # Pydantic config models
│
├── models/             # ML model wrappers
│   ├── __init__.py
│   ├── ocr_model.py    # PaddleOCR wrapper
│   ├── vlm_model.py    # LLaVA vision-language model
│   ├── llm_model.py    # Mistral/Phi local LLM
│   └── downloads/      # Downloaded model files
│
├── perception/         # Visual perception pipeline
│   ├── __init__.py
│   ├── screen_capture.py    # mss screenshot capture
│   ├── bbox_fusion.py       # Bounding box fusion
│   └── perception_engine.py # Orchestrator
│
├── actions/            # OS-level interaction
│   ├── __init__.py
│   ├── actions.py      # Action dataclasses
│   └── executor.py     # PyAutoGUI executor
│
├── agent/              # Agent control loop
│   ├── __init__.py
│   ├── state.py        # Agent state tracking
│   ├── verification.py # Visual verification
│   └── agent_loop.py   # Main loop
│
├── memory/             # Pattern memory (no training)
│   ├── __init__.py
│   └── memory.py       # SQLite pattern storage
│
├── data/               # Runtime data
│   └── .gitkeep
│
├── __init__.py         # Package exports
├── main.py             # CLI entry point
├── requirements.txt    # Dependencies
└── errors.txt          # Known issues tracker
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models
- **LLaVA**: Auto-downloads from HuggingFace on first run
- **Mistral/Phi**: Download GGUF from [TheBloke](https://huggingface.co/TheBloke)

### 3. Run
```bash
# Single command
python main.py "Open Notepad and type hello"

# Interactive mode
python main.py --interactive
```

---

## 🔧 Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `llm_model.extract_intent()` | Parse user prompt to structured intent |
| 2 | `llm_model.create_task_plan()` | Break intent into abstract steps |
| 3 | `screen_capture.grab()` | Capture screenshot |
| 4 | `ocr_model.detect()` | Extract text + bounding boxes |
| 5 | `vlm_model.detect_ui_elements()` | Identify UI regions |
| 6 | `bbox_fusion.fuse()` | Merge all detections |
| 7 | `llm_model.plan_action()` | Decide one atomic action |
| 8 | `agent_loop._resolve_action()` | Convert to pixel coordinates |
| 9 | `executor.execute()` | Perform OS-level input |
| 10 | `verifier.verify()` | Confirm action effect |

---

## ⚠️ Known Issues

See `errors.txt` for tracked issues:
- **CRITICAL**: DPI scaling, modal handling, window state, multi-monitor
- **NON-CRITICAL**: Error taxonomy, typing speed, VLM latency

---

## 🎯 Design Principles

1. **Never trust one perception pass**
2. **Never assume a click worked**
3. **Never hardcode coordinates**
4. **Always verify visually**
5. **Screen state is the only truth**
