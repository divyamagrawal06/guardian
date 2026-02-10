# 🚀 Icarus Redesigned Architecture Proposal

## Executive Summary

**Current Performance**: 15-30 seconds per command  
**Target Performance**: 1-3 seconds per command  
**Expected Improvement**: **10x faster**

---

## 🎯 Redesigned System Architecture

### **1. NEW LAYERED ARCHITECTURE**

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND LAYER                               │
│  Desktop (Tauri) │ Android (Kotlin) │ Web (React)                │
│                  WebSocket Connection                            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   API GATEWAY LAYER (FastAPI)                    │
│  - WebSocket Server (Real-time bidirectional)                   │
│  - REST API (HTTP fallback)                                      │
│  - Request Queue (Redis/Memory)                                  │
│  - Session Management                                            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                AGENT ORCHESTRATION LAYER                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  VisionActionLoop (Single Persistent Instance)           │   │
│  │  - Event Loop (Async)                                    │   │
│  │  - Task Queue                                            │   │
│  │  - State Machine                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              VISION LAYER (Parallel Execution)                   │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │ Screen Capture │  │   VLM Service  │  │   OCR Service    │   │
│  │   (mss.rs)     │  │   (YOLOv8)     │  │  (PaddleOCR)     │   │
│  │   - DPI-aware  │  │   - Singleton  │  │   - Singleton    │   │
│  │   - ROI mode   │  │   - GPU accel  │  │   - Parallel     │   │
│  └────────────────┘  └────────────────┘  └──────────────────┘   │
│                ▼              ▼                   ▼              │
│         ┌──────────────────────────────────────────────┐         │
│         │    Fusion Engine (DPI-Aware Grounder)        │         │
│         │    - Vector embeddings for caching           │         │
│         └──────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PLANNING LAYER (LLM Brain)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Intent Parser (Cached LLM - Gemini/GPT-4)               │   │
│  │  - Plans cached by intent hash                           │   │
│  │  - Few-shot examples in vector DB                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXECUTION LAYER (Action Primitives)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Action Executor (Native Bindings)                        │   │
│  │  - Windows: SendInput API (not PyAutoGUI)                │   │
│  │  - macOS: CGEvent API                                     │   │
│  │  - Linux: XTest/uinput                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Verification Engine (Lightweight)                        │   │
│  │  - Incremental ISR diff (not full reprocessing)          │   │
│  │  - Confidence-based skip                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PERSISTENCE LAYER                              │
│  - PostgreSQL (Session history, analytics)                      │
│  - Redis (ISR cache, plan cache, task queue)                    │
│  - Vector DB (Embedding similarity search)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Architectural Changes

### **1. PERSISTENT MODEL SINGLETON PATTERN**

**Problem**: Models reload on every request  
**Solution**: Single persistent service instances

```python
# NEW: vision/model_service.py
class VisionModelService:
    """Singleton service that runs continuously"""
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self):
        """Load models once at startup"""
        if not self._models_loaded:
            self.yolo = await asyncio.to_thread(self._load_yolo)
            self.ocr = await asyncio.to_thread(self._load_ocr)
            self._models_loaded = True
            logger.info("Models loaded successfully (one-time)")
    
    async def analyze_parallel(self, image: np.ndarray):
        """Run VLM and OCR in parallel"""
        vlm_task = asyncio.create_task(self._run_yolo(image))
        ocr_task = asyncio.create_task(self._run_ocr(image))
        
        vlm_result, ocr_result = await asyncio.gather(vlm_task, ocr_task)
        return vlm_result, ocr_result

# Usage
service = VisionModelService()
await service.initialize()  # Called ONCE at startup
```

**Impact**: **Eliminates 8-15s cold start per request**

---

### **2. CONTINUOUS FEEDBACK LOOP (Not Linear)**

**Problem**: Perception → Plan → Execute runs once  
**Solution**: Closed-loop with re-perception

```python
# NEW: core/vision_action_loop.py
class VisionActionLoop:
    """Event-driven perception-action cycle"""
    
    async def execute_task(self, task: Task):
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            # 1. Perceive current state
            state = await self.perceive()
            
            # 2. Check if goal achieved
            if await self.verifier.check_goal(state, task.goal):
                return {"success": True, "iterations": iteration}
            
            # 3. Decide next action
            action = await self.planner.next_action(state, task)
            
            # 4. Execute
            await self.executor.execute(action)
            
            # 5. Wait for UI update (adaptive)
            await asyncio.sleep(0.3)  # Much shorter
            
            iteration += 1
        
        return {"success": False, "reason": "max_iterations"}
```

**Impact**: **Adapts to failures instead of starting over**

---

### **3. WEBSOCKET REAL-TIME COMMUNICATION**

**Problem**: HTTP polling creates latency  
**Solution**: Bidirectional WebSocket

```python
# backend/main.py (NEW)
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SessionManager.create_session()
    
    try:
        while True:
            # Receive command
            data = await websocket.receive_json()
            command = data['command']
            
            # Stream progress updates
            async for update in agent.execute_stream(command):
                await websocket.send_json({
                    "type": "progress",
                    "step": update['step'],
                    "status": update['status']
                })
            
            # Send final result
            await websocket.send_json({
                "type": "result",
                "data": result
            })
    except WebSocketDisconnect:
        SessionManager.close_session(session)
```

**Frontend (TypeScript)**:
```typescript
// Desktop app
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'progress') {
        updateProgress(data.step, data.status);
    }
};

ws.send(JSON.stringify({ command: "open calculator" }));
```

**Impact**: **Real-time feedback, 0 polling overhead**

---

### **4. SMART CACHING LAYERS**

```python
# NEW: core/cache_manager.py
import hashlib
from functools import lru_cache

class ISRCache:
    """Cache ISR by screenshot hash"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 5  # 5 seconds for ISR
    
    def get_image_hash(self, image: np.ndarray) -> str:
        """Fast perceptual hash"""
        resized = cv2.resize(image, (64, 64))
        return hashlib.sha256(resized.tobytes()).hexdigest()
    
    async def get_or_compute(self, image: np.ndarray, compute_fn):
        img_hash = self.get_image_hash(image)
        
        # Check cache
        cached = await self.redis.get(f"isr:{img_hash}")
        if cached:
            return json.loads(cached)
        
        # Compute
        isr = await compute_fn(image)
        
        # Store
        await self.redis.setex(
            f"isr:{img_hash}", 
            self.ttl, 
            json.dumps(isr)
        )
        return isr

# Usage
isr = await cache.get_or_compute(screenshot, vision.analyze)
```

**Impact**: **Avoids re-processing identical screens**

---

### **5. NATIVE SCREEN CAPTURE (DPI-Aware)**

**Problem**: PyAutoGUI slow + DPI issues  
**Solution**: Use native APIs via Rust/C bindings

```python
# NEW: vision/capture_native.py (calls Rust)
import ctypes
from ctypes import wintypes

class WindowsCaptureModule:
    """DPI-aware Windows screen capture"""
    
    def __init__(self):
        # Set DPI awareness
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        
        self.dpi_scale = self._get_dpi_scale()
    
    def _get_dpi_scale(self) -> float:
        """Get monitor DPI scale factor"""
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0  # 96 DPI is 100% scale
    
    def capture_screen(self) -> np.ndarray:
        """Fast screen capture using Windows Graphics API"""
        # Use mss (faster than pyautogui)
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            return np.array(img)
    
    def adjust_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Adjust coordinates for DPI scaling"""
        return int(x / self.dpi_scale), int(y / self.dpi_scale)
```

**Impact**: **Fixes DPI click failures + 2-3x faster capture**

---

### **6. INCREMENTAL VERIFICATION (Not Full Re-processing)**

**Problem**: Re-runs full perception after every action  
**Solution**: Diff-based verification

```python
# NEW: core/incremental_verifier.py
class IncrementalVerifier:
    """Lightweight verification without full ISR rebuild"""
    
    async def verify_action(self, before_state: Dict, action: Action, expected_change: str):
        """Verify using targeted checks, not full perception"""
        
        if action.type == "click":
            # Just check if clicked element changed state
            region = self._get_element_region(action.target)
            after_region = await self.capture.capture_region(region)
            
            # Use lightweight OCR on region only
            text_changed = self._quick_ocr_check(after_region, expected_change)
            return text_changed
        
        elif action.type == "type":
            # Check if input field has new text
            # (Much faster than full screen analysis)
            pass
    
    def _quick_ocr_check(self, region_img: np.ndarray, expected: str) -> bool:
        """Fast OCR on small region"""
        # Use Tesseract PSM 7 (single line) for speed
        text = pytesseract.image_to_string(region_img, config='--psm 7')
        return expected.lower() in text.lower()
```

**Impact**: **Verification 5-10x faster**

---

## 📊 Performance Comparison Table

| Operation | Current Time | Optimized Time | Improvement |
|-----------|--------------|----------------|-------------|
| **Model Loading** | 8-15s per request | 0s (loaded once) | ∞ |
| **Screen Capture** | 800ms | 50ms (mss) | **16x** |
| **VLM + OCR (Sequential)** | 5-7s | 2-3s (parallel) | **2.5x** |
| **Verification** | 3-5s (full ISR) | 0.3-0.5s (diff) | **10x** |
| **Backend Overhead** | 300-500ms | 5-10ms (WebSocket) | **50x** |
| **Total Per Command** | **20-30s** | **2-4s** | **10x** |

---

## 🛠️ Implementation Roadmap

### **Phase 1: Model Singleton & Async (Week 1-2)**
- [ ] Create `VisionModelService` singleton
- [ ] Refactor to async/await everywhere
- [ ] Parallel VLM+OCR execution
- [ ] Load models at startup, not per request

**Expected Gain**: 5-7x speedup

---

### **Phase 2: WebSocket & Caching (Week 3)**
- [ ] Implement WebSocket endpoint
- [ ] Add Redis for ISR caching
- [ ] Frontend WebSocket client
- [ ] Session management

**Expected Gain**: Additional 2x speedup

---

### **Phase 3: Native Capture & DPI (Week 4)**
- [ ] Rust/mss-based screen capture
- [ ] DPI-aware coordinate mapping
- [ ] ROI (region of interest) capture mode

**Expected Gain**: 3x capture speedup + fixes click accuracy

---

### **Phase 4: Incremental Verification (Week 5)**
- [ ] Diff-based state verification
- [ ] Targeted region OCR
- [ ] Confidence-based skip logic

**Expected Gain**: 5-10x verification speedup

---

### **Phase 5: Feedback Loop (Week 6)**
- [ ] Refactor to continuous perception-action cycle
- [ ] State machine for task execution
- [ ] Adaptive retry with re-planning

**Expected Gain**: Handles failures gracefully

---

## 🔧 Quick Wins (Can Implement Today)

### **1. Use `mss` instead of `pyautogui.screenshot()`**
```python
# REPLACE in capture.py
import mss

def capture_screen():
    with mss.mss() as sct:
        return np.array(sct.grab(sct.monitors[1]))
```
**Gain**: 5-10x faster capture

---

### **2. Run VLM and OCR in parallel**
```python
# REPLACE in perception_pipeline.py
import asyncio

async def process_screenshot_async(self, image_path):
    image = self._load_image(image_path)
    
    # Parallel execution
    vlm_task = asyncio.to_thread(self.vlm_detector.detect, image)
    ocr_task = asyncio.to_thread(self.ocr_extractor.extract, image)
    
    ui_elements, ocr_tokens = await asyncio.gather(vlm_task, ocr_task)
    # ... rest of fusion
```
**Gain**: 2x speedup on perception

---

### **3. Add simple ISR cache**
```python
# ADD to perception_pipeline.py
import hashlib
from functools import lru_cache

@lru_cache(maxsize=10)
def _compute_isr_cached(image_hash: str, image_bytes: bytes):
    # ... ISR computation
    pass
```
**Gain**: Instant response for repeated screens

---

## 📚 Technology Stack Changes

| Layer | Current | Proposed | Reason |
|-------|---------|----------|--------|
| **Screen Capture** | PyAutoGUI | `mss` (Python) or Rust bindings | 10x faster |
| **Backend Framework** | FastAPI (sync) | FastAPI (async) + WebSockets | Real-time |
| **Task Queue** | None | Redis/Celery | Async processing |
| **Cache** | None | Redis | ISR caching |
| **DB** | JSON files | PostgreSQL | Persistence |
| **Action Execution** | PyAutoGUI | Native APIs (SendInput/CGEvent) | Reliability |
| **Model Hosting** | In-process | Separate microservice (optional) | Scalability |

---

## 🎯 Success Metrics

- ✅ Cold start time: **< 2 seconds** (vs 15s)
- ✅ Command execution: **< 3 seconds** (vs 30s)
- ✅ Click accuracy: **> 95%** on all DPI settings
- ✅ Memory usage: **< 2GB** (persistent models)
- ✅ Concurrent sessions: **> 5**

---

## 🚨 Critical Fixes Required

1. **Fix DPI scaling** (causes 50%+ click failures on Windows)
2. **Eliminate model reload** (biggest bottleneck)
3. **Implement feedback loop** (current linear model fails on errors)
4. **Replace PyAutoGUI** (unreliable for production)

---

## 📖 Next Steps

1. Review this proposal
2. Prioritize phases
3. I can help implement any of the quick wins immediately
4. Set up development environment for Phase 1

Let me know which areas you want to dive deeper into!
