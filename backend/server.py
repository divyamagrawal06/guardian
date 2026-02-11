import sys
import os
import asyncio
import json
from typing import Dict, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add ml/ to Python path so we can import the agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_2"))

# Load .env from workspace root
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from agent import AgentLoop

# ── Globals ──────────────────────────────────────────────────────────────────

agent: AgentLoop = None
connected_clients: Set[WebSocket] = set()


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup, cleanup on shutdown."""
    global agent
    logger.info("Starting ATLAS backend...")
    
    agent = AgentLoop()
    # Pre-initialize models in background thread (heavy, don't block server start)
    logger.info("Pre-loading ML models (this may take a moment)...")
    try:
        await asyncio.to_thread(agent.initialize)
        logger.info("Agent initialized and ready")
    except Exception as e:
        logger.warning(f"Agent pre-init failed (will retry on first command): {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down ATLAS backend...")
    if agent and agent.memory:
        agent.memory.close()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ATLAS Backend",
    description="WebSocket server for the ATLAS desktop automation agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "agent_ready": agent is not None and agent._initialized,
        "connected_clients": len(connected_clients),
    }


@app.get("/status")
async def status():
    """Get current agent status."""
    if agent is None:
        return {"status": "not_initialized"}
    return {
        "status": agent.state.status.value,
        "current_step": agent.state.current_step_index,
        "total_steps": len(agent.state.task_steps),
    }


@app.get("/api/companion")
async def companion_info():
    """
    REST endpoint for the Flutter companion app.
    Returns server identity + connection info so the app can verify it's
    talking to the right machine before opening a WebSocket.
    """
    import socket
    hostname = socket.gethostname()
    return {
        "server": "atlas",
        "version": "0.1.0",
        "hostname": hostname,
        "agent_ready": agent is not None and agent._initialized,
        "ws_path": "/ws",
    }


# ── WebSocket Endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for bidirectional communication.
    
    Client sends:
        {"type": "command", "command": "Open Notepad and type hello"}
        {"type": "stop"}
    
    Server sends:
        {"type": "progress", "step": "intent", "status": "started", "detail": "..."}
        {"type": "progress", "step": "action", "status": "executing", "detail": "..."}
        {"type": "result", "success": true/false, "detail": "..."}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected ({len(connected_clients)} total)")
    
    try:
        # Send ready confirmation
        await websocket.send_json({
            "type": "connected",
            "agent_ready": agent is not None and agent._initialized,
        })
        
        while True:
            # Wait for client message
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue
            
            msg_type = data.get("type", "")
            
            if msg_type == "command":
                command = data.get("command", "").strip()
                if not command:
                    await websocket.send_json({"type": "error", "message": "Empty command"})
                    continue
                
                await _handle_command(websocket, command)
                
            elif msg_type == "stop":
                if agent:
                    agent.stop()
                await websocket.send_json({"type": "stopped"})
                
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)


async def _handle_command(websocket: WebSocket, command: str):
    """Run an agent command and stream progress to the WebSocket."""
    
    loop = asyncio.get_event_loop()
    progress_queue: asyncio.Queue = asyncio.Queue()
    
    def on_progress(step: str, status: str, detail: str = ""):
        """Callback from agent (runs in worker thread) → async queue."""
        loop.call_soon_threadsafe(
            progress_queue.put_nowait,
            {"type": "progress", "step": step, "status": status, "detail": detail}
        )
    
    logger.info(f"Executing command: {command}")
    
    # Run the synchronous agent in a background thread
    task = asyncio.create_task(
        asyncio.to_thread(agent.run, command, on_progress)
    )
    
    # Stream progress updates to client while agent runs
    try:
        while not task.done():
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                continue
        
        # Drain remaining progress messages
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            await websocket.send_json(msg)
        
        # Get final result
        success = task.result()
        await websocket.send_json({
            "type": "result",
            "success": success,
            "detail": "Task completed" if success else "Task failed",
        })
        
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })


# ── Broadcast Helper ─────────────────────────────────────────────────────────

async def broadcast(message: dict):
    """Send a message to all connected WebSocket clients."""
    disconnected = set()
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.add(ws)
    connected_clients -= disconnected


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
