"""Quick WebSocket test for ATLAS backend."""
import asyncio
import json
import sys
import time
import websockets


async def test():
    command = sys.argv[1] if len(sys.argv) > 1 else "open notepad and type hello"
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Send a simple command
        cmd = {"type": "command", "command": command}
        print(f"Sending: {cmd}")
        await ws.send(json.dumps(cmd))
        # Collect responses for 5 minutes
        start = time.time()
        while time.time() - start < 300:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                print(f"RAW: {msg}")
                data = json.loads(msg)
                typ = data.get("type", "?")
                step = data.get("step", "")
                status = data.get("status", "")
                detail = str(data.get("detail", ""))[:120]
                print(f"  => [{typ}] {step} | {status} | {detail}")
                if step == "task" and status in ("completed", "failed"):
                    break
                if step == "result":
                    break
            except asyncio.TimeoutError:
                print("  (waiting...)")
                continue
            except Exception as e:
                print(f"Error: {e}")
                break
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(test())
#to fix - import WebSockets