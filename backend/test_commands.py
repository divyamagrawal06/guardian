"""Quick test script to send commands to the ATLAS agent via WebSocket."""
import asyncio
import json
import websockets
import sys

async def test_command(command: str, timeout: int = 60):
    """Send a command and print all responses."""
    print(f"\n{'='*60}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        # Wait for connected message
        msg = json.loads(await ws.recv())
        print(f"  [connected] agent_ready={msg.get('agent_ready')}")
        
        # Send command
        await ws.send(json.dumps({"type": "command", "command": command}))
        
        # Collect responses
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                msg = json.loads(raw)
                msg_type = msg.get("type", "")
                
                if msg_type == "progress":
                    step = msg.get("step", "")
                    status = msg.get("status", "")
                    detail = msg.get("detail", "")[:120]
                    print(f"  [{step}] {status}: {detail}")
                    
                elif msg_type == "result":
                    success = msg.get("success")
                    detail = msg.get("detail", "")
                    symbol = "✓" if success else "✗"
                    print(f"\n  {symbol} RESULT: {detail}")
                    return success
                    
                elif msg_type == "error":
                    print(f"\n  ✗ ERROR: {msg.get('message')}")
                    return False
                    
        except asyncio.TimeoutError:
            print(f"\n  ✗ TIMEOUT after {timeout}s")
            return False

async def main():
    commands = [
        "Open Notepad and type hello world",
    ]
    
    if len(sys.argv) > 1:
        commands = [" ".join(sys.argv[1:])]
    
    for cmd in commands:
        result = await test_command(cmd, timeout=90)
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
