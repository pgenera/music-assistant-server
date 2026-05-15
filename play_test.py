"""Quick test: authenticate, find the SoundBridge player, and play a radio stream via MA."""

import asyncio
import json
import sys
import websockets

MA_URL = "ws://localhost:8095/ws"
USERNAME = "test"
PASSWORD = "testtest"
PLAYER_ID = "soundbridge_roku_fivesevenfive_org"

# SomaFM Groove Salad - publicly accessible Icecast stream
STREAM_URL = "http://ice1.somafm.com/groovesalad-128-mp3"


async def ws_cmd(ws, mid, cmd, **args):
    msg = {"message_id": mid, "command": cmd}
    if args:
        msg["args"] = args
    await ws.send(json.dumps(msg))
    # Drain events until we see our message_id response
    deadline = asyncio.get_event_loop().time() + 15
    while asyncio.get_event_loop().time() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        resp = json.loads(raw)
        if str(resp.get("message_id")) == str(mid):
            if resp.get("error_code"):
                raise RuntimeError(f"{cmd}: {resp.get('details')}")
            return resp.get("result")
        # Event or other command response — discard and keep waiting
    raise TimeoutError(f"No response for {cmd}")


async def main():
    # Get token
    import urllib.request
    req = urllib.request.Request(
        "http://localhost:8095/auth/login",
        data=json.dumps({
            "provider_id": "builtin",
            "credentials": {"username": USERNAME, "password": PASSWORD},
            "device_name": "play_test",
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        token = json.loads(r.read())["token"]
    print(f"Authenticated as {USERNAME}")

    async with websockets.connect(MA_URL, max_size=10 * 1024 * 1024) as ws:
        await ws.recv()
        await ws_cmd(ws, "auth", "auth", token=token)
        print("WebSocket authenticated")

        # Check player state
        players = await ws_cmd(ws, "pl", "players/all")
        sb = next((p for p in players if p.get("player_id") == PLAYER_ID), None)
        if not sb:
            ids = [p.get("player_id") for p in players]
            print("SoundBridge not found. Available:", ids, file=sys.stderr)
            sys.exit(1)

        print(f"Player found: {sb['name']} | available={sb.get('available')} powered={sb.get('powered')} state={sb.get('playback_state')}")

        # Power on if needed
        if not sb.get("powered"):
            print("Powering on...")
            await ws_cmd(ws, "pw", "players/cmd/power", player_id=PLAYER_ID, powered=True)
            await asyncio.sleep(3)

        # Play the stream URL directly on the player queue
        print(f"Playing: {STREAM_URL}")
        result = await ws_cmd(ws, "play", "player_queues/play_media",
            queue_id=PLAYER_ID,
            media=STREAM_URL,
            option="play",
        )
        print("Play result:", result)

        # Poll state a few times
        for i in range(6):
            await asyncio.sleep(2)
            players = await ws_cmd(ws, f"st{i}", "players/all")
            sb = next((p for p in players if p.get("player_id") == PLAYER_ID), {})
            print(f"  [{i*2}s] state={sb.get('playback_state')} vol={sb.get('volume_level')} powered={sb.get('powered')}")

        print("Done — leaving playback running.")


asyncio.run(main())
