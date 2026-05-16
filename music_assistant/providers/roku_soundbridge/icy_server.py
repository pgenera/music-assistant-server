"""ICY/Shoutcast metadata proxy for Roku SoundBridge."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import aiohttp

_LOGGER = logging.getLogger(__name__)

ICY_PROXY_PORT = 8098

# Headers from the upstream (MA) response that we suppress in the ICY reply —
# either because the ICY protocol uses its own equivalents or because they would
# confuse the SoundBridge firmware.
_SUPPRESS_UPSTREAM_HEADERS = frozenset(
    {
        "transfer-encoding",
        "connection",
        "server",
        "date",
        "icy-name",         # we set our own with the real track title
        "icy-description",  # we set our own
        "icy-version",
        "icy-logo",
        "contentfeatures.dlna.org",
        "transfermode.dlna.org",
    }
)


class IcyServer:
    """
    Asyncio TCP proxy that translates MA's HTTP stream response into the
    ICY/Shoutcast format the SoundBridge requires.

    The SoundBridge always sends ``Icy-Metadata: 1`` and expects the response
    to start with ``ICY 200 OK`` rather than a standard HTTP status line.

    MA's flow stream already handles track transitions; we just translate the
    protocol boundary and override ``icy-name`` / ``icy-description`` with the
    real track title and artist (MA's flow stream sends a generic station name).

    Only one client is served at a time — when the SoundBridge reconnects, the
    previous upstream connection to MA is cancelled so MA does not keep multiple
    competing flow streams alive simultaneously.
    """

    def __init__(self, port: int = ICY_PROXY_PORT) -> None:
        self._port = port
        self._server: asyncio.Server | None = None
        self._upstream_url: str = ""
        self._title: str = ""
        self._artist: str = ""
        self._client_task: asyncio.Task | None = None

    @property
    def port(self) -> int:
        """Return the port the server is listening on."""
        return self._port

    def set_stream(self, url: str, title: str, artist: str) -> None:
        """
        Update the upstream stream URL and display metadata.

        :param url: Full MA stream URL to proxy.
        :param title: Track title shown on line 1 of the display.
        :param artist: Artist name shown on line 2 of the display.
        """
        self._upstream_url = url
        self._title = title
        self._artist = artist

    def stream_title(self) -> str:
        """Return the formatted display title for ICY headers."""
        parts = [p for p in (self._artist, self._title) if p]
        return " - ".join(parts) or "Music Assistant"

    async def start(self) -> None:
        """Start the proxy server."""
        self._server = await asyncio.start_server(
            self._handle_client, "0.0.0.0", self._port
        )
        self._port = self._server.sockets[0].getsockname()[1]
        _LOGGER.debug("ICY proxy listening on port %d", self._port)

    async def stop(self) -> None:
        """Stop the proxy server."""
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Cancel any still-running connection so only one MA stream is active.
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
        self._client_task = asyncio.current_task()
        try:
            await self._serve(reader, writer)
        except asyncio.CancelledError:
            pass
        except Exception as err:
            _LOGGER.debug("ICY client error: %s", err)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Read and discard the SoundBridge's HTTP request headers.
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            if not chunk:
                return
            raw += chunk

        url = self._upstream_url
        if not url:
            writer.write(b"ICY 503 No stream\r\n\r\n")
            await writer.drain()
            return

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    _LOGGER.warning(
                        "Upstream returned HTTP %d for %s", resp.status, url
                    )
                    return

                # Build the ICY response.
                # Start with the non-standard ICY status line the SoundBridge requires.
                icy_lines = [
                    "ICY 200 OK",
                    f"icy-name: {self.stream_title()}",
                    f"icy-description: {self._artist}",
                ]
                # Pass through the remaining upstream headers, skipping ones we override
                # or that are irrelevant/harmful for the SoundBridge.
                for key, val in resp.headers.items():
                    if key.lower() not in _SUPPRESS_UPSTREAM_HEADERS:
                        icy_lines.append(f"{key}: {val}")
                icy_lines += ["", ""]
                writer.write("\r\n".join(icy_lines).encode())
                await writer.drain()

                # Stream the audio body verbatim.
                async for chunk in resp.content.iter_chunked(8192):
                    writer.write(chunk)
                    await writer.drain()
