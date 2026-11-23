"""Constants for the Roku SoundBridge player provider."""

from typing import Final

DEFAULT_PORT: Final[int] = 5555
DEFAULT_NAME: Final[str] = "SoundBridge"
MDNS_SERVICE_TYPE: Final[str] = "_roku-rcp._tcp.local."
PLAYER_ID_PREFIX: Final[str] = "soundbridge_"
