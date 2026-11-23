"""Roku SoundBridge player provider for Music Assistant."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType
from zeroconf import ServiceStateChange

from music_assistant.constants import CONF_ENTRY_MANUAL_DISCOVERY_IPS
from music_assistant.helpers.util import (
    get_port_from_zeroconf,
    get_primary_ip_address_from_zeroconf,
)
from music_assistant.models.player_provider import PlayerProvider

from .constants import DEFAULT_NAME, DEFAULT_PORT, PLAYER_ID_PREFIX
from .player import SoundBridgePlayer

if TYPE_CHECKING:
    from zeroconf.asyncio import AsyncServiceInfo

CONF_MANUAL_IPS = CONF_ENTRY_MANUAL_DISCOVERY_IPS.key


def parse_host_entry(entry: str) -> tuple[str, int] | None:
    """
    Parse a manual host configuration entry into ``(host, port)``.

    Accepts ``host`` or ``host:port`` (defaulting port to ``5555``).

    :param entry: Raw host configuration string.
    :return: Tuple of ``(host, port)`` or ``None`` if blank.
    """
    cleaned = entry.strip()
    if not cleaned:
        return None
    if ":" in cleaned:
        host_part, port_part = cleaned.rsplit(":", 1)
        try:
            port = int(port_part)
        except ValueError:
            return cleaned, DEFAULT_PORT
        host_part = host_part.strip()
        if host_part and 1 <= port <= 65535:
            return host_part, port
    return cleaned, DEFAULT_PORT


def make_protocol_id(host: str, port: int = DEFAULT_PORT) -> str:
    """
    Build a deterministic protocol identifier from ``host`` and ``port``.

    :param host: Hostname or IPv4 address.
    :param port: RCP TCP port.
    :return: Protocol identifier string.
    """
    sanitized = host.strip().lower().replace(".", "_").replace(":", "_")
    if port == DEFAULT_PORT:
        return f"{PLAYER_ID_PREFIX}{sanitized}"
    return f"{PLAYER_ID_PREFIX}{sanitized}_{port}"


def extract_mdns_friendly_name(service_name: str, fallback_host: str) -> str:
    """
    Extract a human-friendly player name from a Zeroconf service name.

    :param service_name: Full mDNS service name (e.g. ``Kitchen._roku-rcp._tcp.local.``).
    :param fallback_host: Host/IP used to disambiguate generic default names.
    :return: Display name string.
    """
    instance = service_name.split("._", 1)[0].strip()
    if not instance or instance.lower() == DEFAULT_NAME.lower():
        return f"{DEFAULT_NAME} ({fallback_host})"
    return instance


class SoundBridgeProvider(PlayerProvider):
    """
    Player provider for Roku and Pinnacle SoundBridge devices.

    Supports multiple SoundBridge units simultaneously via both automatic
    Bonjour/mDNS discovery (``_roku-rcp._tcp.local.``) and an optional list of
    manual ``host`` / ``host:port`` entries for devices across subnets/VLANs.
    """

    _players: dict[str, SoundBridgePlayer]
    _manual_protocol_ids: set[str]
    _mdns_protocol_ids: set[str]

    async def handle_async_init(self) -> None:
        """Initialize provider state containers."""
        self._players = {}
        self._manual_protocol_ids = set()
        self._mdns_protocol_ids = set()
        self._register_lock = asyncio.Lock()

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return provider configuration entries (optional manual host list)."""
        return (
            ConfigEntry(
                key=CONF_MANUAL_IPS,
                type=ConfigEntryType.STRING,
                default_value=[],
                required=False,
                multi_value=True,
            ),
        )

    async def loaded_in_mass(self) -> None:
        """Reconcile removed manual host entries when provider config is loaded or updated."""
        entries = cast("list[str]", self.config.get_value(CONF_MANUAL_IPS) or [])
        parsed = [item for raw in entries if (item := parse_host_entry(raw)) is not None]
        current_manual_ids: set[str] = set()
        for host, port in parsed:
            existing = self._find_existing_player(make_protocol_id(host, port), host, port, None)
            current_manual_ids.add(
                existing.protocol_id if existing is not None else make_protocol_id(host, port)
            )

        removed_manual_ids = (
            self._manual_protocol_ids - current_manual_ids - self._mdns_protocol_ids
        )
        self._manual_protocol_ids = current_manual_ids
        for protocol_id in removed_manual_ids:
            self._players.pop(protocol_id, None)
            if self.mass.config.get_protocol_record(protocol_id) or self.get_protocol(protocol_id):
                await self.mass.players.remove_protocol(protocol_id)

    async def discover_players(self) -> None:
        """Register manually configured SoundBridge hosts; mDNS discovers local subnet devices."""
        entries = cast("list[str]", self.config.get_value(CONF_MANUAL_IPS) or [])
        for raw in entries:
            parsed = parse_host_entry(raw)
            if parsed is None:
                continue
            host, port = parsed
            resolved_ip = await self._resolve_ipv4(host, port)
            protocol_id = make_protocol_id(host, port)
            player = await self._register_endpoint(
                protocol_id=protocol_id,
                host=host,
                port=port,
                name=f"{DEFAULT_NAME} ({host})",
                resolved_ip=resolved_ip,
            )
            if player is not None:
                self._manual_protocol_ids.add(player.protocol_id)

    async def on_mdns_service_state_change(
        self,
        name: str,
        state_change: ServiceStateChange,
        info: AsyncServiceInfo | None,
    ) -> None:
        """
        Handle mDNS ``_roku-rcp._tcp.local.`` discovery announcements.

        :param name: Full mDNS service name.
        :param state_change: Zeroconf service state transition.
        :param info: Resolved service info, if available.
        """
        if state_change == ServiceStateChange.Removed or info is None:
            return

        ip_address = get_primary_ip_address_from_zeroconf(info, prefer_ipv6=False)
        if ip_address is None:
            return
        port = get_port_from_zeroconf(info) or DEFAULT_PORT
        friendly_name = extract_mdns_friendly_name(name, ip_address)
        protocol_id = make_protocol_id(ip_address, port)

        player = await self._register_endpoint(
            protocol_id=protocol_id,
            host=ip_address,
            port=port,
            name=friendly_name,
            resolved_ip=ip_address,
        )
        if player is not None:
            self._mdns_protocol_ids.add(player.protocol_id)

    async def on_protocol_removed(self, protocol_id: str) -> None:
        """
        Clean up provider bookkeeping when a protocol is permanently removed.

        :param protocol_id: ID of the removed protocol.
        """
        player = self._players.pop(protocol_id, None)
        self._manual_protocol_ids.discard(protocol_id)
        self._mdns_protocol_ids.discard(protocol_id)
        if player is not None:
            self.forget_host(player.host, player.port)

    def forget_host(self, host: str, port: int) -> None:
        """
        Remove ``(host, port)`` from ``manual_discovery_ip_addresses`` if present.

        :param host: Hostname or IP address of the SoundBridge.
        :param port: RCP TCP port of the SoundBridge.
        """
        for pid, tracked in list(self._players.items()):
            if (tracked.host, tracked.port) == (host, port):
                self._players.pop(pid, None)
                self._manual_protocol_ids.discard(pid)

        entries = cast("list[str]", self.config.get_value(CONF_MANUAL_IPS) or [])
        new_entries = [raw for raw in entries if parse_host_entry(raw) not in {(host, port), None}]
        if new_entries != entries:
            self._update_config_value(CONF_MANUAL_IPS, new_entries)

    async def _register_endpoint(
        self,
        protocol_id: str,
        host: str,
        port: int,
        name: str,
        resolved_ip: str | None = None,
    ) -> SoundBridgePlayer | None:
        """Register a new SoundBridgePlayer or reconcile an already-known one."""
        async with self._register_lock:
            existing = self._find_existing_player(protocol_id, host, port, resolved_ip)
            if existing is not None:
                await existing.update_endpoint(
                    host=existing.host, port=port, resolved_ip=resolved_ip
                )
                if self.get_protocol(
                    existing.protocol_id
                ) is None and self.mass.players.wants_protocol(existing.protocol_id):
                    await self.mass.players.register(existing)
                return existing

            if not self.mass.players.wants_protocol(protocol_id):
                return None

            player = SoundBridgePlayer(
                provider=self,
                protocol_id=protocol_id,
                host=host,
                port=port,
                name=name,
                resolved_ip=resolved_ip,
            )
            self._players[protocol_id] = player
            await self.mass.players.register(player)
            return player

    def _find_existing_player(
        self,
        protocol_id: str,
        host: str,
        port: int,
        resolved_ip: str | None,
    ) -> SoundBridgePlayer | None:
        """Find an already-known SoundBridgePlayer matching ID, host, or IPv4 address."""
        if direct := self._players.get(protocol_id):
            return direct
        candidate_addrs = {host.lower()}
        if resolved_ip:
            candidate_addrs.add(resolved_ip.lower())
        for player in self._players.values():
            if player.port != port:
                continue
            known_addrs = {player.host.lower()}
            if player.resolved_ip:
                known_addrs.add(player.resolved_ip.lower())
            if candidate_addrs & known_addrs:
                return player
        return None

    @staticmethod
    async def _resolve_ipv4(host: str, port: int) -> str | None:
        """Resolve ``host`` to an IPv4 address string without blocking the event loop."""
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return str(addr) if isinstance(addr, ipaddress.IPv4Address) else None

        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host,
                port,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        except OSError:
            return None
        if not infos:
            return None
        sockaddr = infos[0][4]
        return str(sockaddr[0]) if sockaddr else None
