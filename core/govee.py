"""
Sterling Govee Integration
==========================
Two backends, same interface:

  GoveeCloud  — Govee HTTP API (api.developer.govee.com)
                Needs an API key and device model (SKU).
                Run scripts/discover_govee.py to get both.

  GoveeLocal  — Local LAN UDP (no cloud, no API key)
                Needs Local Control enabled in the Govee Home app.
                Run scripts/discover_govee.py --local to scan.

main.py picks GoveeCloud when govee.api_key is set in config,
GoveeLocal otherwise.
"""

import json
import socket
import time
import requests
from dataclasses import dataclass
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("sterling.govee")

# ─────────────────────────────────────────────────────────────────────────────
# Color table — shared by both backends and the intent parser in main.py
# ─────────────────────────────────────────────────────────────────────────────

COLORS: dict[str, tuple[int, int, int]] = {
    "red":        (255,   0,   0),
    "green":      (  0, 200,   0),
    "blue":       (  0,   0, 255),
    "white":      (255, 255, 255),
    "warm white": (255, 200, 100),
    "yellow":     (255, 220,   0),
    "orange":     (255, 100,   0),
    "purple":     (128,   0, 255),
    "pink":       (255,   0, 150),
    "cyan":       (  0, 255, 255),
    "teal":       (  0, 180, 180),
    "magenta":    (255,   0, 255),
    "lime":       (150, 255,   0),
    "indigo":     ( 75,   0, 130),
    "lavender":   (150, 100, 255),
}


# ─────────────────────────────────────────────────────────────────────────────
# Device data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoveeDevice:
    name:      str
    device_id: str
    model:     str        # SKU e.g. "H6160" — required for cloud API
    ip:        str = ""   # only used by GoveeLocal


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_color(color_name: str) -> Optional[tuple[int, int, int]]:
    """Return RGB tuple for a colour name, with fuzzy substring fallback."""
    name = color_name.lower().strip()
    if name in COLORS:
        return COLORS[name]
    for known, rgb in COLORS.items():
        if known in name or name in known:
            return rgb
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cloud API backend
# ─────────────────────────────────────────────────────────────────────────────

CLOUD_CONTROL_URL = "https://developer-api.govee.com/v1/devices/control"
CLOUD_DEVICES_URL = "https://developer-api.govee.com/v1/devices"


class GoveeCloud:
    """
    Govee HTTP cloud API controller.

    Requires:
      - api_key:   from developer.govee.com (free)
      - device_id: MAC-style ID e.g. "3C:E1:CA:38:32:34:5B:67"
      - model:     SKU e.g. "H6160" — run discover_govee.py to find it

    Rate limit: ~100 req/min on the free tier (plenty for voice commands).
    """

    def __init__(self, api_key: str, devices: list[dict]):
        self._api_key = api_key
        self._headers = {
            "Govee-API-Key": api_key,
            "Content-Type": "application/json",
        }
        self._devices: list[GoveeDevice] = [
            GoveeDevice(
                name=d.get("name", f"Light {i + 1}"),
                device_id=d["device_id"],
                model=d.get("model", ""),
            )
            for i, d in enumerate(devices)
            if "device_id" in d
        ]

        if not self._devices:
            logger.warning(
                "GoveeCloud: no devices configured. "
                "Run: python scripts/discover_govee.py"
            )
        else:
            names = [d.name for d in self._devices]
            logger.info(f"GoveeCloud ready — {len(self._devices)} device(s): {names}")

        # Warn if any device is missing a model — the API will reject those calls
        for d in self._devices:
            if not d.model:
                logger.warning(
                    f"GoveeCloud: device '{d.name}' has no model (SKU). "
                    "API calls will fail. Run discover_govee.py to find it."
                )

    # ── Control ───────────────────────────────────────────────────────────────

    def turn_on(self, device_name: Optional[str] = None):
        self._send({"name": "turn", "value": "on"}, device_name)
        logger.info(f"Govee: ON — {device_name or 'all'}")

    def turn_off(self, device_name: Optional[str] = None):
        self._send({"name": "turn", "value": "off"}, device_name)
        logger.info(f"Govee: OFF — {device_name or 'all'}")

    def set_brightness(self, percent: int, device_name: Optional[str] = None):
        percent = max(0, min(100, percent))
        self._send({"name": "brightness", "value": percent}, device_name)
        logger.info(f"Govee: brightness {percent}% — {device_name or 'all'}")

    def set_color(self, r: int, g: int, b: int, device_name: Optional[str] = None):
        self._send({"name": "color", "value": {"r": r, "g": g, "b": b}}, device_name)
        logger.info(f"Govee: color rgb({r},{g},{b}) — {device_name or 'all'}")

    def set_color_by_name(self, color_name: str, device_name: Optional[str] = None) -> bool:
        rgb = _resolve_color(color_name)
        if rgb:
            self.set_color(*rgb, device_name=device_name)
            return True
        logger.warning(f"Govee: unknown color '{color_name}'")
        return False

    @property
    def has_devices(self) -> bool:
        return bool(self._devices)

    @property
    def device_names(self) -> list[str]:
        return [d.name for d in self._devices]

    def close(self):
        pass  # No persistent connection to clean up

    # ── Cloud helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def list_devices(api_key: str) -> list[dict]:
        """
        Fetch all devices registered to the account.
        Used by scripts/discover_govee.py to find device IDs and models.
        """
        try:
            resp = requests.get(
                CLOUD_DEVICES_URL,
                headers={"Govee-API-Key": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("data", {}).get("devices", [])
        except Exception as e:
            logger.error(f"GoveeCloud: list_devices failed: {e}")
            return []

    def _send(self, cmd: dict, device_name: Optional[str] = None):
        """POST a control command to one or all devices."""
        targets = self._devices
        if device_name:
            needle = device_name.lower()
            targets = [d for d in self._devices if needle in d.name.lower()]
            if not targets:
                logger.warning(f"GoveeCloud: no device matching '{device_name}'")
                return

        for device in targets:
            if not device.model:
                logger.error(
                    f"GoveeCloud: skipping '{device.name}' — model (SKU) not set. "
                    "Add it to config.yaml."
                )
                continue
            body = {
                "device": device.device_id,
                "model":  device.model,
                "cmd":    cmd,
            }
            try:
                resp = requests.put(
                    CLOUD_CONTROL_URL,
                    headers=self._headers,
                    json=body,
                    timeout=10,
                )
                if not resp.ok:
                    logger.error(
                        f"GoveeCloud: API error {resp.status_code} for '{device.name}': "
                        f"{resp.text}"
                    )
            except Exception as e:
                logger.error(f"GoveeCloud: request failed for '{device.name}': {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Local LAN backend
# ─────────────────────────────────────────────────────────────────────────────

DISCOVERY_HOST = "239.255.255.250"
DISCOVERY_PORT = 4001
LISTEN_PORT    = 4002
CONTROL_PORT   = 4003


class GoveeLocal:
    """
    Govee Local LAN controller (UDP — no cloud required).

    Requirements:
      - Govee Home app → Device → More Settings → Local Control → ON
      - Device on the same WiFi network as Sterling
    """

    def __init__(self, devices: list[dict], discovery_timeout: float = 3.0):
        self._devices: list[GoveeDevice] = [
            GoveeDevice(
                name=d.get("name", f"Light {i + 1}"),
                device_id=d.get("device_id", ""),
                model=d.get("model", ""),
                ip=d["ip"],
            )
            for i, d in enumerate(devices)
            if "ip" in d
        ]
        self._discovery_timeout = discovery_timeout
        self._sock: Optional[socket.socket] = None
        self._init_socket()

        if self._devices:
            names = [d.name for d in self._devices]
            logger.info(f"GoveeLocal ready — {len(self._devices)} device(s): {names}")
        else:
            logger.warning(
                "GoveeLocal: no devices configured. "
                "Run: python scripts/discover_govee.py --local"
            )

    def _init_socket(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.settimeout(1.0)
        except Exception as e:
            logger.error(f"GoveeLocal: socket init failed: {e}")

    # ── Control ───────────────────────────────────────────────────────────────

    def turn_on(self, device_name: Optional[str] = None):
        self._send({"cmd": "turn", "data": {"value": 1}}, device_name)
        logger.info(f"Govee: ON — {device_name or 'all'}")

    def turn_off(self, device_name: Optional[str] = None):
        self._send({"cmd": "turn", "data": {"value": 0}}, device_name)
        logger.info(f"Govee: OFF — {device_name or 'all'}")

    def set_brightness(self, percent: int, device_name: Optional[str] = None):
        percent = max(0, min(100, percent))
        self._send({"cmd": "brightness", "data": {"value": percent}}, device_name)
        logger.info(f"Govee: brightness {percent}% — {device_name or 'all'}")

    def set_color(self, r: int, g: int, b: int, device_name: Optional[str] = None):
        self._send({
            "cmd": "colorwc",
            "data": {"color": {"r": r, "g": g, "b": b}, "colorTemInKelvin": 0},
        }, device_name)
        logger.info(f"Govee: color rgb({r},{g},{b}) — {device_name or 'all'}")

    def set_color_by_name(self, color_name: str, device_name: Optional[str] = None) -> bool:
        rgb = _resolve_color(color_name)
        if rgb:
            self.set_color(*rgb, device_name=device_name)
            return True
        logger.warning(f"GoveeLocal: unknown color '{color_name}'")
        return False

    @property
    def has_devices(self) -> bool:
        return bool(self._devices)

    @property
    def device_names(self) -> list[str]:
        return [d.name for d in self._devices]

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    # ── LAN discovery ─────────────────────────────────────────────────────────

    def discover(self) -> list[GoveeDevice]:
        found: list[GoveeDevice] = []
        try:
            listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_sock.bind(("", LISTEN_PORT))
            listen_sock.settimeout(self._discovery_timeout)

            scan = json.dumps({"msg": {"cmd": "scan", "data": {"account_topic": "reserve"}}}).encode()
            tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            tx.sendto(scan, (DISCOVERY_HOST, DISCOVERY_PORT))
            tx.close()

            deadline = time.time() + self._discovery_timeout
            while time.time() < deadline:
                try:
                    data, addr = listen_sock.recvfrom(1024)
                    d   = json.loads(data.decode()).get("msg", {}).get("data", {})
                    ip  = d.get("ip") or addr[0]
                    dev = GoveeDevice(
                        name=d.get("sku", f"Govee_{ip}"),
                        device_id=d.get("device", ""),
                        model=d.get("sku", ""),
                        ip=ip,
                    )
                    found.append(dev)
                    logger.info(f"  Found: {dev.model} at {dev.ip} ({dev.device_id})")
                except socket.timeout:
                    break
                except Exception:
                    continue

            listen_sock.close()
        except Exception as e:
            logger.error(f"GoveeLocal discovery error: {e}")
        return found

    def _send(self, command: dict, device_name: Optional[str] = None):
        if not self._sock:
            return
        targets = self._devices
        if device_name:
            needle = device_name.lower()
            targets = [d for d in self._devices if needle in d.name.lower()]
        payload = json.dumps({"msg": command}).encode()
        for device in targets:
            try:
                self._sock.sendto(payload, (device.ip, CONTROL_PORT))
            except Exception as e:
                logger.error(f"GoveeLocal: send failed to {device.name}: {e}")
