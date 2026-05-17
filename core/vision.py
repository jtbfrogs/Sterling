"""
Sterling Vision Module — HuskyLens2
Communicates with HuskyLens2 over USB-Serial (UART) using the binary protocol.

HuskyLens2 modes supported:
  - Face Recognition (0x01)
  - Object Tracking (0x02)
  - Object Recognition (0x03)

Setup:
  1. Connect HuskyLens2 via USB
  2. On device: Settings → Protocol Type → UART
  3. Baud rate: 9600
  4. Find port: ls /dev/tty.usb*

HuskyLens Binary Protocol (simplified):
  Request:  [0x55] [0xAA] [0x11] [LEN] [CMD] [DATA...] [CHECKSUM]
  Response: [0x55] [0xAA] [0x11] [LEN] [CMD] [DATA...] [CHECKSUM]
"""

import struct
import time
import serial
import serial.tools.list_ports
from dataclasses import dataclass
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("sterling.vision")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Block:
    """A detected object bounding box from HuskyLens2."""
    x: int          # Center X
    y: int          # Center Y
    width: int
    height: int
    id: int         # Learned ID (0 = unlearned/unknown)

    @property
    def is_learned(self) -> bool:
        return self.id > 0


@dataclass
class Arrow:
    """A detected arrow (line tracking) from HuskyLens2."""
    x_tail: int
    y_tail: int
    x_head: int
    y_head: int
    id: int


# ─────────────────────────────────────────────────────────────────────────────
# HuskyLens Protocol Constants
# ─────────────────────────────────────────────────────────────────────────────

HEADER = bytes([0x55, 0xAA, 0x11])

CMD_REQUEST_ALL            = 0x20
CMD_REQUEST_BLOCKS         = 0x21
CMD_REQUEST_ARROWS         = 0x22
CMD_REQUEST_LEARNED        = 0x23
CMD_REQUEST_BLOCKS_LEARNED = 0x24
CMD_RETURN_INFO            = 0x29
CMD_RETURN_BLOCK           = 0x2A
CMD_RETURN_ARROW           = 0x2B
CMD_REQUEST_KNOCK          = 0x2C   # correct ping command
CMD_RETURN_OK              = 0x2E   # device ACK

ALGORITHM_FACE_RECOGNITION  = 0x00
ALGORITHM_OBJECT_TRACKING   = 0x01
ALGORITHM_OBJECT_RECOGNITION = 0x02
ALGORITHM_LINE_TRACKING     = 0x03
ALGORITHM_COLOR_RECOGNITION = 0x04
ALGORITHM_TAG_RECOGNITION   = 0x05


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class HuskyLens:
    """
    Interface to HuskyLens2 via UART over USB serial.

    Basic usage:
        lens = HuskyLens()  # Auto-detect port
        lens.switch_algorithm(ALGORITHM_FACE_RECOGNITION)
        blocks = lens.get_blocks()
        for block in blocks:
            print(f"Face ID {block.id} at ({block.x}, {block.y})")
        lens.disconnect()
    """

    def __init__(self, port: Optional[str] = None, baud_rate: int = 9600, timeout: float = 1.0):
        """
        Args:
            port:      Serial port path. None = auto-detect from USB descriptors.
            baud_rate: Must match HuskyLens2 settings. Default 9600.
            timeout:   Read timeout in seconds.
        """
        self._port = port or self._auto_detect_port()
        self._baud_rate = baud_rate
        self._timeout = timeout
        self._serial: Optional[serial.Serial] = None
        self._connect()

    # ─────────────────────────────────────────────────────────────────────────
    # Connection
    # ─────────────────────────────────────────────────────────────────────────

    def _auto_detect_port(self) -> str:
        """Scan USB serial ports for likely HuskyLens devices."""
        ports = serial.tools.list_ports.comports()
        candidates = []
        for port in ports:
            desc = (port.description or "").lower()
            dev = port.device.lower()
            if any(k in dev for k in ("usbserial", "usbmodem", "ch340", "cp210")):
                candidates.append(port.device)
            elif any(k in desc for k in ("ch340", "cp2102", "ftdi", "uart")):
                candidates.append(port.device)

        if not candidates:
            available = [p.device for p in ports]
            raise RuntimeError(
                f"HuskyLens2 not found. Available ports: {available}\n"
                "Check USB connection and set vision.port in config.yaml."
            )

        logger.info(f"HuskyLens2 auto-detected on: {candidates[0]}")
        return candidates[0]

    def _connect(self):
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud_rate,
                timeout=self._timeout,
            )
            time.sleep(0.5)  # Allow device to settle after connection
            logger.info(f"HuskyLens2 connected on {self._port} at {self._baud_rate} baud.")
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to connect to HuskyLens2 on {self._port}: {e}") from e

    def disconnect(self):
        """Close the serial connection."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("HuskyLens2 disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    # ─────────────────────────────────────────────────────────────────────────
    # High-level API
    # ─────────────────────────────────────────────────────────────────────────

    def switch_algorithm(self, algorithm: int):
        """
        Switch HuskyLens2 to a different detection mode.
        Allow ~0.5s after switching before querying.
        """
        self._send_command(0x2D, bytes([algorithm]))
        time.sleep(0.5)   # give the lens time to switch and stabilise
        logger.debug(f"Switched to algorithm {algorithm:#04x}")

    def startup_scan(self, face_map: dict = None) -> str:
        """
        Query the camera once at startup and return a human-readable description
        of what it currently sees. Used for boot-time diagnostics.

        Args:
            face_map: Optional {id: name} dict to resolve face IDs to names.

        Returns:
            A short string describing what the camera sees, or an empty string.
        """
        face_map = face_map or {}
        try:
            blocks, _ = self.get_all()
            if not blocks:
                logger.info("HuskyLens2 startup scan: nothing detected in frame.")
                return ""

            parts = []
            for b in blocks:
                if b.is_learned:
                    name = face_map.get(b.id) or face_map.get(str(b.id)) or f"face #{b.id}"
                    parts.append(name)
                else:
                    parts.append(f"unrecognised subject at ({b.x},{b.y})")

            desc = ", ".join(parts)
            logger.info(f"HuskyLens2 startup scan: detected — {desc}")
            return desc
        except Exception as e:
            logger.warning(f"HuskyLens2 startup scan failed: {e}")
            return ""

    def get_blocks(self) -> list[Block]:
        """Request all detected blocks (bounding boxes) from current algorithm."""
        self._send_command(CMD_REQUEST_BLOCKS)
        return self._read_blocks_and_arrows()[0]

    def get_learned_blocks(self) -> list[Block]:
        """Request only learned (ID'd) bounding boxes."""
        self._send_command(CMD_REQUEST_BLOCKS_LEARNED)
        return self._read_blocks_and_arrows()[0]

    def get_all(self) -> tuple[list[Block], list[Arrow]]:
        """Request all blocks and arrows."""
        self._send_command(CMD_REQUEST_ALL)
        return self._read_blocks_and_arrows()

    def ping(self) -> bool:
        """
        Send a knock request and verify the device replies with RETURN_OK (0x2E).
        Expected packet: [0x55, 0xAA, 0x11, 0x00, 0x2E, 0x3E]
        """
        try:
            self._serial.reset_input_buffer()
            self._send_command(CMD_REQUEST_KNOCK)
            resp = self._serial.read(6)

            if len(resp) >= 5 and resp[4] == CMD_RETURN_OK:
                logger.debug("HuskyLens2 ping OK.")
                return True

            # Log what actually came back so the user can diagnose mode issues
            if len(resp) == 0:
                logger.warning(
                    "HuskyLens2 ping: no response (0 bytes). "
                    "The device is likely in I²C mode. "
                    "On the device: Settings → Protocol Type → UART"
                )
            else:
                logger.warning(
                    f"HuskyLens2 ping: unexpected response bytes {[hex(b) for b in resp]}. "
                    f"Expected [0x55, 0xaa, 0x11, 0x00, 0x2e, 0x3e]. "
                    f"Check: Settings → Protocol Type → UART"
                )
            return False

        except Exception as e:
            logger.warning(f"HuskyLens2 ping exception: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Protocol implementation
    # ─────────────────────────────────────────────────────────────────────────

    def _send_command(self, command: int, data: bytes = b""):
        """
        Build and send a HuskyLens command packet.
        Format: [0x55] [0xAA] [0x11] [LEN] [CMD] [DATA...] [CHECKSUM]
        """
        packet = HEADER + bytes([len(data), command]) + data
        checksum = sum(packet) & 0xFF
        packet += bytes([checksum])
        self._serial.write(packet)
        self._serial.flush()

    def _read_blocks_and_arrows(self) -> tuple[list[Block], list[Arrow]]:
        """
        Read the response packets following a request command.
        Returns parsed blocks and arrows.
        """
        blocks: list[Block] = []
        arrows: list[Arrow] = []

        try:
            # First packet should be CMD_RETURN_INFO
            info = self._read_packet()
            if not info or info[0] != CMD_RETURN_INFO:
                return blocks, arrows

            # Parse info packet: [count_low, count_high, type_low, type_high]
            if len(info[1]) >= 2:
                count = struct.unpack_from("<H", info[1], 0)[0]
            else:
                return blocks, arrows

            # Read `count` block/arrow packets
            for _ in range(count):
                pkt = self._read_packet()
                if not pkt:
                    break

                cmd, data = pkt
                if cmd == CMD_RETURN_BLOCK and len(data) >= 10:
                    # Block: x(2) y(2) w(2) h(2) id(2)
                    x, y, w, h, obj_id = struct.unpack_from("<HHHHH", data, 0)
                    blocks.append(Block(x=x, y=y, width=w, height=h, id=obj_id))

                elif cmd == CMD_RETURN_ARROW and len(data) >= 10:
                    # Arrow: x_tail(2) y_tail(2) x_head(2) y_head(2) id(2)
                    xt, yt, xh, yh, obj_id = struct.unpack_from("<HHHHH", data, 0)
                    arrows.append(Arrow(x_tail=xt, y_tail=yt, x_head=xh, y_head=yh, id=obj_id))

        except Exception as e:
            logger.debug(f"Vision read error: {e}")

        return blocks, arrows

    def _read_packet(self) -> Optional[tuple[int, bytes]]:
        """
        Read one response packet from HuskyLens2.

        Returns:
            (command_byte, data_bytes) or None on failure.
        """
        try:
            # Look for 0x55 0xAA header
            while True:
                b = self._serial.read(1)
                if not b:
                    return None
                if b[0] == 0x55:
                    b2 = self._serial.read(1)
                    if b2 and b2[0] == 0xAA:
                        break

            # Read: 0x11, LEN, CMD
            header_rest = self._serial.read(3)
            if len(header_rest) < 3 or header_rest[0] != 0x11:
                return None

            data_len = header_rest[1]
            command = header_rest[2]

            # Read data + checksum
            data = self._serial.read(data_len)
            _checksum = self._serial.read(1)  # We trust it for now

            return command, data

        except Exception as e:
            logger.debug(f"Packet read error: {e}")
            return None
