"""Hardware integration (Arduino serial)."""

from visionlink.hardware.bridge import GestureSerialBridge
from visionlink.hardware.gesture_commands import gestures_to_commands
from visionlink.hardware.serial_sender import SerialSender

__all__ = ["GestureSerialBridge", "SerialSender", "gestures_to_commands"]
