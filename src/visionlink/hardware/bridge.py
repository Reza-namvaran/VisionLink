"""Send gestures to Arduino only when the command set changes."""

import logging

from visionlink.hardware.gesture_commands import gestures_to_commands
from visionlink.hardware.serial_sender import SerialSender
from visionlink.models import Face

logger = logging.getLogger(__name__)


class GestureSerialBridge:
    """Bridge between VisionLink faces and an Arduino serial receiver."""

    def __init__(self, sender: SerialSender) -> None:
        self._sender = sender
        self._last_signature: tuple[str, ...] = ()

    def publish(self, faces: list[Face]) -> list[str]:
        """Send commands when gestures change. Returns the commands sent."""
        commands = gestures_to_commands(faces)
        signature = tuple(commands)

        if signature == self._last_signature:
            return []

        self._sender.send_commands(commands)
        self._last_signature = signature
        logger.debug("Serial → %s", ", ".join(commands))
        return commands

    def reset(self) -> None:
        self._last_signature = ()

    def close(self) -> None:
        self._sender.close()
