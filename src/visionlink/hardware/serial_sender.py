"""USB serial output for Arduino — FR-8."""

from visionlink.exceptions import SerialConnectionError


class SerialSender:
    """Send newline-terminated ASCII commands over USB serial."""

    def __init__(self, port: str, baud_rate: int = 9600) -> None:
        try:
            import serial
        except ImportError as exc:
            raise SerialConnectionError(
                "pyserial is required for Arduino support. "
                'Install with: pip install -e ".[arduino]"'
            ) from exc

        try:
            self._port = serial.Serial(port, baud_rate, timeout=1)
        except serial.SerialException as exc:
            raise SerialConnectionError(f"Could not open serial port {port!r}: {exc}") from exc

        self._port_name = port

    @property
    def port(self) -> str:
        return self._port_name

    def send(self, command: str) -> None:
        """Send a single command line to the Arduino."""
        line = f"{command.strip()}\n"
        self._port.write(line.encode("ascii"))
        self._port.flush()

    def send_commands(self, commands: list[str]) -> None:
        for command in commands:
            self.send(command)

    def close(self) -> None:
        if self._port.is_open:
            self._port.close()

    def __enter__(self) -> "SerialSender":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
