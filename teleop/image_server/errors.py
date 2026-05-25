"""Custom error classes for device connection management."""


class DeviceAlreadyConnectedError(ConnectionError):
    """Raised when attempting to connect a device that is already connected."""

    def __init__(self, message="This device is already connected."):
        super().__init__(message)


class DeviceNotConnectedError(ConnectionError):
    """Raised when attempting to use a device that is not connected."""

    def __init__(self, message="This device is not connected. Try calling `connect()` first."):
        super().__init__(message)
