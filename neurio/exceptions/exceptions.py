#!/user/bin/env python

class DeviceNotReadyException(Exception):
    """
    Exception raised when the device is not ready for inference.
    """
    pass

class InvalidImageRangeError(Exception):
    """
    Exception raised when the image range is invalid with respect to its format.
    """
    pass