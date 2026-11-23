"""Stable error categories shared by clients and workers."""

class VVLAError(Exception):
    pass


class InputError(VVLAError, ValueError):
    """Rejected before model execution; the owner remains usable."""


class UnsupportedConfiguration(InputError):
    pass


class ClosedError(VVLAError, RuntimeError):
    pass


class WorkerFatalError(VVLAError, RuntimeError):
    """Owner state is no longer trusted. Create a new worker to recover."""


class BackendLoadError(WorkerFatalError):
    pass


class AdmissionRejected(VVLAError, TimeoutError):
    """A deadline prediction rejected work before model execution."""
    def __init__(self, event):
        self.event = dict(event)
        super().__init__(event['status'])

