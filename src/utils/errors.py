"""Errors that decide whether the experiment controller may retry automatically."""


class ProtocolError(RuntimeError):
    """An immutable protocol or artifact changed; do not retry automatically."""


class InfrastructureError(RuntimeError):
    """Retry the pending stage without advancing the scored attempt number."""
