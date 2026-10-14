"""Structured launcher errors are distinct from generated-program diagnostics."""


class SolverLaunchError(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence
