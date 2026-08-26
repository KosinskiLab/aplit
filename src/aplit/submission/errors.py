"""Submission-domain exceptions with messages suitable for the UI."""


class SubmissionError(Exception):
    """Base class for expected submission failures."""


class ValidationError(SubmissionError):
    """The requested run or site configuration is invalid."""


class ProjectCollisionError(SubmissionError):
    """An existing project cannot be safely resumed."""


class ProjectBusyError(SubmissionError):
    """Another process is mutating or controlling the project."""


class RunNotFoundError(SubmissionError):
    """The requested run project has no APLit manifest."""
