class ApplicationError(Exception):
    """Base class for errors safe to expose through the API."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ValidationError(ApplicationError):
    """Raised when application input fails validation."""


class PayloadTooLargeError(ValidationError):
    """Raised when an uploaded payload exceeds the configured limit."""


class NotFoundError(ApplicationError):
    """Raised when an owned resource cannot be found."""


class ConflictError(ApplicationError):
    """Raised when an operation conflicts with existing state."""


class AuthenticationError(ApplicationError):
    """Raised when credentials are invalid."""


class AuthorizationError(ApplicationError):
    """Raised when an authenticated principal is not allowed to act."""


class DocumentProcessingError(ApplicationError):
    """Raised when a document cannot be safely processed."""


class EmbeddingError(ApplicationError):
    """Raised when an embedding provider returns an invalid result."""


class RetrievalError(ApplicationError):
    """Raised when retrieval infrastructure cannot complete a search."""


class LLMError(ApplicationError):
    """Raised when an LLM provider returns an invalid result."""


class ConfigurationError(ApplicationError):
    """Raised when a required application integration is not configured."""


class QueueUnavailableError(ApplicationError):
    """Raised when a background job cannot be queued."""
