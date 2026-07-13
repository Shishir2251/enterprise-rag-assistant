from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    ApplicationError,
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DocumentProcessingError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationError,
)


STATUS_BY_EXCEPTION: dict[type[ApplicationError], int] = {
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
    PayloadTooLargeError: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    DocumentProcessingError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ValidationError: status.HTTP_400_BAD_REQUEST,
}


def register_exception_handlers(app: FastAPI) -> None:
    """Register transport mappings for application-layer exceptions."""

    @app.exception_handler(ApplicationError)
    async def handle_application_error(
        request: Request,
        exc: ApplicationError,
    ) -> JSONResponse:
        del request
        status_code = next(
            (
                code
                for exception_type, code in STATUS_BY_EXCEPTION.items()
                if isinstance(exc, exception_type)
            ),
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        headers = None
        if isinstance(exc, AuthenticationError):
            headers = {"WWW-Authenticate": "Bearer"}

        return JSONResponse(
            status_code=status_code,
            content={"detail": exc.detail},
            headers=headers,
        )
