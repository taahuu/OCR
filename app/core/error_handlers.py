"""Exception handlers that turn errors into safe, uniform JSON responses.

Every error body has the shape::

    {"error": {"code": "<machine_code>", "message": "<safe message>"}}
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger("errors")

_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "file_too_large",
    415: "unsupported_format",
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    level = logger.warning if exc.status_code >= 500 else logger.info
    level("Request failed: code=%s status=%s", exc.code, exc.status_code)
    return _error_response(exc.status_code, exc.code, exc.message)


async def _handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_CODES.get(exc.status_code, f"http_{exc.status_code}")
    message = (
        "The uploaded file exceeds the maximum allowed size."
        if exc.status_code == 413
        else str(exc.detail)
    )
    return _error_response(exc.status_code, code, message)


async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Report only field locations and messages, never the submitted values.
    problems = [
        f"{'.'.join(str(part) for part in err['loc'] if part != 'body')}: {err['msg']}"
        for err in exc.errors()
    ]
    return _error_response(422, "validation_error", "Invalid request. " + "; ".join(problems))


async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    # Traceback goes to the server log only; the client gets a generic message.
    logger.error("Unhandled error: %s", type(exc).__name__, exc_info=exc)
    return _error_response(500, "internal_error", "An internal error occurred.")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the application."""
    app.add_exception_handler(AppError, _handle_app_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unexpected)
