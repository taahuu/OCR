"""ASGI middleware."""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import get_logger

logger = get_logger("middleware")

# Allowance for multipart boundaries and headers on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 256 * 1024


class RequestSizeLimitMiddleware:
    """Reject oversized request bodies before they are fully buffered.

    Two layers:
    1. A declared ``Content-Length`` above the limit is rejected immediately.
    2. For streamed/chunked bodies the bytes actually received are counted and
       the request is aborted as soon as the limit is crossed.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                declared_size = 0
            if declared_size > self.max_body_bytes:
                logger.info("Rejected request: declared body exceeds limit")
                await self._reject(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    logger.info("Aborted request: streamed body exceeds limit")
                    raise HTTPException(status_code=413, detail="payload too large")
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        from fastapi.responses import JSONResponse

        response = JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "file_too_large",
                    "message": "The uploaded file exceeds the maximum allowed size.",
                }
            },
        )
        await response(scope, receive, send)
