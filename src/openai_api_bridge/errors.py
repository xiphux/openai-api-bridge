"""OpenAI-shaped error envelope and typed exceptions.

Every error returned to a client must conform to:

    {"error": {"message": str, "type": str, "param": str|null, "code": str|null}}

Backend code raises typed exceptions; a FastAPI exception handler
converts them to the right HTTP status + envelope.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class BridgeError(Exception):
    """Base for any error the bridge intentionally surfaces."""

    status_code: int = 500
    error_type: str = "api_error"
    code: str = "internal_error"

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


# 4xx — client errors


class Unauthorized(BridgeError):
    status_code = 401
    error_type = "invalid_request_error"
    code = "invalid_api_key"


class ModelNotFound(BridgeError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class ProviderNotFound(BridgeError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "provider_not_found"


class ImageRequired(BridgeError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "image_required"


class UnsupportedOperation(BridgeError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "unsupported_operation"


class WorkflowInvalid(BridgeError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "workflow_invalid"


class InvalidRequest(BridgeError):
    """Generic 400 for anything that doesn't fit a more specific subclass."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class JobNotFound(BridgeError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "not_found"


class JobNotReady(BridgeError):
    status_code = 409
    error_type = "invalid_request_error"
    code = "job_not_ready"


# 5xx — upstream / infra errors


class UpstreamError(BridgeError):
    status_code = 502
    error_type = "api_error"
    code = "upstream_error"


class UpstreamAuthError(UpstreamError):
    """The upstream rejected our credentials (401/403).

    Split from the generic UpstreamError because it is **not** transient:
    provider tokens are read from the environment once at startup, so a
    rejected credential cannot start working again without a restart. Backends
    that retry failed calls should treat this as permanent and stop, rather
    than re-attempting on a cooldown forever against a key that will never
    work.
    """

    code = "upstream_auth_error"


class GenerationTimeout(BridgeError):
    status_code = 504
    error_type = "api_error"
    code = "generation_timeout"


# Helpers


def error_payload(
    *,
    message: str,
    type_: str,
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": type_,
            "param": param,
            "code": code,
        }
    }


async def bridge_error_handler(_request: Request, exc: BridgeError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            message=exc.message,
            type_=exc.error_type,
            code=exc.code,
            param=exc.param,
        ),
    )
