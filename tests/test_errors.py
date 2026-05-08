"""Error envelope shape + status-code mapping for each typed exception."""

from __future__ import annotations

import pytest

from openai_api_bridge.errors import (
    BridgeError,
    GenerationTimeout,
    ImageRequired,
    InvalidRequest,
    JobNotFound,
    JobNotReady,
    ModelNotFound,
    ProviderNotFound,
    Unauthorized,
    UnsupportedOperation,
    UpstreamError,
    WorkflowInvalid,
    error_payload,
)


@pytest.mark.parametrize(
    ("exc_cls", "status", "type_", "code"),
    [
        (Unauthorized, 401, "invalid_request_error", "invalid_api_key"),
        (ModelNotFound, 404, "invalid_request_error", "model_not_found"),
        (ProviderNotFound, 404, "invalid_request_error", "provider_not_found"),
        (ImageRequired, 400, "invalid_request_error", "image_required"),
        (UnsupportedOperation, 400, "invalid_request_error", "unsupported_operation"),
        (WorkflowInvalid, 400, "invalid_request_error", "workflow_invalid"),
        (InvalidRequest, 400, "invalid_request_error", "invalid_request"),
        (JobNotFound, 404, "invalid_request_error", "not_found"),
        (JobNotReady, 409, "invalid_request_error", "job_not_ready"),
        (UpstreamError, 502, "api_error", "upstream_error"),
        (GenerationTimeout, 504, "api_error", "generation_timeout"),
    ],
)
def test_typed_exceptions_have_correct_envelope(
    exc_cls: type[BridgeError], status: int, type_: str, code: str,
) -> None:
    e = exc_cls("test message", param="x")
    assert e.status_code == status
    assert e.error_type == type_
    assert e.code == code
    assert e.message == "test message"
    assert e.param == "x"


def test_error_payload_structure() -> None:
    body = error_payload(
        message="oops",
        type_="invalid_request_error",
        code="bad_thing",
        param="model",
    )
    assert body == {
        "error": {
            "message": "oops",
            "type": "invalid_request_error",
            "param": "model",
            "code": "bad_thing",
        }
    }


def test_error_payload_optional_fields_default_to_none() -> None:
    body = error_payload(message="x", type_="api_error")
    assert body["error"]["param"] is None
    assert body["error"]["code"] is None
