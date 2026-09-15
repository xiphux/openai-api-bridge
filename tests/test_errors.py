"""Error envelope shape + status-code mapping for each typed exception."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openai_api_bridge.config import reset_caches_for_tests
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
    exc_cls: type[BridgeError],
    status: int,
    type_: str,
    code: str,
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


# --- the handlers create_app installs ----------------------------------------


@pytest.fixture
def raising_client(
    monkeypatch: pytest.MonkeyPatch, empty_config: Path, files_dir: Path, sqlite_path: Path
) -> Iterator[TestClient]:
    """The real app plus one route that fails the way a bug would.

    ``raise_server_exceptions=False`` so the client sees what a real client
    sees — the response — rather than the test re-raising the exception.
    """
    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(empty_config))
    monkeypatch.setenv("FILES_DIR", str(files_dir))
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    app = create_app()

    async def boom() -> None:
        raise RuntimeError("db password is hunter2")

    app.add_api_route("/v1/boom", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    reset_caches_for_tests()


def test_an_unexpected_exception_gets_the_openai_envelope_without_its_details(
    raising_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """OpenAI SDKs parse ``error`` out of every non-2xx; a bare 500 page breaks
    them. And the exception text is for the log, never the wire."""
    with caplog.at_level(logging.ERROR):
        r = raising_client.get("/v1/boom")

    assert r.status_code == 500
    assert r.json() == error_payload(
        message="Internal server error", type_="api_error", code="internal_error"
    )
    assert "hunter2" not in r.text
    assert any("hunter2" in (rec.exc_text or "") for rec in caplog.records)


def test_a_validation_error_names_the_offending_field(
    raising_client: TestClient,
) -> None:
    r = raising_client.post(
        "/v1/images/generations",
        headers={"Authorization": "Bearer test-bridge-api-key"},
        json={"prompt": "a cat"},  # no model
    )

    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_request"
    assert error["param"] == "model"
    assert error["message"].startswith("body.model:")
