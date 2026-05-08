"""End-to-end endpoint tests via TestClient.

Empty-providers config — covers auth, validation, parse_model_id error mapping,
and 404 paths. Backend-success paths are covered separately with respx stubs.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_auth_required(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_models_lists_empty(client: TestClient, auth_headers: dict[str, str]) -> None:
    r = client.get("/v1/models", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == {"object": "list", "data": []}


def test_images_generations_malformed_model(
    client: TestClient, auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/generations",
        headers=auth_headers,
        json={"model": "no-slash", "prompt": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"
    assert r.json()["error"]["param"] == "model"


def test_images_generations_unknown_provider(
    client: TestClient, auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/generations",
        headers=auth_headers,
        json={"model": "ghost/x", "prompt": "x"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "provider_not_found"


def test_images_generations_n_too_large_via_pydantic(
    client: TestClient, auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/generations",
        headers=auth_headers,
        json={"model": "ghost/x", "prompt": "x", "n": 10},
    )
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["type"] == "invalid_request_error"
    assert "n" in body.get("param", "")


def test_images_edits_empty_image(
    client: TestClient, auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/edits",
        headers=auth_headers,
        files={"image": ("x.png", b"", "image/png")},
        data={"model": "ghost/x", "prompt": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_videos_get_nonexistent(
    client: TestClient, auth_headers: dict[str, str],
) -> None:
    r = client.get("/v1/videos/abcdef", headers=auth_headers)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_videos_content_nonexistent(
    client: TestClient, auth_headers: dict[str, str],
) -> None:
    r = client.get("/v1/videos/abcdef/content", headers=auth_headers)
    assert r.status_code == 404


def test_files_content_nonexistent(
    client: TestClient, auth_headers: dict[str, str],
) -> None:
    r = client.get("/v1/files/abcdef/content", headers=auth_headers)
    assert r.status_code == 404
