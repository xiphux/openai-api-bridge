"""Dispatcher / config-loading / model-id parsing."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from openai_api_bridge.config import (
    ComfyUIProviderConfig,
    ConfigError,
    ImageRouterProviderConfig,
    VeniceProviderConfig,
    load_providers,
    parse_model_id,
)
from openai_api_bridge.dispatcher import BackendDispatcher
from openai_api_bridge.errors import InvalidRequest, ProviderNotFound

# --- parse_model_id ---------------------------------------------------------


def test_parse_model_id_valid() -> None:
    assert parse_model_id("comfyui/ltxv-t2i") == ("comfyui", "ltxv-t2i")


def test_parse_model_id_with_nested_slashes() -> None:
    # The full slug after the first slash is preserved verbatim.
    assert parse_model_id("comfyui/with/slashes") == ("comfyui", "with/slashes")


@pytest.mark.parametrize("bad", ["", "no-slash", "/no-provider", "no-slug/", "/"])
def test_parse_model_id_rejects_malformed(bad: str) -> None:
    with pytest.raises(InvalidRequest):
        parse_model_id(bad)


# --- load_providers ---------------------------------------------------------


def test_load_providers_round_trip(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""
        [defaults]
        cache_workflows = true

        [[providers]]
        id = "comfyui"
        backend = "comfyui"
        url = "http://127.0.0.1:8188"
        workflows_dir = "/tmp/workflows"

        [[providers]]
        id = "venice"
        backend = "venice"
        api_token_env = "TEST_VENICE_TOKEN"
    """))
    providers = load_providers(cfg)
    assert len(providers.providers) == 2
    by_id = {p.id: p for p in providers.providers}
    assert isinstance(by_id["comfyui"], ComfyUIProviderConfig)
    assert isinstance(by_id["venice"], VeniceProviderConfig)
    assert providers.by_id("missing") is None


def test_load_providers_rejects_duplicate_ids(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""
        [[providers]]
        id = "dup"
        backend = "comfyui"
        url = "http://x"
        workflows_dir = "/tmp/a"

        [[providers]]
        id = "dup"
        backend = "venice"
        api_token_env = "X"
    """))
    with pytest.raises(ConfigError, match="Duplicate"):
        load_providers(cfg)


def test_load_providers_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_providers(tmp_path / "nope.toml")


def test_venice_resolve_api_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_VENICE", "v-secret")
    cfg = VeniceProviderConfig(
        backend="venice", id="v", api_token_env="MY_VENICE"
    )
    assert cfg.resolve_api_token() == "v-secret"


def test_venice_resolve_api_token_raises_when_unset() -> None:
    if "MISSING_VAR_FOR_TESTING" in os.environ:
        del os.environ["MISSING_VAR_FOR_TESTING"]
    cfg = VeniceProviderConfig(
        backend="venice", id="v", api_token_env="MISSING_VAR_FOR_TESTING"
    )
    with pytest.raises(ConfigError, match="not set"):
        cfg.resolve_api_token()


def test_imagerouter_provider_loads_from_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(textwrap.dedent("""
        [[providers]]
        id = "ir"
        backend = "imagerouter"
        api_token_env = "TEST_IR_TOKEN"
    """))
    providers = load_providers(cfg)
    by_id = {p.id: p for p in providers.providers}
    assert isinstance(by_id["ir"], ImageRouterProviderConfig)
    # base_url is optional with a default — confirm we get the expected one.
    assert by_id["ir"].base_url == "https://api.imagerouter.io"


def test_imagerouter_resolve_api_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_IR", "ir-secret")
    cfg = ImageRouterProviderConfig(
        backend="imagerouter", id="ir", api_token_env="MY_IR"
    )
    assert cfg.resolve_api_token() == "ir-secret"


def test_imagerouter_resolve_api_token_raises_when_unset() -> None:
    if "MISSING_IR_VAR" in os.environ:
        del os.environ["MISSING_IR_VAR"]
    cfg = ImageRouterProviderConfig(
        backend="imagerouter", id="ir", api_token_env="MISSING_IR_VAR"
    )
    with pytest.raises(ConfigError, match="not set"):
        cfg.resolve_api_token()


# --- BackendDispatcher ------------------------------------------------------


async def test_dispatcher_for_provider_unknown(tmp_path: Path) -> None:
    from openai_api_bridge.config import ProvidersFile

    dispatcher = BackendDispatcher(ProvidersFile(providers=[]))
    with pytest.raises(ProviderNotFound):
        dispatcher.for_provider("missing")
    await dispatcher.aclose()


async def test_dispatcher_routes_to_comfyui_backend(tmp_path: Path) -> None:
    from openai_api_bridge.backends.comfyui.adapter import ComfyUIBackend
    from openai_api_bridge.config import ProvidersFile

    cfg = ComfyUIProviderConfig(
        backend="comfyui",
        id="comfy-a",
        url="http://127.0.0.1:8188",
        workflows_dir=tmp_path,
    )
    dispatcher = BackendDispatcher(ProvidersFile(providers=[cfg]))
    backend = dispatcher.for_provider("comfy-a")
    assert isinstance(backend, ComfyUIBackend)
    await dispatcher.aclose()
