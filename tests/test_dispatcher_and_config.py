"""Dispatcher / config-loading / model-id parsing."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from openai_api_bridge.config import (
    BridgeSettings,
    ComfyUIProviderConfig,
    ConfigError,
    ImageRouterProviderConfig,
    VeniceProviderConfig,
    load_providers,
    parse_model_id,
)
from openai_api_bridge.dispatcher import BackendDispatcher
from openai_api_bridge.errors import InvalidRequest, ProviderNotFound

# --- BRIDGE_API_KEY validation ----------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",  # `BRIDGE_API_KEY=` — satisfies Field(...), and compare_digest(b"", b"") is True
        "   ",
        " " * 32,  # whitespace can't pad its way over the floor
        "short",
        "replace-me-with-a-strong-random-token",  # the .env.example placeholder
    ],
)
def test_bridge_settings_rejects_unusable_api_key(bad: str) -> None:
    """A key that authenticates everyone must abort startup, not serve.

    An empty BRIDGE_API_KEY used to make `Authorization: Bearer ` a valid
    credential for any caller that could reach the port, with nothing in the
    logs to say so.
    """
    with pytest.raises(ValidationError):
        BridgeSettings(BRIDGE_API_KEY=bad)  # type: ignore[call-arg]


def test_bridge_settings_accepts_a_real_key() -> None:
    settings = BridgeSettings(BRIDGE_API_KEY="0123456789abcdef")  # type: ignore[call-arg]
    assert settings.api_key == "0123456789abcdef"


def test_bridge_settings_preserves_key_verbatim() -> None:
    """Validation measures the stripped value; it must not *store* it stripped.

    Trimming would change what an existing, working credential compares
    against and lock its holder out on upgrade.

    Leading whitespace, specifically — this once used a value padded on both
    sides, which encoded the wrong belief. A *trailing*-padded key cannot
    survive the wire at all and is now rejected outright; see
    ``test_bridge_settings_rejects_a_trailing_whitespace_key``.
    """
    settings = BridgeSettings(BRIDGE_API_KEY="  0123456789abcdef")  # type: ignore[call-arg]
    assert settings.api_key == "  0123456789abcdef"


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
    cfg.write_text(
        textwrap.dedent("""
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
    """)
    )
    providers = load_providers(cfg)
    assert len(providers.providers) == 2
    by_id = {p.id: p for p in providers.providers}
    assert isinstance(by_id["comfyui"], ComfyUIProviderConfig)
    assert isinstance(by_id["venice"], VeniceProviderConfig)
    assert providers.by_id("missing") is None


def test_load_providers_rejects_duplicate_ids(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        textwrap.dedent("""
        [[providers]]
        id = "dup"
        backend = "comfyui"
        url = "http://x"
        workflows_dir = "/tmp/a"

        [[providers]]
        id = "dup"
        backend = "venice"
        api_token_env = "X"
    """)
    )
    with pytest.raises(ConfigError, match="Duplicate"):
        load_providers(cfg)


def test_load_providers_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_providers(tmp_path / "nope.toml")


def test_venice_resolve_api_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_VENICE", "v-secret")
    cfg = VeniceProviderConfig(backend="venice", id="v", api_token_env="MY_VENICE")
    assert cfg.resolve_api_token() == "v-secret"


def test_venice_resolve_api_token_raises_when_unset() -> None:
    if "MISSING_VAR_FOR_TESTING" in os.environ:
        del os.environ["MISSING_VAR_FOR_TESTING"]
    cfg = VeniceProviderConfig(backend="venice", id="v", api_token_env="MISSING_VAR_FOR_TESTING")
    with pytest.raises(ConfigError, match="not set"):
        cfg.resolve_api_token()


def test_imagerouter_provider_loads_from_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        textwrap.dedent("""
        [[providers]]
        id = "ir"
        backend = "imagerouter"
        api_token_env = "TEST_IR_TOKEN"
    """)
    )
    providers = load_providers(cfg)
    by_id = {p.id: p for p in providers.providers}
    assert isinstance(by_id["ir"], ImageRouterProviderConfig)
    # base_url is optional with a default — confirm we get the expected one.
    assert by_id["ir"].base_url == "https://api.imagerouter.io"


def test_imagerouter_resolve_api_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_IR", "ir-secret")
    cfg = ImageRouterProviderConfig(backend="imagerouter", id="ir", api_token_env="MY_IR")
    assert cfg.resolve_api_token() == "ir-secret"


def test_imagerouter_resolve_api_token_raises_when_unset() -> None:
    if "MISSING_IR_VAR" in os.environ:
        del os.environ["MISSING_IR_VAR"]
    cfg = ImageRouterProviderConfig(backend="imagerouter", id="ir", api_token_env="MISSING_IR_VAR")
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


@pytest.mark.parametrize("bad", ["0123456789abcdef ", "0123456789abcdef\t", "0123456789abcdef\n"])
def test_bridge_settings_rejects_a_trailing_whitespace_key(bad: str) -> None:
    """A key HTTP will not deliver is a key that authenticates nobody.

    Trailing optional whitespace is stripped from a header value on the wire,
    so such a key can never equal what a client presents: the bridge would boot
    reporting success and then 401 every request, with nothing naming the
    cause.
    """
    with pytest.raises(ValidationError, match="trailing whitespace"):
        BridgeSettings(BRIDGE_API_KEY=bad)  # type: ignore[call-arg]


def test_bridge_settings_allows_a_leading_whitespace_key() -> None:
    """Leading whitespace survives the wire, so such a key genuinely works.

    ``Authorization: Bearer  secret`` presents as ``" secret"`` after the
    prefix is removed. Rejecting it would break a working deployment.
    """
    settings = BridgeSettings(BRIDGE_API_KEY=" 0123456789abcdef")  # type: ignore[call-arg]
    assert settings.api_key == " 0123456789abcdef"


@pytest.mark.parametrize("value", [-1, -100])
def test_bridge_settings_rejects_a_negative_request_cap(value: int) -> None:
    """Only 0 is documented to disable the cap; a negative would do it silently."""
    with pytest.raises(ValidationError):
        BridgeSettings(BRIDGE_API_KEY="0123456789abcdef", BRIDGE_MAX_REQUEST_MB=value)  # type: ignore[call-arg]


def test_provider_configs_reject_a_negative_asset_cap() -> None:
    from openai_api_bridge.config import FalProviderConfig

    with pytest.raises(ValidationError):
        FalProviderConfig(backend="fal", id="f", api_token_env="X", max_asset_mb=-1)
