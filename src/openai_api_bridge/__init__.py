"""OpenAI-API-compatible bridge for ComfyUI and Venice."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("openai-api-bridge")
except PackageNotFoundError:
    # Package not installed (e.g. running tests against an editable checkout
    # before `uv sync` has registered the dist-info). Surface a stub so
    # `from openai_api_bridge import __version__` never crashes.
    __version__ = "0.0.0+unknown"
