"""OpenRouter Backend implementation.

OpenRouter is *almost* fully OpenAI-compatible: chat completions, embeddings,
and the model catalog all speak the standard wire format. The one place it
diverges is **image generation**, which OpenRouter exposes via chat
completions with a non-standard ``message.images`` array on the response
rather than via OpenAI's ``/v1/images/generations`` endpoint.

We compose the existing ``OpenAIClient`` for the spec-compliant surfaces
(chat / embedding / models) and add image-via-chat translation on top —
so a single ``[[providers]] backend = "openrouter"`` block in the config
gives clients a unified OpenAI-shaped surface: image gen routes to
``/v1/images/generations``, edits route to ``/v1/images/edits``, chat goes
to ``/v1/chat/completions``, no model catalog duplication, no surprises.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...config import OpenRouterProviderConfig
from ..base import Backend, GeneratedAsset, ModelEntry
from ..openai.client import OpenAIClient
from .client import classify_kind, extract_image_data_urls, fetch_image_bytes

log = logging.getLogger(__name__)


class OpenRouterBackend(Backend):
	def __init__(self, cfg: OpenRouterProviderConfig) -> None:
		self.cfg = cfg
		# Compose the existing passthrough client for chat / embeddings /
		# models — keeps streaming, error mapping, and SSE forwarding logic
		# in one place. The image translation wraps a separate chat call
		# through this same client.
		self._client = OpenAIClient(
			base_url=cfg.base_url,
			api_token=cfg.resolve_api_token(),
			request_timeout_seconds=cfg.request_timeout_seconds,
		)
		# Separate plain client for fetching CDN-hosted images. Most
		# OpenRouter outputs are inline data URLs (no fetch needed) but a
		# few models return hosted URLs that we'd want to GET without our
		# bridge-side Authorization header attached.
		self._download_client = httpx.AsyncClient(timeout=120.0)

	async def aclose(self) -> None:
		await self._client.aclose()
		await self._download_client.aclose()

	# --- model catalog ---------------------------------------------------

	async def list_models(self) -> list[ModelEntry]:
		raw = await self._client.list_models()
		entries: list[ModelEntry] = []
		for m in raw:
			if not isinstance(m, dict):
				continue
			model_id = m.get("id")
			if not isinstance(model_id, str) or not model_id:
				continue
			kind = classify_kind(m)
			# Skip models we can't usefully expose (audio, unknown modalities).
			if kind is None:
				continue
			# OpenRouter's catalog has a ``name`` field that's often more
			# human-readable than the slug-shaped id; surface it as
			# display_name when present.
			display = m.get("name") if isinstance(m.get("name"), str) else model_id
			# OpenRouter publishes per-model `supported_parameters` (an
			# array of OpenAI parameter names the model accepts). When
			# present we can definitively answer the tool-support
			# question; when absent we leave it None so the client falls
			# back to its per-endpoint config.
			supports_tools: bool | None = None
			params = m.get("supported_parameters")
			if isinstance(params, list):
				supports_tools = "tools" in params
			entries.append(
				ModelEntry(
					id=model_id,
					kind=kind,
					display_name=display,
					supports_tools=supports_tools,
				)
			)
		return entries

	# --- chat / embedding passthrough ------------------------------------

	async def chat_completion(
		self,
		body: dict[str, Any],
		*,
		stream: bool,
	) -> dict[str, Any] | AsyncIterator[bytes]:
		if stream:
			return await self._client.chat_completion_stream(body)
		return await self._client.chat_completion(body)

	async def create_embedding(self, body: dict[str, Any]) -> dict[str, Any]:
		return await self._client.create_embedding(body)

	# --- image generation (translated via chat completions) --------------

	async def generate_image(
		self,
		*,
		model_slug: str,
		prompt: str,
		size: str | None = None,
		n: int = 1,
	) -> list[GeneratedAsset]:
		# OpenRouter doesn't support a server-side ``n`` for image output —
		# each chat call returns one image. We loop here so the bridge's
		# ``n>1`` semantics still work.
		# ``size`` is ignored: OpenRouter's image-via-chat protocol doesn't
		# expose a size parameter; the model picks its own resolution.
		del size
		results: list[GeneratedAsset] = []
		for _ in range(n):
			results.append(await self._generate_one(model_slug, prompt, image=None))
		return results

	async def edit_image(
		self,
		*,
		model_slug: str,
		prompt: str,
		image: bytes,
		image_content_type: str,
		size: str | None = None,
		n: int = 1,
	) -> list[GeneratedAsset]:
		# Edit = generate with the input image attached to the user message as
		# a base64 data URL. OpenRouter's multimodal models accept this exact
		# shape; the chat completions API does the rest.
		del size
		b64 = base64.b64encode(image).decode("ascii")
		input_data_url = f"data:{image_content_type};base64,{b64}"
		results: list[GeneratedAsset] = []
		for _ in range(n):
			results.append(
				await self._generate_one(model_slug, prompt, image=input_data_url)
			)
		return results

	# --- internal helper -------------------------------------------------

	async def _generate_one(
		self, model_slug: str, prompt: str, *, image: str | None
	) -> GeneratedAsset:
		"""Run a single OpenRouter image generation: build the chat body,
		await the response, pull the first image out, return as a
		GeneratedAsset."""
		if image is None:
			messages: list[dict[str, Any]] = [
				{"role": "user", "content": prompt}
			]
		else:
			messages = [
				{
					"role": "user",
					"content": [
						{"type": "text", "text": prompt},
						{
							"type": "image_url",
							"image_url": {"url": image},
						},
					],
				}
			]
		body = {
			"model": model_slug,
			"messages": messages,
			# OpenRouter accepts ``modalities`` to hint that the response
			# should include image output. Image-capable models will then
			# emit the ``message.images`` array; chat-only models ignore
			# the hint (and the bridge surfaces "no image returned" as a
			# clean UpstreamError if the user picked a non-image model).
			"modalities": ["image", "text"],
		}
		response = await self._client.chat_completion(body)
		urls = extract_image_data_urls(response)
		# OpenRouter occasionally returns multiple images per response (some
		# models do n>1 internally). We pick the first one — n>1 is handled
		# by the caller invoking _generate_one multiple times. The extras
		# are dropped on the floor; in practice OpenRouter returns one.
		data, content_type = await fetch_image_bytes(urls[0], self._download_client)
		return GeneratedAsset(data=data, content_type=content_type, kind="image")
