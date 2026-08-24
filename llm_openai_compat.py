from __future__ import annotations

from typing import Any, Mapping

import requests


def build_openai_base_url(server: str, port: int, *, use_https: bool = False) -> str:
    host = (server or "").strip().strip("/") or "127.0.0.1"
    scheme = "https" if use_https else "http"
    return f"{scheme}://{host}:{int(port)}/v1"


def normalize_openai_base_url(base_url: str) -> str:
    """Return an OpenAI-compatible API root ending in /v1."""
    value = (base_url or "").strip().rstrip("/")
    if not value:
        value = "http://127.0.0.1:1234"
    if not value.endswith("/v1"):
        value += "/v1"
    return value


def _headers(api_key: str = "") -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def list_openai_models(
    base_url: str,
    *,
    api_key: str = "",
    timeout_sec: float = 120.0,
) -> list[str]:
    url = normalize_openai_base_url(base_url) + "/models"
    response = requests.get(url, headers=_headers(api_key), timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()

    data = payload.get("data") or []
    model_ids = [str(item.get("id") or "").strip() for item in data if isinstance(item, Mapping)]
    return sorted(model_id for model_id in model_ids if model_id)


def openai_chat_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_sec: float,
    temperature: float,
    top_p: float,
    system_prompt: str = "",
    json_response: bool = False,
) -> tuple[str, dict[str, Any]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
    }
    if json_response:
        body["response_format"] = {"type": "json_object"}

    url = normalize_openai_base_url(base_url) + "/chat/completions"
    response = requests.post(url, headers=_headers(api_key), json=body, timeout=timeout_sec)
    response.raise_for_status()
    payload = response.json()

    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], Mapping):
        raise ValueError("OpenAI-compatible response has no choices")
    message = choices[0].get("message") or {}
    if not isinstance(message, Mapping):
        raise ValueError("OpenAI-compatible response has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content, payload
    # A few compatible servers use content-part arrays.
    if isinstance(content, list):
        text = "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") in (None, "text")
        )
        return text, payload
    return "", payload
