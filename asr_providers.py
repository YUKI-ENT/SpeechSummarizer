"""ASR provider adapters used by SpeechSummarizer.

Providers return only metrics actually supplied by their backend.  In
particular, HTTP API adapters never manufacture Whisper confidence metrics.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx


class ASRProviderError(RuntimeError):
    """A safe, user-displayable provider failure."""

    def __init__(self, message: str, *, code: str = "provider_error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass
class ASRResult:
    text: str
    provider: str
    engine: str
    model: str | None
    language: str | None
    provider_metrics: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float | None] = field(default_factory=dict)
    request_id: str | None = None
    backend: str | None = None
    model_id: str | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)


class WhisperASRProvider:
    name = "whisper"

    def __init__(self, model_loader: Callable[[str], Any]):
        self._model_loader = model_loader

    async def transcribe(self, wav_path: str | Path, config: dict[str, Any]) -> ASRResult:
        return await asyncio.to_thread(self._transcribe_sync, str(wav_path), config)

    def _transcribe_sync(self, wav_path: str, config: dict[str, Any]) -> ASRResult:
        model = self._model_loader(config["model_path"])
        segments, _info = model.transcribe(
            wav_path,
            language=config["language"],
            vad_filter=False,
            beam_size=config["beam_size"],
            temperature=config["temperature"],
            condition_on_previous_text=config["condition_on_previous_text"],
            initial_prompt=config["prompt"],
        )

        texts: list[str] = []
        metric_values: dict[str, list[float]] = {
            "avg_logprob": [],
            "no_speech_prob": [],
            "compression_ratio": [],
        }
        for segment in segments:
            text = (getattr(segment, "text", "") or "").strip()
            if text:
                texts.append(text)
            for metric_name in metric_values:
                value = getattr(segment, metric_name, None)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metric_values[metric_name].append(float(value))

        metrics = {
            name: round(sum(values) / len(values), 3)
            for name, values in metric_values.items()
            if values
        }
        return ASRResult(
            text="\n".join(texts).strip(),
            provider=self.name,
            engine="faster-whisper",
            model=config.get("model_name"),
            language=config.get("language"),
            provider_metrics=metrics,
        )


class HTTPASRProvider:
    """Adapter for the schema-v1 localhost API shared by Qwen and VibeVoice."""

    name = "http-asr"
    display_name = "ASR"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_sec: float = 35.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._client = client
        self._ready_checked = False

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if self._client is not None:
            return await self._client.request(method, path, **kwargs)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_sec) as client:
            return await client.request(method, path, **kwargs)

    async def ready(self) -> dict[str, Any]:
        try:
            response = await self._request("GET", "/ready", timeout=min(self.timeout_sec, 2.0))
        except httpx.HTTPError as exc:
            raise ASRProviderError(
                f"{self.display_name} APIへ接続できません。",
                code="not_ready",
                retryable=True,
            ) from exc
        payload = self._decode_payload(response)
        if response.status_code != 200 or payload.get("status") != "ready":
            raise self._api_error(response.status_code, payload)
        if payload.get("schema_version") != 1:
            raise ASRProviderError(f"{self.display_name} APIのschema_versionに対応していません。")
        self._ready_checked = True
        return payload

    async def transcribe(self, wav_path: str | Path, config: dict[str, Any]) -> ASRResult:
        if not self._ready_checked:
            await self.ready()
        request_id = f"ss-{uuid.uuid4()}"
        fields = {
            "request_id": request_id,
            "language": config.get("language") or "",
            "context": config.get("context") or "",
        }
        hotwords = config.get("hotwords")
        if hotwords:
            fields["hotwords"] = json.dumps(hotwords, ensure_ascii=False)
        if config.get("include_segments"):
            fields["include_segments"] = "true"
        try:
            with Path(wav_path).open("rb") as audio_file:
                response = await self._request(
                    "POST",
                    "/transcribe",
                    data=fields,
                    files={"audio": ("segment.wav", audio_file, "audio/wav")},
                )
        except (OSError, httpx.HTTPError) as exc:
            raise ASRProviderError(
                f"{self.display_name} APIへの認識要求に失敗しました。",
                code="connection_failed",
                retryable=True,
            ) from exc

        payload = self._decode_payload(response)
        if response.status_code != 200:
            raise self._api_error(response.status_code, payload)
        if payload.get("schema_version") != 1:
            raise ASRProviderError(f"{self.display_name} APIのschema_versionに対応していません。")
        if not isinstance(payload.get("text"), str):
            raise ASRProviderError(f"{self.display_name} APIの応答にtextがありません。")

        provider_metrics = payload.get("provider_metrics")
        if not isinstance(provider_metrics, dict):
            provider_metrics = {}
        timing = payload.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        segments = payload.get("segments")
        if not isinstance(segments, list):
            segments = []
        segments = [segment for segment in segments if isinstance(segment, dict)]

        return ASRResult(
            text=payload["text"].strip(),
            provider=self.name,
            engine=str(payload.get("engine") or self.name),
            model=str(payload["model"]) if payload.get("model") is not None else None,
            language=str(payload["language"]) if payload.get("language") is not None else None,
            provider_metrics=provider_metrics,
            timing=timing,
            request_id=str(payload.get("request_id") or request_id),
            backend=str(payload["backend"]) if payload.get("backend") is not None else None,
            model_id=str(payload["model_id"]) if payload.get("model_id") is not None else None,
            segments=segments,
        )

    @staticmethod
    def _decode_payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ASRProviderError("ASR APIから不正なJSON応答を受信しました。") from exc
        if not isinstance(payload, dict):
            raise ASRProviderError("ASR APIから不正な応答を受信しました。")
        return payload

    @staticmethod
    def _api_error(status_code: int, payload: dict[str, Any]) -> ASRProviderError:
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        code = str(error.get("code") or f"http_{status_code}")
        message = str(error.get("message") or "ASR APIでエラーが発生しました。")
        retryable = bool(error.get("retryable", status_code >= 500 or status_code == 429))
        return ASRProviderError(message, code=code, retryable=retryable)


class Qwen3ASRProvider(HTTPASRProvider):
    name = "qwen3-asr"
    display_name = "Qwen3-ASR"


class VibeVoiceASRProvider(HTTPASRProvider):
    name = "vibevoice-asr"
    display_name = "VibeVoice-ASR"


def create_asr_provider(config: dict[str, Any], model_loader: Callable[[str], Any]):
    provider_name = str(config.get("provider", "whisper")).strip().lower()
    if provider_name == "whisper":
        return WhisperASRProvider(model_loader)
    if provider_name in {"qwen", "qwen3-asr"}:
        qwen_config = config.get("qwen") or {}
        base_url = str(qwen_config.get("base_url", "http://127.0.0.1:8010")).strip()
        parsed_url = urlparse(base_url)
        if parsed_url.scheme != "http" or parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("asr.qwen.base_url must use localhost")
        timeout_sec = float(qwen_config.get("timeout_sec", 35.0))
        if timeout_sec <= 0:
            raise ValueError("asr.qwen.timeout_sec must be greater than zero")
        return Qwen3ASRProvider(base_url, timeout_sec=timeout_sec)
    if provider_name in {"vibevoice", "vibevoice-asr"}:
        vibevoice_config = config.get("vibevoice") or {}
        base_url = str(vibevoice_config.get("base_url", "http://127.0.0.1:8020")).strip()
        parsed_url = urlparse(base_url)
        if parsed_url.scheme != "http" or parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("asr.vibevoice.base_url must use localhost")
        timeout_sec = float(vibevoice_config.get("timeout_sec", 135.0))
        if timeout_sec <= 0:
            raise ValueError("asr.vibevoice.timeout_sec must be greater than zero")
        return VibeVoiceASRProvider(base_url, timeout_sec=timeout_sec)
    raise ValueError(f"unsupported asr.provider: {provider_name}")


def asr_model_label(provider_name: str, model_name: str) -> str:
    """Build a provider-neutral model label for display in the client."""
    provider = str(provider_name or "asr").strip().lower() or "asr"
    model = str(model_name or "").strip()
    return f"{provider}:{model}"
