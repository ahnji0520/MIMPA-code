"""OpenAI, Anthropic, and local-vLLM clients with one experiment interface."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


API_URL = "https://api.openai.com/v1/responses"


def supports_reasoning_effort(model: str) -> bool:
    """Return whether the OpenAI model accepts the Responses reasoning field."""
    name = model.lower()
    return name.startswith("gpt-5") or name.startswith(("o1", "o3", "o4"))


class ModelClient(Protocol):
    """Interface used by the dataset adapters and debate engine."""

    def call(
        self,
        *,
        model: str,
        effort: str,
        messages: list[dict[str, str]],
        purpose: str,
        max_output_tokens: int,
        schema_name: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]: ...


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _extract_output_text(response: dict[str, Any]) -> str:
    if response.get("status") == "failed":
        raise RuntimeError(f"Model request failed: {response.get('error')}")
    chunks: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise RuntimeError(f"Model refusal: {content.get('refusal')}")
            if content.get("type") == "output_text":
                chunks.append(str(content.get("text", "")))
    if chunks:
        return "".join(chunks).strip()
    text = response.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError(f"No output text; status={response.get('status')}")


class ResponsesClient:
    def __init__(self, api_key: str, audit_path: Path, timeout: float = 300, retries: int = 3) -> None:
        self.api_key = api_key
        self.audit_path = audit_path
        self.timeout = timeout
        self.retries = retries
        self._audit_lock = threading.Lock()

    def _append_audit(self, value: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_lock, self.audit_path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(value, ensure_ascii=False) + "\n")

    def call(
        self,
        *,
        model: str,
        effort: str,
        messages: list[dict[str, str]],
        purpose: str,
        max_output_tokens: int,
        schema_name: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "store": False,
            "input": messages,
            "max_output_tokens": max_output_tokens,
        }
        reasoning_sent = supports_reasoning_effort(model)
        if reasoning_sent:
            payload["reasoning"] = {"effort": effort}
        if schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name or "structured_output",
                    "strict": True,
                    "schema": schema,
                }
            }
        prompt_material = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        audit_base = {
            "time": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "model": model,
            "reasoning_effort": effort if reasoning_sent else None,
            "reasoning_parameter_sent": reasoning_sent,
            "prompt_sha256": hashlib.sha256(prompt_material.encode()).hexdigest(),
            "max_output_tokens": max_output_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                API_URL,
                data=body,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                text = _extract_output_text(parsed)
                self._append_audit({
                    **audit_base,
                    "attempt": attempt,
                    "status": "ok",
                    "response_id": parsed.get("id"),
                    "usage": parsed.get("usage"),
                })
                return text, parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {detail[:2000]}")
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504, 520}
            except Exception as exc:
                last_error = exc
                retryable = not str(exc).startswith("Model refusal:")
            self._append_audit({**audit_base, "attempt": attempt, "status": "error", "error": str(last_error)})
            if not retryable or attempt == self.retries:
                break
            time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"API call failed for {purpose}: {last_error}")

    def call_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        text, raw = self.call(**kwargs)
        try:
            return json.loads(text), raw
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON for {kwargs.get('purpose')}: {text[:1000]}") from exc

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class _AnthropicHTTPError(RuntimeError):
    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"HTTP {code}: {detail[:2000]}")
        self.code = code


def _anthropic_post(body: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """POST to Anthropic, optionally connecting through an SSH TCP forward.

    ``ANTHROPIC_CONNECT_HOST`` and ``ANTHROPIC_CONNECT_PORT`` change only the
    TCP destination. TLS SNI and the HTTP Host header remain
    ``api.anthropic.com``, so certificate verification is preserved.
    """
    connect_host = os.environ.get("ANTHROPIC_CONNECT_HOST")
    if not connect_host:
        request = urllib.request.Request(
            ANTHROPIC_API_URL, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    parsed = urllib.parse.urlsplit(ANTHROPIC_API_URL)
    api_host = parsed.hostname or "api.anthropic.com"
    api_port = parsed.port or 443
    connect_port = int(os.environ.get("ANTHROPIC_CONNECT_PORT", str(api_port)))
    connection = http.client.HTTPSConnection(api_host, api_port, timeout=timeout)
    connection._create_connection = lambda _address, timeout=timeout, source_address=None: (  # type: ignore[attr-defined]
        socket.create_connection((connect_host, connect_port), timeout, source_address)
    )
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            raise _AnthropicHTTPError(response.status, raw)
        return json.loads(raw)
    finally:
        connection.close()


def _anthropic_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Move system text to the top level and merge adjacent conversational roles."""
    system_parts = [str(item["content"]) for item in messages if item["role"] == "system"]
    conversational: list[dict[str, str]] = []
    for item in messages:
        role = item["role"]
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported Anthropic message role: {role}")
        content = str(item["content"])
        if conversational and conversational[-1]["role"] == role:
            conversational[-1]["content"] += "\n\n" + content
        else:
            conversational.append({"role": role, "content": content})
    if not conversational:
        raise ValueError("Anthropic request requires at least one user or assistant message")
    return "\n\n".join(system_parts), conversational


def _anthropic_schema(schema: Any) -> Any:
    """Remove array cardinality keywords unsupported by Claude structured output.

    Exact cardinality remains enforced by the experiment parser after generation.
    """
    if isinstance(schema, dict):
        return {
            key: _anthropic_schema(value)
            for key, value in schema.items()
            if key not in {"minItems", "maxItems", "minimum", "maximum"}
        }
    if isinstance(schema, list):
        return [_anthropic_schema(value) for value in schema]
    return schema


def _extract_anthropic_text(response: dict[str, Any]) -> str:
    chunks = [
        str(block.get("text", ""))
        for block in response.get("content", [])
        if block.get("type") == "text"
    ]
    text = "".join(chunks).strip()
    if text:
        return text
    raise RuntimeError(
        f"No text block in Anthropic response; stop_reason={response.get('stop_reason')}"
    )


class AnthropicMessagesClient:
    """Claude Messages API client implementing the shared experiment interface."""

    def __init__(self, api_key: str, audit_path: Path,
                 timeout: float = 300, retries: int = 8) -> None:
        self.api_key = api_key
        self.audit_path = audit_path
        self.timeout = timeout
        self.retries = retries
        self.max_concurrency = int(os.environ.get("ANTHROPIC_MAX_CONCURRENCY", "10"))
        self.low_effort_max_tokens = int(os.environ.get("ANTHROPIC_LOW_MAX_TOKENS", "1600"))
        self._request_semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._audit_lock = threading.Lock()

    def _append_audit(self, value: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_lock, self.audit_path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(value, ensure_ascii=False) + "\n")

    def call(self, *, model: str, effort: str,
             messages: list[dict[str, str]], purpose: str,
             max_output_tokens: int, schema_name: str | None = None,
             schema: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        del schema_name
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"Unsupported Anthropic effort: {effort}")
        system, anthropic_messages = _anthropic_messages(messages)
        output_config: dict[str, Any] = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": _anthropic_schema(schema)}
        sent_max_tokens = (
            min(max_output_tokens, self.low_effort_max_tokens)
            if effort == "low" else max_output_tokens
        )
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": sent_max_tokens,
            "messages": anthropic_messages,
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": output_config,
        }
        if system:
            payload["system"] = system
        prompt_material = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        audit_base = {
            "time": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "backend": "anthropic",
            "model": model,
            "reasoning_effort": effort,
            "thinking": "adaptive",
            "prompt_sha256": hashlib.sha256(prompt_material.encode()).hexdigest(),
            "max_output_tokens_requested": max_output_tokens,
            "max_output_tokens_sent": sent_max_tokens,
            "max_concurrency": self.max_concurrency,
            "structured_output": schema is not None,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with self._request_semaphore:
                    parsed = _anthropic_post(body, headers, self.timeout)
                text = _extract_anthropic_text(parsed)
                self._append_audit({
                    **audit_base, "attempt": attempt, "status": "ok",
                    "response_id": parsed.get("id"), "usage": parsed.get("usage"),
                    "stop_reason": parsed.get("stop_reason"),
                })
                return text, parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {detail[:2000]}")
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504, 520, 529}
            except _AnthropicHTTPError as exc:
                last_error = exc
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504, 520, 529}
            except Exception as exc:
                last_error = exc
                retryable = True
            self._append_audit({
                **audit_base, "attempt": attempt, "status": "error",
                "error": str(last_error),
            })
            if not retryable or attempt == self.retries:
                break
            time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Anthropic API call failed for {purpose}: {last_error}")

    def call_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        text, raw = self.call(**kwargs)
        try:
            return json.loads(text), raw
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON for {kwargs.get('purpose')}: {text[:1000]}"
            ) from exc


class VLLMChatClient:
    """Client for a local vLLM OpenAI-compatible Chat Completions server.

    ``effort`` is accepted for interface compatibility but is not sent to the
    local model. Qwen3.5 thinking is explicitly disabled through its official
    chat-template argument. Gemma 4 thinking remains disabled by omitting the
    ``<|think|>`` control token from the system prompt.
    """

    def __init__(
        self,
        base_url: str,
        audit_path: Path,
        *,
        api_key: str = "EMPTY",
        timeout: float = 300,
        retries: int = 3,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        root = base_url.rstrip("/")
        self.api_url = root if root.endswith("/chat/completions") else f"{root}/chat/completions"
        self.api_key = api_key
        self.audit_path = audit_path
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.seed = seed
        self._audit_lock = threading.Lock()

    def _append_audit(self, value: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_lock, self.audit_path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(value, ensure_ascii=False) + "\n")

    def call(
        self,
        *,
        model: str,
        effort: str,
        messages: list[dict[str, str]],
        purpose: str,
        max_output_tokens: int,
        schema_name: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if "qwen3.5" in model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or "structured_output",
                    "strict": True,
                    "schema": schema,
                },
            }

        prompt_material = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        audit_base = {
            "time": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "backend": "vllm",
            "model": model,
            "thinking": "disabled",
            "reasoning_effort": effort,
            "prompt_sha256": hashlib.sha256(prompt_material.encode()).hexdigest(),
            "max_output_tokens": max_output_tokens,
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                self.api_url,
                data=body,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                choices = parsed.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices in vLLM response: {str(parsed)[:1000]}")
                message = choices[0].get("message") or {}
                text = message.get("content")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"No output text in vLLM response: {str(parsed)[:1000]}")
                latency = time.perf_counter() - started
                self._append_audit({
                    **audit_base,
                    "attempt": attempt,
                    "status": "ok",
                    "response_id": parsed.get("id"),
                    "latency_seconds": latency,
                    "usage": parsed.get("usage"),
                })
                return text.strip(), parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {detail[:2000]}")
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504, 520}
            except Exception as exc:
                last_error = exc
                retryable = True
            self._append_audit({
                **audit_base,
                "attempt": attempt,
                "status": "error",
                "latency_seconds": time.perf_counter() - started,
                "error": str(last_error),
            })
            if not retryable or attempt == self.retries:
                break
            time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"vLLM call failed for {purpose}: {last_error}")

    def call_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        text, raw = self.call(**kwargs)
        try:
            return json.loads(text), raw
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON for {kwargs.get('purpose')}: {text[:1000]}") from exc


def create_client(
    *,
    backend: str,
    audit_path: Path,
    base_url: str = "http://127.0.0.1:8000/v1",
    temperature: float = 0.0,
    seed: int = 42,
) -> ModelClient:
    """Construct the requested backend without requiring cloud credentials locally."""
    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --backend openai")
        return ResponsesClient(api_key, audit_path)
    if backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for --backend anthropic")
        return AnthropicMessagesClient(api_key, audit_path)
    if backend == "vllm":
        return VLLMChatClient(
            base_url,
            audit_path,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            temperature=temperature,
            seed=seed,
        )
    raise ValueError(f"Unsupported backend: {backend}")
