from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 90.0
DEFAULT_OPENROUTER_APP_TITLE = "GCG Comment Generation"
DEFAULT_OPENROUTER_MAX_TOKENS = 512
DEFAULT_OPENROUTER_TEMPERATURE = 0.8
OPENROUTER_CONFIG_FILE = Path(__file__).with_name("openrouter_api_pool.json")
OPENROUTER_CONFIG_EXAMPLE_FILE = Path(__file__).with_name("openrouter_api_pool.example.json")

OPENROUTER_MODEL_ALIASES = {
    "glm": "z-ai/glm-4.5-air:free",
    "dsr1": "deepseek/deepseek-r1:free",
    "kimi": "moonshotai/kimi-k2:free",
    "llama": "meta-llama/llama-4-maverick",
    "gptoss": "openai/gpt-oss-120b:free",
    "qwen": "qwen/qwen3.6-plus-preview:free",
    "r1": "deepseek/deepseek-r1",
    "gpt54": "openai/gpt-5.4",
}

_LOCAL_GENERATORS: dict[str, Callable[[str], str]] = {}
_ACTIVE_OPENROUTER_KEY: str | None = None
_OPENROUTER_KEY_LOCK = threading.Lock()


@dataclass(slots=True)
class _AttemptFailure:
    masked_key: str
    status_code: int | None
    reason: str
    switchable: bool


@dataclass(slots=True)
class _OpenRouterError(Exception):
    status_code: int | None
    reason: str
    switchable: bool

    def __str__(self) -> str:
        if self.status_code is None:
            return self.reason
        return f"HTTP {self.status_code}: {self.reason}"


@dataclass(slots=True)
class _OpenRouterSettings:
    api_keys: list[str]
    base_url: str
    timeout_seconds: float
    app_title: str
    http_referer: str | None
    max_tokens: int
    temperature: float


def register_local_generator(platform: str, generator: Callable[[str], str]) -> None:
    _LOCAL_GENERATORS[platform] = generator


def generate_comment_via_api(prompt: str, *, platform: str,
                             model_alias: str | None = None,
                             video_record: dict | None = None) -> str:
    del video_record

    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        raise ValueError("Prompt is empty; cannot call the OpenRouter backend.")

    settings = _load_openrouter_settings()
    model_slug = _resolve_openrouter_model(model_alias)
    request_payload = _build_openrouter_payload(prompt_text, model_slug, settings)
    base_headers = _build_openrouter_headers(settings)
    endpoint = settings.base_url.rstrip("/") + "/chat/completions"

    failures: list[_AttemptFailure] = []
    attempted_indexes: set[int] = set()
    total_keys = len(settings.api_keys)

    for key_index, api_key in _iter_keys_from_active(settings.api_keys):
        if key_index in attempted_indexes:
            continue
        attempted_indexes.add(key_index)

        print(
            f"[OpenRouter] platform={platform} model={model_slug} "
            f"key={key_index + 1}/{total_keys}"
        )

        try:
            comment = _request_comment_with_same_key_retries(
                endpoint=endpoint,
                headers=_build_authorized_headers(base_headers, api_key),
                payload=request_payload,
                timeout_seconds=settings.timeout_seconds,
                platform=platform,
                model_slug=model_slug,
                key_index=key_index,
                total_keys=total_keys,
            )
            _set_active_openrouter_key(api_key)
            print(
                f"[OpenRouter] platform={platform} model={model_slug} "
                f"success key={key_index + 1}/{total_keys}"
            )
            return comment
        except _OpenRouterError as exc:
            failures.append(
                _AttemptFailure(
                    masked_key=_mask_key(api_key),
                    status_code=exc.status_code,
                    reason=exc.reason,
                    switchable=exc.switchable,
                )
            )
            if exc.switchable and len(attempted_indexes) < total_keys:
                print(
                    f"[OpenRouter] platform={platform} switching key after {exc} "
                    f"(attempted {len(attempted_indexes)}/{total_keys})"
                )
                continue
            break

    raise RuntimeError(_build_openrouter_failure_message(platform, model_slug, failures))


def generate_comment_with_backend(prompt: str, *, backend: str, platform: str,
                                  model_alias: str | None = None,
                                  video_record: dict | None = None) -> str:
    if backend == "ollama":
        generator = _LOCAL_GENERATORS.get(platform)
        if generator is None:
            raise ValueError(f"No local generator registered for platform={platform!r}")
        return generator(prompt)

    if backend == "api":
        return generate_comment_via_api(
            prompt,
            platform=platform,
            model_alias=model_alias,
            video_record=video_record,
        )

    raise ValueError(f"Unsupported generation backend: {backend!r}")


def _load_openrouter_settings() -> _OpenRouterSettings:
    repo_config = _load_repo_openrouter_config()
    api_keys = _load_openrouter_key_pool(repo_config)

    base_url = _normalize_nonempty_string(
        repo_config.get("base_url") if repo_config else None,
        env_name="OPENROUTER_BASE_URL",
        default=DEFAULT_OPENROUTER_BASE_URL,
        setting_name="base_url",
    )
    timeout_seconds = _normalize_positive_float(
        repo_config.get("timeout_seconds") if repo_config else None,
        env_name="OPENROUTER_TIMEOUT_SECONDS",
        default=DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
        setting_name="timeout_seconds",
    )
    app_title = _normalize_nonempty_string(
        repo_config.get("app_title") if repo_config else None,
        env_name="OPENROUTER_APP_TITLE",
        default=DEFAULT_OPENROUTER_APP_TITLE,
        setting_name="app_title",
    )
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip() or None
    max_tokens = _normalize_positive_int(
        repo_config.get("max_tokens") if repo_config else None,
        default=DEFAULT_OPENROUTER_MAX_TOKENS,
        setting_name="max_tokens",
    )
    temperature = _normalize_float(
        repo_config.get("temperature") if repo_config else None,
        default=DEFAULT_OPENROUTER_TEMPERATURE,
        setting_name="temperature",
    )

    return _OpenRouterSettings(
        api_keys=api_keys,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        app_title=app_title,
        http_referer=http_referer,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _load_repo_openrouter_config() -> dict | None:
    if not OPENROUTER_CONFIG_FILE.exists():
        return None

    try:
        payload = json.loads(OPENROUTER_CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{OPENROUTER_CONFIG_FILE.name} is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"{OPENROUTER_CONFIG_FILE.name} must contain a JSON object.")

    return payload


def _load_openrouter_key_pool(repo_config: dict | None) -> list[str]:
    repo_keys = None
    if repo_config is not None and "api_keys" in repo_config:
        repo_keys = repo_config["api_keys"]
        if not isinstance(repo_keys, list):
            raise RuntimeError(f"{OPENROUTER_CONFIG_FILE.name} field 'api_keys' must be a JSON array of strings.")
        invalid_item = next((item for item in repo_keys if not isinstance(item, str)), None)
        if invalid_item is not None:
            raise RuntimeError(f"{OPENROUTER_CONFIG_FILE.name} field 'api_keys' must contain strings only.")
        keys = [item.strip() for item in repo_keys if item.strip()]
    else:
        keys_json = os.getenv("OPENROUTER_API_KEYS_JSON", "").strip()
        if keys_json:
            try:
                parsed = json.loads(keys_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError("OPENROUTER_API_KEYS_JSON is not valid JSON.") from exc
            if not isinstance(parsed, list):
                raise RuntimeError("OPENROUTER_API_KEYS_JSON must be a JSON array of strings.")
            invalid_item = next((item for item in parsed if not isinstance(item, str)), None)
            if invalid_item is not None:
                raise RuntimeError("OPENROUTER_API_KEYS_JSON must contain strings only.")
            keys = [item.strip() for item in parsed if item.strip()]
        else:
            single_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            keys = [single_key] if single_key else []

    deduped_keys: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped_keys.append(key)

    if not deduped_keys:
        raise RuntimeError(
            "No OpenRouter API key configured. Add api_keys to openrouter_api_pool.json "
            f"(see {OPENROUTER_CONFIG_EXAMPLE_FILE.name}), "
            "or set OPENROUTER_API_KEYS_JSON / OPENROUTER_API_KEY."
        )

    return deduped_keys


def _resolve_openrouter_model(model_alias: str | None) -> str:
    raw_value = str(model_alias or "").strip()
    if not raw_value:
        supported = ", ".join(sorted(OPENROUTER_MODEL_ALIASES))
        raise ValueError(
            "OpenRouter model alias is required for backend='api'. "
            f"Use one of: {supported}, or pass a full model slug."
        )

    normalized = raw_value.lower()
    if "/" in raw_value:
        return raw_value
    if normalized in OPENROUTER_MODEL_ALIASES:
        return OPENROUTER_MODEL_ALIASES[normalized]

    supported = ", ".join(sorted(OPENROUTER_MODEL_ALIASES))
    raise ValueError(
        f"Unsupported OpenRouter model alias: {raw_value!r}. "
        f"Use one of: {supported}, or pass a full model slug."
    )


def _build_openrouter_payload(prompt: str, model_slug: str, settings: _OpenRouterSettings) -> dict:
    return {
        "model": model_slug,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }


def _build_openrouter_headers(settings: _OpenRouterSettings) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "X-Title": settings.app_title,
    }

    if settings.http_referer:
        headers["HTTP-Referer"] = settings.http_referer

    return headers


def _build_authorized_headers(base_headers: dict[str, str], api_key: str) -> dict[str, str]:
    headers = dict(base_headers)
    headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _normalize_nonempty_string(raw_value, *, env_name: str | None = None,
                               default: str, setting_name: str) -> str:
    if raw_value is None and env_name:
        raw_value = os.getenv(env_name, "")
    value = str(raw_value or "").strip()
    if value:
        return value
    if default:
        return default
    raise RuntimeError(f"OpenRouter setting {setting_name!r} cannot be empty.")


def _normalize_positive_float(raw_value, *, env_name: str | None = None,
                              default: float, setting_name: str) -> float:
    if raw_value is None and env_name:
        raw_value = os.getenv(env_name, "")
    if raw_value in (None, ""):
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenRouter setting {setting_name!r} must be a number.") from exc
    if value <= 0:
        raise RuntimeError(f"OpenRouter setting {setting_name!r} must be greater than 0.")
    return value


def _normalize_positive_int(raw_value, *, default: int, setting_name: str) -> int:
    if raw_value in (None, ""):
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenRouter setting {setting_name!r} must be an integer.") from exc
    if value <= 0:
        raise RuntimeError(f"OpenRouter setting {setting_name!r} must be greater than 0.")
    return value


def _normalize_float(raw_value, *, default: float, setting_name: str) -> float:
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenRouter setting {setting_name!r} must be a number.") from exc


def _iter_keys_from_active(keys: list[str]):
    with _OPENROUTER_KEY_LOCK:
        active_key = _ACTIVE_OPENROUTER_KEY

    if active_key in keys:
        start_index = keys.index(active_key)
    else:
        start_index = 0

    for offset in range(len(keys)):
        index = (start_index + offset) % len(keys)
        yield index, keys[index]


def _set_active_openrouter_key(api_key: str) -> None:
    global _ACTIVE_OPENROUTER_KEY
    with _OPENROUTER_KEY_LOCK:
        _ACTIVE_OPENROUTER_KEY = api_key


def _post_openrouter_chat(*, endpoint: str, headers: dict[str, str], payload: dict,
                          timeout_seconds: float) -> dict:
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise _classify_openrouter_http_error(exc.code, body_text) from exc
    except urllib.error.URLError as exc:
        raise _OpenRouterError(
            status_code=None,
            reason=f"Network error while calling OpenRouter: {exc.reason}",
            switchable=False,
        ) from exc
    except TimeoutError as exc:
        raise _OpenRouterError(
            status_code=None,
            reason="Timed out while calling OpenRouter.",
            switchable=False,
        ) from exc

    try:
        response_json = json.loads(body_text)
    except json.JSONDecodeError as exc:
        snippet = _truncate(body_text)
        raise _OpenRouterError(
            status_code=None,
            reason=f"OpenRouter returned invalid JSON: {snippet}",
            switchable=False,
        ) from exc

    return response_json


def _request_comment_with_same_key_retries(*, endpoint: str, headers: dict[str, str], payload: dict,
                                           timeout_seconds: float, platform: str,
                                           model_slug: str, key_index: int, total_keys: int) -> str:
    base_max_tokens = int(payload.get("max_tokens", DEFAULT_OPENROUTER_MAX_TOKENS))
    retry_token_budgets = [base_max_tokens]
    for candidate in (max(base_max_tokens * 2, 1024), max(base_max_tokens * 4, 2048)):
        if candidate not in retry_token_budgets:
            retry_token_budgets.append(candidate)

    last_exc: _OpenRouterError | None = None
    for attempt_index, max_tokens in enumerate(retry_token_budgets):
        attempt_payload = dict(payload)
        attempt_payload["max_tokens"] = max_tokens
        if attempt_index > 0:
            print(
                f"[OpenRouter] platform={platform} model={model_slug} "
                f"retrying same key={key_index + 1}/{total_keys} with max_tokens={max_tokens}"
            )
        try:
            response_json = _post_openrouter_chat(
                endpoint=endpoint,
                headers=headers,
                payload=attempt_payload,
                timeout_seconds=timeout_seconds,
            )
            return _extract_openrouter_text(response_json)
        except _OpenRouterError as exc:
            last_exc = exc
            if not _should_retry_with_more_tokens(exc):
                raise

    assert last_exc is not None
    raise last_exc


def _extract_openrouter_text(response_json: dict) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _OpenRouterError(
            status_code=None,
            reason="OpenRouter response is missing choices[0].",
            switchable=False,
        )

    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    if not isinstance(message, dict):
        raise _OpenRouterError(
            status_code=None,
            reason="OpenRouter response is missing message content.",
            switchable=False,
        )

    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        text = "".join(parts).strip()
    else:
        text = ""

    if not text:
        finish_reason = first_choice.get("finish_reason")
        reason = "OpenRouter response contained empty message content."
        if finish_reason == "length":
            reason += " The model exhausted max_tokens before producing final content."
        raise _OpenRouterError(
            status_code=None,
            reason=reason,
            switchable=False,
        )

    return text


def _classify_openrouter_http_error(status_code: int, body_text: str) -> _OpenRouterError:
    normalized = body_text.lower()
    message = _extract_error_message(body_text)

    if status_code in (401, 403) and _contains_any(
        normalized,
        (
            "invalid api key",
            "incorrect api key",
            "unauthorized",
            "forbidden",
            "permission",
            "disabled",
            "deactivated",
            "revoked",
        ),
    ):
        return _OpenRouterError(status_code, message or "API key rejected by OpenRouter.", True)

    if status_code == 402 or _contains_any(
        normalized,
        (
            "insufficient credits",
            "insufficient balance",
            "quota",
            "rate limit quota",
            "payment required",
            "credits",
            "balance",
        ),
    ):
        return _OpenRouterError(status_code, message or "OpenRouter credits/quota exhausted.", True)

    if status_code == 429:
        return _OpenRouterError(status_code, message or "OpenRouter rate limit exceeded.", True)

    if 500 <= status_code <= 599:
        return _OpenRouterError(status_code, message or "OpenRouter server error.", True)

    if status_code == 400:
        return _OpenRouterError(status_code, message or "Invalid OpenRouter request.", False)

    if _contains_any(normalized, ("model not found", "unknown model", "no endpoints found")):
        return _OpenRouterError(status_code, message or "Requested OpenRouter model was not found.", False)

    return _OpenRouterError(
        status_code,
        message or f"OpenRouter request failed. Response: {_truncate(body_text)}",
        False,
    )


def _extract_error_message(body_text: str) -> str:
    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError:
        return _truncate(body_text)

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("metadata")
            if message:
                return _truncate(str(message))
        if payload.get("message"):
            return _truncate(str(payload["message"]))

    return _truncate(body_text)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _should_retry_with_more_tokens(exc: _OpenRouterError) -> bool:
    return exc.status_code is None and "exhausted max_tokens" in exc.reason


def _build_openrouter_failure_message(platform: str, model_slug: str,
                                      failures: list[_AttemptFailure]) -> str:
    if not failures:
        return f"OpenRouter request failed for platform={platform} model={model_slug}."

    parts = [
        f"key={failure.masked_key} status={failure.status_code or 'n/a'} reason={failure.reason}"
        for failure in failures
    ]
    return (
        f"OpenRouter request failed for platform={platform} model={model_slug}. "
        f"Attempt summary: {'; '.join(parts)}"
    )


def _mask_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return api_key[:2] + "***"
    return f"{api_key[:6]}...{api_key[-4:]}"


def _truncate(text: str, limit: int = 240) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."
