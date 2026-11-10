import base64
import hashlib
import inspect
import os
import time
from collections import defaultdict
from typing import Any, Optional

import requests
from django.conf import settings
from openai import OpenAI


SUPPORTED_LLM_PROVIDERS = {"anthropic", "gemini", "groq", "openai"}
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-1.5-flash",
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
}
GROQ_BASE = "https://api.groq.com/openai/v1"
CHAT_COMPLETIONS_ENDPOINT = f"{GROQ_BASE}/chat/completions"
LLM_REQUEST_TIMEOUT_SECONDS = 60.0
RATE_LIMIT_RETRY_DELAYS_SECONDS = (1.5, 3.0)
DEBUG_LLM = False
LLM_USAGE_SESSION_KEY = "_llm_usage_tracker"
_PROCESS_LLM_USAGE_TRACKER = {
    "active": False,
    "stage": "",
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "by_provider": {},
}
_LLM_DEBUG_STATE = {
    "call_counter": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    # Backward-compatible alias for any stale code path still reading this key.
    "total_combined_tokens": 0,
    "prompt_hash_counts": defaultdict(int),
    "prompt_hash_preview": {},
    "largest_prompt": {
        "hash": None,
        "length": 0,
        "tokens": 0,
        "preview": "",
        "function": "",
        "file": "",
    },
}


def _reset_debug_llm_state() -> None:
    _LLM_DEBUG_STATE["call_counter"] = 0
    _LLM_DEBUG_STATE["total_prompt_tokens"] = 0
    _LLM_DEBUG_STATE["total_completion_tokens"] = 0
    _LLM_DEBUG_STATE["total_combined_tokens"] = 0
    _LLM_DEBUG_STATE["prompt_hash_counts"] = defaultdict(int)
    _LLM_DEBUG_STATE["prompt_hash_preview"] = {}
    _LLM_DEBUG_STATE["largest_prompt"] = {
        "hash": None,
        "length": 0,
        "tokens": 0,
        "preview": "",
        "function": "",
        "file": "",
    }


def _new_llm_usage_tracker(stage: Optional[str] = None) -> dict[str, Any]:
    return {
        "active": True,
        "stage": str(stage or "").strip(),
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_provider": {},
    }


def _coerce_token_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return max(int(value), 0)
    except Exception:
        return 0


def _copy_llm_usage_tracker(tracker: Optional[dict[str, Any]]) -> dict[str, Any]:
    source = tracker or {}
    return {
        "active": bool(source.get("active", False)),
        "stage": str(source.get("stage", "") or ""),
        "calls": _coerce_token_int(source.get("calls")),
        "prompt_tokens": _coerce_token_int(source.get("prompt_tokens")),
        "completion_tokens": _coerce_token_int(source.get("completion_tokens")),
        "total_tokens": _coerce_token_int(source.get("total_tokens")),
        "by_provider": {
            str(provider): {
                "calls": _coerce_token_int(stats.get("calls")),
                "prompt_tokens": _coerce_token_int(stats.get("prompt_tokens")),
                "completion_tokens": _coerce_token_int(stats.get("completion_tokens")),
                "total_tokens": _coerce_token_int(stats.get("total_tokens")),
            }
            for provider, stats in dict(source.get("by_provider") or {}).items()
        },
    }


def _get_llm_usage_tracker(request=None) -> dict[str, Any]:
    if request is not None:
        try:
            tracker = request.session.get(LLM_USAGE_SESSION_KEY)
            if isinstance(tracker, dict):
                return _copy_llm_usage_tracker(tracker)
        except Exception:
            pass
    return _copy_llm_usage_tracker(None)


def _set_llm_usage_tracker(tracker: dict[str, Any], request=None) -> dict[str, Any]:
    tracker_copy = _copy_llm_usage_tracker(tracker)
    if request is not None:
        try:
            request.session[LLM_USAGE_SESSION_KEY] = tracker_copy
            request.session.modified = True
        except Exception:
            pass
    return tracker_copy


def reset_llm_usage_tracking(request=None, stage: Optional[str] = None) -> dict[str, Any]:
    _reset_debug_llm_state()
    return _set_llm_usage_tracker(_new_llm_usage_tracker(stage=stage), request=request)


def ensure_llm_usage_tracking(request=None, stage: Optional[str] = None) -> dict[str, Any]:
    tracker = _get_llm_usage_tracker(request=request)
    if not tracker.get("active"):
        return reset_llm_usage_tracking(request=request, stage=stage)
    if stage and not tracker.get("stage"):
        tracker["stage"] = str(stage).strip()
        return _set_llm_usage_tracker(tracker, request=request)
    return tracker


def finalize_llm_usage_tracking(request=None) -> dict[str, Any]:
    tracker = _get_llm_usage_tracker(request=request)
    tracker["active"] = False
    return _set_llm_usage_tracker(tracker, request=request)


def get_llm_usage_tracking(request=None) -> dict[str, Any]:
    return _get_llm_usage_tracker(request=request)


def print_llm_usage_summary(request=None, prefix: str = "[LLM TOKENS]") -> dict[str, Any]:
    tracker = get_llm_usage_tracking(request=request)
    total_calls = _coerce_token_int(tracker.get("calls"))
    prompt_tokens = _coerce_token_int(tracker.get("prompt_tokens"))
    completion_tokens = _coerce_token_int(tracker.get("completion_tokens"))
    total_tokens = prompt_tokens + completion_tokens
    average_tokens = round(total_tokens / total_calls, 2) if total_calls else 0

    tracker["total_tokens"] = total_tokens
    _set_llm_usage_tracker(tracker, request=request)

    print("----------------------------------------")
    print("LLM USAGE SUMMARY")
    print("----------------------------------------")
    print(f"Total Calls: {total_calls}")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Average Tokens per Call: {average_tokens}")
    print("----------------------------------------")
    return tracker


def _extract_token_usage(provider: str, response_or_payload: Any) -> dict[str, int]:
    provider = resolve_llm_provider(provider=provider)
    prompt_tokens = 0
    completion_tokens = 0

    if provider == "anthropic":
        usage = getattr(response_or_payload, "usage", None)
        prompt_tokens = sum(
            _coerce_token_int(getattr(usage, attr, 0))
            for attr in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        completion_tokens = _coerce_token_int(getattr(usage, "output_tokens", 0))
    elif provider == "gemini":
        usage = getattr(response_or_payload, "usage_metadata", None)
        prompt_tokens = _coerce_token_int(
            getattr(usage, "prompt_token_count", None) or getattr(usage, "input_token_count", 0)
        )
        completion_tokens = _coerce_token_int(
            getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", 0)
        )
    elif provider == "openai":
        usage = getattr(response_or_payload, "usage", None)
        prompt_tokens = _coerce_token_int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = _coerce_token_int(getattr(usage, "completion_tokens", 0))
    elif provider == "groq":
        usage = {}
        if isinstance(response_or_payload, dict):
            usage = response_or_payload.get("usage") or {}
        prompt_tokens = _coerce_token_int(usage.get("prompt_tokens"))
        completion_tokens = _coerce_token_int(usage.get("completion_tokens"))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def record_llm_usage(request=None, provider: Optional[str] = None, usage: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if request is None:
        print("WARNING: LLM tracking called without request")
    resolved_provider = resolve_llm_provider(provider=provider)
    tracker = ensure_llm_usage_tracking(request=request)
    prompt_tokens = _coerce_token_int((usage or {}).get("prompt_tokens"))
    completion_tokens = _coerce_token_int((usage or {}).get("completion_tokens"))

    if prompt_tokens == 0 and completion_tokens > 0:
        print("WARNING: Missing prompt tokens for this call")

    tracker["calls"] = _coerce_token_int(tracker.get("calls")) + 1
    tracker["prompt_tokens"] = _coerce_token_int(tracker.get("prompt_tokens")) + prompt_tokens
    tracker["completion_tokens"] = _coerce_token_int(tracker.get("completion_tokens")) + completion_tokens
    tracker["total_tokens"] = (
        _coerce_token_int(tracker.get("prompt_tokens")) +
        _coerce_token_int(tracker.get("completion_tokens"))
    )

    by_provider = dict(tracker.get("by_provider") or {})
    provider_stats = dict(by_provider.get(resolved_provider) or {})
    provider_stats["calls"] = _coerce_token_int(provider_stats.get("calls")) + 1
    provider_stats["prompt_tokens"] = _coerce_token_int(provider_stats.get("prompt_tokens")) + prompt_tokens
    provider_stats["completion_tokens"] = _coerce_token_int(provider_stats.get("completion_tokens")) + completion_tokens
    provider_stats["total_tokens"] = (
        _coerce_token_int(provider_stats.get("prompt_tokens")) +
        _coerce_token_int(provider_stats.get("completion_tokens"))
    )
    by_provider[resolved_provider] = provider_stats
    tracker["by_provider"] = by_provider

    return _set_llm_usage_tracker(tracker, request=request)


class LLMProviderError(Exception):
    def __init__(self, message: str, provider: Optional[str] = None, credit_exhausted: bool = False):
        super().__init__(message)
        self.provider = provider
        self.credit_exhausted = credit_exhausted


def is_credit_exhausted_error(error_or_text: Any) -> bool:
    message = str(error_or_text or "").strip().lower()
    if not message:
        return False

    markers = [
        "out of credits",
        "credit exhausted",
        "credits exhausted",
        "credit balance",
        "insufficient credits",
        "insufficient balance",
        "insufficient_quota",
        "quota exceeded",
        "quota has been exceeded",
        "resource_exhausted",
        "billing hard limit",
        "exceeded your current quota",
    ]
    return any(marker in message for marker in markers)


def is_rate_limit_error(error_or_text: Any) -> bool:
    message = str(error_or_text or "").strip().lower()
    if not message:
        return False

    markers = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "requests per min",
        "requests per minute",
        "tokens per min",
        "tokens per minute",
        "try again in",
        "please try again later",
        "429",
    ]
    return any(marker in message for marker in markers)


def _provider_display_name(provider: Optional[str] = None) -> str:
    normalized = _normalize_provider_name(provider) or str(provider or "selected").strip().lower()
    labels = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "groq": "Groq",
    }
    return labels.get(normalized, str(provider or "selected").strip().title())


def build_credit_exhausted_message(provider: Optional[str] = None) -> str:
    provider_name = _provider_display_name(provider)
    return f"{provider_name} credits are over. Please recharge or switch the LLM provider."


def build_rate_limit_message(provider: Optional[str] = None) -> str:
    provider_name = _provider_display_name(provider)
    return f"{provider_name} rate limit was reached. Please retry in a moment or switch the LLM provider."


def build_llm_failure_chat_message(error: Any, provider: Optional[str] = None) -> str:
    if isinstance(error, LLMProviderError) and error.credit_exhausted:
        return str(error)

    provider_name = _provider_display_name(
        getattr(error, "provider", None) or provider
    )
    detail = str(error or "").strip()
    if not detail:
        detail = "The request could not reach the LLM service."

    return (
        f"{provider_name} LLM request failed: {detail}. "
        "Please check your internet/API connection or retry in a moment."
    )


def _normalize_provider_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    provider = str(value).strip().lower()
    if not provider:
        return None
    if provider in SUPPORTED_LLM_PROVIDERS:
        return provider
    return None


def resolve_llm_provider(request=None, provider: Optional[str] = None, default: Optional[str] = None) -> str:
    session_provider = None
    if request is not None:
        try:
            session_provider = request.session.get("llm_provider")
        except Exception:
            session_provider = None

    candidates = [
        provider,
        session_provider,
        getattr(settings, "LLM_PROVIDER", None),
        os.environ.get("LLM_PROVIDER"),
        default,
        DEFAULT_LLM_PROVIDER,
    ]

    for candidate in candidates:
        normalized = _normalize_provider_name(candidate)
        if normalized:
            return normalized

    return DEFAULT_LLM_PROVIDER


def resolve_llm_model(provider: str, model: Optional[str] = None) -> str:
    provider = resolve_llm_provider(provider=provider)
    env_key = f"{provider.upper()}_MODEL"
    return (
        str(model or "").strip()
        or str(getattr(settings, env_key, "") or "").strip()
        or str(os.environ.get(env_key, "") or "").strip()
        or DEFAULT_LLM_MODELS[provider]
    )


def persist_selected_llm_provider(request=None, provider: Optional[str] = None, default: Optional[str] = None) -> str:
    resolved_provider = resolve_llm_provider(request=request, provider=provider, default=default)
    if request is not None:
        try:
            request.session["llm_provider"] = resolved_provider
            request.session.modified = True
        except Exception:
            pass
    return resolved_provider


def _coerce_messages(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, list):
        return messages
    raise ValueError("messages must be a string or list of chat messages")


def _flatten_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system_parts = []
    user_parts = []

    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = message.get("content", "")
        if isinstance(content, list):
            text_chunks = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_chunks.append(str(item.get("text", "")))
                    elif item.get("type") == "image_url":
                        text_chunks.append("[Image attached]")
                else:
                    text_chunks.append(str(item))
            content_text = "\n".join(chunk for chunk in text_chunks if chunk)
        else:
            content_text = str(content or "")

        if role == "system":
            system_parts.append(content_text)
        else:
            user_parts.append(content_text)

    return "\n\n".join(part for part in system_parts if part), "\n\n".join(part for part in user_parts if part)


def _encode_image_for_llm(image_path: Optional[str]):
    if not image_path:
        return None

    try:
        normalized_path = os.path.normpath(image_path)
        if not os.path.exists(normalized_path):
            return None

        extension = os.path.splitext(normalized_path)[1].lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(extension)
        if not mime_type:
            return None

        with open(normalized_path, "rb") as image_file:
            return {
                "mime_type": mime_type,
                "data": base64.b64encode(image_file.read()).decode("utf-8"),
            }
    except Exception:
        return None


def _resolve_provider_api_key(provider: str) -> str:
    provider = resolve_llm_provider(provider=provider)

    if provider == "anthropic":
        return (
            getattr(settings, "ANTHROPIC_API_KEY", "")
            or getattr(settings, "CLAUDE_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("CLAUDE_API_KEY", "")
        )

    if provider == "gemini":
        return (
            getattr(settings, "GOOGLE_API_KEY", "")
            or getattr(settings, "GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", "")
        )

    if provider == "openai":
        return (
            getattr(settings, "OPENAI_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )

    if provider == "groq":
        return (
            getattr(settings, "GROQ_API_KEY", "")
            or os.environ.get("GROQ_API_KEY", "")
        )

    return ""


def _truncate_debug_preview(text: str, limit: int = 500) -> str:
    cleaned = str(text or "")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "...[truncated]"


def _safe_usage_value(usage: Optional[dict[str, Any]], key: str) -> Any:
    if not isinstance(usage, dict):
        return 0
    return usage.get(key, 0)


def _extract_debug_call_context() -> dict[str, Any]:
    stack = inspect.stack()
    caller = {"function": "<unknown>", "file": "<unknown>", "stack_lines": []}
    stack_lines = []

    for frame_info in stack[2:]:
        file_name = os.path.basename(frame_info.filename)
        if file_name == "llm_provider.py":
            continue
        function_name = frame_info.function
        stack_lines.append(f"- {file_name} -> {function_name}()")
        if caller["function"] == "<unknown>":
            caller["function"] = function_name
            caller["file"] = file_name

    caller["stack_lines"] = stack_lines[:8]
    return caller


def _debug_llm_start(
    resolved_provider: str,
    resolved_model: str,
    system_prompt: str,
    user_prompt: str,
    request=None,
) -> dict[str, Any]:
    combined_prompt = "\n\n".join(part for part in [system_prompt, user_prompt] if part)
    prompt_preview = _truncate_debug_preview(combined_prompt, limit=500)
    prompt_length = len(combined_prompt)
    prompt_tokens = len(combined_prompt.split())
    prompt_hash = hashlib.sha256(combined_prompt.encode("utf-8")).hexdigest()[:16]
    context = _extract_debug_call_context()

    _LLM_DEBUG_STATE["call_counter"] += 1
    call_number = _LLM_DEBUG_STATE["call_counter"]
    _LLM_DEBUG_STATE["total_prompt_tokens"] += prompt_tokens
    _LLM_DEBUG_STATE["prompt_hash_counts"][prompt_hash] += 1
    _LLM_DEBUG_STATE["prompt_hash_preview"][prompt_hash] = prompt_preview

    if prompt_tokens > int(_LLM_DEBUG_STATE["largest_prompt"]["tokens"] or 0):
        _LLM_DEBUG_STATE["largest_prompt"] = {
            "hash": prompt_hash,
            "length": prompt_length,
            "tokens": prompt_tokens,
            "preview": prompt_preview,
            "function": context["function"],
            "file": context["file"],
        }

    print("----------------------------------------")
    print(f"LLM CALL #{call_number}")
    print(f"FUNCTION: {context['function']}")
    print(f"FILE: {context['file']}")
    print("----------------------------------------")
    print(f"PROVIDER: {resolved_provider}")
    print(f"MODEL: {resolved_model}")
    print(f"PROMPT LENGTH: {prompt_length}")
    print(f"PROMPT TOKENS (approx): {prompt_tokens}")
    print("----------------------------------------")
    print("PROMPT PREVIEW (first 500 chars):")
    print(prompt_preview)
    print("----------------------------------------")

    if _LLM_DEBUG_STATE["prompt_hash_counts"][prompt_hash] > 1:
        print("⚠️ WARNING: DUPLICATE PROMPT DETECTED")

    print("Called from:")
    if context["stack_lines"]:
        for line in context["stack_lines"]:
            print(line)
    else:
        print("- <call stack unavailable>")

    return {
        "call_number": call_number,
        "prompt_hash": prompt_hash,
        "prompt_preview": prompt_preview,
        "prompt_length": prompt_length,
        "prompt_tokens": prompt_tokens,
        "function": context["function"],
        "file": context["file"],
        "request": request,
    }


def _debug_llm_end(debug_context: Optional[dict[str, Any]], usage: Optional[dict[str, Any]] = None, error: Optional[Exception] = None) -> None:
    if not debug_context:
        return
    if debug_context.get("summary_printed"):
        return
    debug_context["summary_printed"] = True

    completion_tokens = _coerce_token_int(_safe_usage_value(usage, "completion_tokens"))
    prompt_tokens = _coerce_token_int(_safe_usage_value(usage, "prompt_tokens")) or _coerce_token_int(debug_context.get("prompt_tokens"))
    total_tokens = prompt_tokens + completion_tokens

    _LLM_DEBUG_STATE["total_completion_tokens"] += completion_tokens

    print("----------------------------------------")
    print(f"OUTPUT TOKENS: {completion_tokens}")
    print(f"TOTAL TOKENS THIS CALL: {total_tokens}")
    if error is not None:
        print(f"CALL STATUS: ERROR -> {error}")


def request_llm(
    messages: Any,
    *,
    request=None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    image_path: Optional[str] = None,
) -> str:
    resolved_provider = persist_selected_llm_provider(request=request, provider=provider)
    resolved_model = resolve_llm_model(resolved_provider, model=model)
    normalized_messages = _coerce_messages(messages)
    system_prompt, user_prompt = _flatten_messages(normalized_messages)
    encoded_image = _encode_image_for_llm(image_path)
    debug_context = (
        _debug_llm_start(
            resolved_provider,
            resolved_model,
            system_prompt,
            user_prompt,
            request=request,
        )
        if DEBUG_LLM else None
    )

    total_attempts = len(RATE_LIMIT_RETRY_DELAYS_SECONDS) + 1
    for attempt_index in range(total_attempts):
        try:
            if resolved_provider == "anthropic":
                import anthropic

                api_key = _resolve_provider_api_key("anthropic")
                if not api_key:
                    raise LLMProviderError("Anthropic API key not found.", provider=resolved_provider)

                client = anthropic.Anthropic(
                    api_key=api_key,
                    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                    max_retries=1,
                )
                user_content = []
                if encoded_image:
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": encoded_image["mime_type"],
                            "data": encoded_image["data"],
                        },
                    })
                user_content.append({"type": "text", "text": user_prompt})
                response = client.messages.create(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                )
                usage = _extract_token_usage(resolved_provider, response)
                record_llm_usage(
                    request=request,
                    provider=resolved_provider,
                    usage=usage,
                )
                if DEBUG_LLM:
                    _debug_llm_end(
                        debug_context,
                        usage=usage,
                    )
                response_text = response.content[0].text.strip()  # type: ignore[index]
                if is_credit_exhausted_error(response_text):
                    raise LLMProviderError(
                        build_credit_exhausted_message(resolved_provider),
                        provider=resolved_provider,
                        credit_exhausted=True,
                    )
                if is_rate_limit_error(response_text):
                    raise LLMProviderError(
                        build_rate_limit_message(resolved_provider),
                        provider=resolved_provider,
                        credit_exhausted=False,
                    )
                return response_text

            if resolved_provider == "gemini":
                import google.generativeai as genai

                api_key = _resolve_provider_api_key("gemini")
                if not api_key:
                    raise LLMProviderError("Gemini API key not found.", provider=resolved_provider)

                genai.configure(api_key=api_key)  # type: ignore[attr-defined]
                prompt_parts: list[Any] = []
                combined_prompt = "\n\n".join(part for part in [system_prompt, user_prompt] if part)
                prompt_parts.append(combined_prompt)
                if encoded_image:
                    prompt_parts.append({
                        "mime_type": encoded_image["mime_type"],
                        "data": encoded_image["data"],
                    })
                model_client = genai.GenerativeModel(resolved_model)  # type: ignore[attr-defined]
                response = model_client.generate_content(prompt_parts)
                usage = _extract_token_usage(resolved_provider, response)
                record_llm_usage(
                    request=request,
                    provider=resolved_provider,
                    usage=usage,
                )
                if DEBUG_LLM:
                    _debug_llm_end(
                        debug_context,
                        usage=usage,
                    )
                response_text = str(getattr(response, "text", "") or "").strip()
                if is_credit_exhausted_error(response_text):
                    raise LLMProviderError(
                        build_credit_exhausted_message(resolved_provider),
                        provider=resolved_provider,
                        credit_exhausted=True,
                    )
                if is_rate_limit_error(response_text):
                    raise LLMProviderError(
                        build_rate_limit_message(resolved_provider),
                        provider=resolved_provider,
                        credit_exhausted=False,
                    )
                return response_text

            if resolved_provider == "openai":
                api_key = _resolve_provider_api_key("openai")
                if not api_key:
                    raise LLMProviderError("OpenAI API key not found.", provider=resolved_provider)

                client = OpenAI(api_key=api_key, timeout=LLM_REQUEST_TIMEOUT_SECONDS, max_retries=2)
                user_content = []
                if encoded_image:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{encoded_image['mime_type']};base64,{encoded_image['data']}"
                        },
                    })
                user_content.append({"type": "text", "text": user_prompt})
                response = client.chat.completions.create(
                    model=resolved_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                usage = _extract_token_usage(resolved_provider, response)
                record_llm_usage(
                    request=request,
                    provider=resolved_provider,
                    usage=usage,
                )
                if DEBUG_LLM:
                    _debug_llm_end(
                        debug_context,
                        usage=usage,
                    )
                response_text = str(response.choices[0].message.content or "").strip()
                if is_credit_exhausted_error(response_text):
                    raise LLMProviderError(
                        build_credit_exhausted_message(resolved_provider),
                        provider=resolved_provider,
                        credit_exhausted=True,
                    )
                if is_rate_limit_error(response_text):
                    raise LLMProviderError(
                        build_rate_limit_message(resolved_provider),
                        provider=resolved_provider,
                        credit_exhausted=False,
                    )
                return response_text

            if resolved_provider == "groq":
                api_key = _resolve_provider_api_key("groq")
                if not api_key:
                    raise LLMProviderError("Groq API key not found.", provider=resolved_provider)

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": resolved_model,
                    "messages": normalized_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                response = requests.post(
                    CHAT_COMPLETIONS_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code != 200:
                    response_text = response.text
                    if is_credit_exhausted_error(response_text):
                        raise LLMProviderError(
                            build_credit_exhausted_message(resolved_provider),
                            provider=resolved_provider,
                            credit_exhausted=True,
                        )
                    if is_rate_limit_error(response_text):
                        raise LLMProviderError(
                            build_rate_limit_message(resolved_provider),
                            provider=resolved_provider,
                            credit_exhausted=False,
                        )
                    raise LLMProviderError(
                        f"[Groq Error] {response.status_code}: {response_text}",
                        provider=resolved_provider,
                    )

                data = response.json()
                usage = _extract_token_usage(resolved_provider, data)
                record_llm_usage(
                    request=request,
                    provider=resolved_provider,
                    usage=usage,
                )
                if DEBUG_LLM:
                    _debug_llm_end(
                        debug_context,
                        usage=usage,
                    )
                response_text = str(data["choices"][0]["message"]["content"]).strip()
                if is_credit_exhausted_error(response_text):
                    raise LLMProviderError(
                        build_credit_exhausted_message(resolved_provider),
                        provider=resolved_provider,
                        credit_exhausted=True,
                    )
                if is_rate_limit_error(response_text):
                    raise LLMProviderError(
                        build_rate_limit_message(resolved_provider),
                        provider=resolved_provider,
                        credit_exhausted=False,
                    )
                return response_text

            raise LLMProviderError(
                f"Unsupported LLM provider configured: {resolved_provider}",
                provider=resolved_provider,
            )

        except LLMProviderError as exc:
            should_retry = (
                not exc.credit_exhausted
                and is_rate_limit_error(exc)
                and attempt_index < len(RATE_LIMIT_RETRY_DELAYS_SECONDS)
            )
            if should_retry:
                time.sleep(RATE_LIMIT_RETRY_DELAYS_SECONDS[attempt_index])
                continue
            if DEBUG_LLM:
                _debug_llm_end(debug_context, error=exc)
            raise
        except Exception as exc:
            should_retry = (
                is_rate_limit_error(exc)
                and attempt_index < len(RATE_LIMIT_RETRY_DELAYS_SECONDS)
            )
            if should_retry:
                time.sleep(RATE_LIMIT_RETRY_DELAYS_SECONDS[attempt_index])
                continue
            if DEBUG_LLM:
                _debug_llm_end(debug_context, error=exc)
            raise LLMProviderError(
                build_credit_exhausted_message(resolved_provider)
                if is_credit_exhausted_error(exc)
                else build_rate_limit_message(resolved_provider)
                if is_rate_limit_error(exc)
                else str(exc),
                provider=resolved_provider,
                credit_exhausted=is_credit_exhausted_error(exc),
            ) from exc

    raise LLMProviderError(
        build_rate_limit_message(resolved_provider),
        provider=resolved_provider,
        credit_exhausted=False,
    )
