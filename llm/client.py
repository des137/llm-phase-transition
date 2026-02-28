"""
Multi-provider LLM client with dry-run / stub support.

Supported providers (set via model string prefix):
  openai:     OpenAI Chat Completions API      (OPENAI_API_KEY)
  anthropic:  Anthropic Messages API           (ANTHROPIC_API_KEY)
  together:   Together AI API                  (TOGETHER_API_KEY)
  groq:       Groq API                         (GROQ_API_KEY)
  mistral:    Mistral AI API                   (MISTRAL_API_KEY)
  deepseek:   DeepSeek API                     (DEEPSEEK_API_KEY)
  google:     Google Gemini API                (GOOGLE_API_KEY)  [uses google-genai SDK]
  openrouter: OpenRouter API                   (OPENROUTER_API_KEY)
  grok:       xAI Grok API                     (GROK_API_KEY)
  local:      Local vllm / Ollama server       (LOCAL_LLM_BASE_URL, LOCAL_LLM_API_KEY)

Model string format:  "<provider>:<model_id>"
  e.g.  "openai:gpt-4o"
        "anthropic:claude-sonnet-4-5-20250929"
        "groq:llama-3.3-70b-versatile"
        "mistral:mistral-small-latest"
        "deepseek:deepseek-chat"
        "google:gemini-2.0-flash"
        "openrouter:meta-llama/llama-3.3-70b-instruct"
        "grok:grok-2-latest"
        "together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        "local:meta-llama/Meta-Llama-3-8B-Instruct"

If no prefix is given, "openai:" is assumed.

API keys are loaded from environment variables (populated from .env by config.py).

OpenAI-compatible providers (use _call_openai_compat):
  together, groq, mistral, deepseek, openrouter, grok, local

Native SDK providers:
  openai    → openai SDK
  anthropic → anthropic SDK
  google    → google-genai SDK
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Provider base URLs for OpenAI-compatible APIs
# ---------------------------------------------------------------------------

_OPENAI_COMPAT_BASE_URLS: dict[str, str] = {
    "together":   "https://api.together.xyz/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "mistral":    "https://api.mistral.ai/v1",
    "deepseek":   "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "grok":       "https://api.x.ai/v1",
}

_OPENAI_COMPAT_ENV_KEYS: dict[str, str] = {
    "together":   "TOGETHER_API_KEY",
    "groq":       "GROQ_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "grok":       "GROK_API_KEY",
}


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    model: str
    prompt: str
    response_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    dry_run: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Provider parsing
# ---------------------------------------------------------------------------

def _parse_model_string(model_str: str) -> tuple[str, str]:
    """
    Parse "provider:model_id" → (provider, model_id).
    If no colon, assume provider="openai".
    Special case: "google:gemini-2.0-flash" — the model_id may contain colons
    in some Google naming schemes, so we only split on the first colon.
    """
    if ":" in model_str:
        provider, model_id = model_str.split(":", 1)
        return provider.lower(), model_id
    return "openai", model_str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Parameters
    ----------
    model : str
        "<provider>:<model_id>" or just "<model_id>" (defaults to openai).
    api_key : str | None
        Override the API key (otherwise read from environment).
    dry_run : bool
        If True, skip real API calls and return deterministic mock responses.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens in the completion.
    timeout : float
        Request timeout in seconds.
    retry_attempts : int
        Number of retry attempts on transient errors.
    retry_delay : float
        Seconds to wait between retries (exponential backoff).
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        api_key: Optional[str] = None,
        dry_run: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 60.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.model_str = model
        self.provider, self.model_id = _parse_model_string(model)
        self.dry_run = dry_run
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._api_key = api_key  # explicit override; otherwise resolved per-call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request and return an LLMResponse.
        """
        if self.dry_run:
            return self._mock_response(system_prompt, user_prompt)

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Route to the correct backend
        if self.provider == "openai":
            call_fn = self._call_openai
        elif self.provider == "anthropic":
            call_fn = self._call_anthropic
        elif self.provider == "google":
            call_fn = self._call_google
        elif self.provider in _OPENAI_COMPAT_BASE_URLS or self.provider == "local":
            call_fn = self._call_openai_compat
        else:
            return LLMResponse(
                model=self.model_str,
                prompt=user_prompt,
                response_text="",
                error=(
                    f"Unknown provider '{self.provider}'. "
                    f"Supported: openai, anthropic, google, together, groq, mistral, "
                    f"deepseek, openrouter, grok, local"
                ),
            )

        last_error: Optional[str] = None
        for attempt in range(self.retry_attempts):
            try:
                return call_fn(system_prompt, user_prompt, effective_max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        return LLMResponse(
            model=self.model_str,
            prompt=user_prompt,
            response_text="",
            error=last_error,
        )

    # ------------------------------------------------------------------
    # OpenAI (native SDK)
    # ------------------------------------------------------------------

    def _call_openai(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> LLMResponse:
        try:
            import openai as _openai  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pip install openai") from e

        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set. Add it to .env or export it.")

        client = _openai.OpenAI(api_key=key, timeout=self.timeout)
        return self._openai_compat_call(client, system_prompt, user_prompt, max_tokens)

    # ------------------------------------------------------------------
    # OpenAI-compatible (Together, Groq, Mistral, DeepSeek, OpenRouter,
    #                    Grok/xAI, local vllm/Ollama)
    # ------------------------------------------------------------------

    def _call_openai_compat(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> LLMResponse:
        try:
            import openai as _openai  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pip install openai") from e

        if self.provider == "local":
            key = self._api_key or os.environ.get("LOCAL_LLM_API_KEY", "EMPTY")
            base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
        else:
            env_key_name = _OPENAI_COMPAT_ENV_KEYS[self.provider]
            key = self._api_key or os.environ.get(env_key_name)
            if not key:
                raise ValueError(
                    f"{env_key_name} not set. Add it to .env or export it."
                )
            base_url = _OPENAI_COMPAT_BASE_URLS[self.provider]

        # OpenRouter requires extra headers for attribution
        extra_headers: dict[str, str] = {}
        if self.provider == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://github.com/phase-transition-llm",
                "X-Title": "LLM Phase Transition Study",
            }

        client = _openai.OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=self.timeout,
            default_headers=extra_headers if extra_headers else None,
        )
        return self._openai_compat_call(client, system_prompt, user_prompt, max_tokens)

    def _openai_compat_call(
        self, client: object, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        t0 = time.perf_counter()
        completion = client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - t0
        return LLMResponse(
            model=self.model_str,
            prompt=user_prompt,
            response_text=completion.choices[0].message.content or "",
            input_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            output_tokens=completion.usage.completion_tokens if completion.usage else 0,
            latency_seconds=latency,
        )

    # ------------------------------------------------------------------
    # Anthropic (native SDK)
    # ------------------------------------------------------------------

    def _call_anthropic(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> LLMResponse:
        try:
            import anthropic as _anthropic  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

        client = _anthropic.Anthropic(api_key=key)
        t0 = time.perf_counter()
        message = client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        latency = time.perf_counter() - t0
        response_text = message.content[0].text if message.content else ""
        return LLMResponse(
            model=self.model_str,
            prompt=user_prompt,
            response_text=response_text,
            input_tokens=message.usage.input_tokens if message.usage else 0,
            output_tokens=message.usage.output_tokens if message.usage else 0,
            latency_seconds=latency,
        )

    # ------------------------------------------------------------------
    # Google Gemini (google-genai SDK — replaces deprecated google-generativeai)
    # ------------------------------------------------------------------

    def _call_google(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> LLMResponse:
        try:
            from google import genai  # type: ignore[import]
            from google.genai import types as genai_types  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pip install google-genai") from e

        key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set. Add it to .env or export it.")

        client = genai.Client(api_key=key)

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=max_tokens,
        )

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=self.model_id,
            contents=user_prompt,
            config=config,
        )
        latency = time.perf_counter() - t0

        response_text = response.text or ""

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return LLMResponse(
            model=self.model_str,
            prompt=user_prompt,
            response_text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
        )

    # ------------------------------------------------------------------
    # Dry-run mock
    # ------------------------------------------------------------------

    def _mock_response(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Return a deterministic mock response based on a hash of the prompts.
        """
        digest = hashlib.md5((system_prompt + user_prompt).encode()).hexdigest()[:8]
        mock_text = f"[DRY-RUN mock response {digest}] This is a placeholder answer."
        return LLMResponse(
            model=self.model_str,
            prompt=user_prompt,
            response_text=mock_text,
            input_tokens=len(user_prompt.split()),
            output_tokens=10,
            latency_seconds=0.001,
            dry_run=True,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self.provider!r}, model={self.model_id!r}, "
            f"dry_run={self.dry_run}, temperature={self.temperature})"
        )


# ---------------------------------------------------------------------------
# Recommended cheap/fast models per provider
# ---------------------------------------------------------------------------

RECOMMENDED_MODELS: dict[str, list[str]] = {
    "groq": [
        "groq:llama-3.3-70b-versatile",       # fast, free tier, high quality
        "groq:llama-3.1-8b-instant",           # ultra-fast, very cheap
        "groq:mixtral-8x7b-32768",             # good for long context
        "groq:gemma2-9b-it",                   # Google Gemma 2 9B
    ],
    "mistral": [
        "mistral:mistral-small-latest",        # cheap, good quality
        "mistral:open-mistral-nemo",           # open-weight, very cheap
        "mistral:mistral-medium-latest",       # mid-tier
    ],
    "deepseek": [
        "deepseek:deepseek-chat",              # DeepSeek V3, very cheap (~$0.27/M input)
        "deepseek:deepseek-reasoner",          # R1 reasoning model
    ],
    "google": [
        "google:gemini-2.0-flash",             # fast, cheap, high quality
        "google:gemini-1.5-flash",             # slightly older, very cheap
        "google:gemini-1.5-pro",               # high quality, more expensive
    ],
    "openrouter": [
        "openrouter:meta-llama/llama-3.3-70b-instruct",
        "openrouter:google/gemini-2.0-flash-001",
        "openrouter:mistralai/mistral-small-3.1-24b-instruct",
        "openrouter:deepseek/deepseek-chat-v3-0324",
    ],
    "grok": [
        "grok:grok-3-mini-beta",       # confirmed working
        "grok:grok-3-mini-fast-beta",  # faster variant
    ],
    "together": [
        "together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "together:mistralai/Mixtral-8x7B-Instruct-v0.1",
    ],
}

# Made with Bob
