"""
llm.py — hosted LLM backend for YourWords.

Replaces the original local llama-cpp-python model with a hosted, OpenAI-compatible
inference API (Groq, Together, Fireworks, ...). The rest of the pipeline is
model-agnostic: it only ever calls generate_response(messages), so pointing at a
different provider is a config change, not a code change.

Configure via environment variables (see .env.example):
    LLM_API_KEY    your provider API key
    LLM_BASE_URL   provider's OpenAI-compatible endpoint
                   (default: Groq — https://api.groq.com/openai/v1)
    LLM_MODEL      model id (default: openai/gpt-oss-20b)

Note on reasoning models: Groq's gpt-oss (and qwen3) models are reasoning models.
Their hidden chain-of-thought is billed against the completion budget, so with a
small budget and the default "medium" effort they can spend the entire budget
thinking and return EMPTY content (finish_reason="length"). We therefore request
low reasoning effort for those models and give the completion room to breathe.
"""
from __future__ import annotations

import os
from openai import OpenAI

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "openai/gpt-oss-20b"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Create the API client once, on first use."""
    global _client
    if _client is None:
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or set it in your deployment's Secrets."
            )
        base_url = os.environ.get("LLM_BASE_URL", _DEFAULT_BASE_URL)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _is_reasoning_model(model: str) -> bool:
    """Groq models whose hidden reasoning is billed against the token budget."""
    m = model.lower()
    return "gpt-oss" in m or "qwen3" in m


def generate_response(
    messages: list[dict],
    *,
    max_completion_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """
    Drop-in replacement for the original local generate_response().

    Takes chat-style messages ([{role, content}, ...]) and returns the assistant's
    text. temperature=0 keeps output deterministic, which matters for a grammar tool.
    """
    client = _get_client()
    model = os.environ.get("LLM_MODEL", _DEFAULT_MODEL)

    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
    # Only reasoning models accept reasoning_effort; sent via extra_body so it works
    # regardless of the installed openai SDK version. "low" keeps reasoning from
    # eating the whole budget and leaving content empty.
    if _is_reasoning_model(model):
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()

    if not content:
        # Turn a silent empty string into a clear error, so the pipeline never
        # mistakes "the model returned nothing" for "no issues found".
        raise RuntimeError(
            f"The model returned empty content (finish_reason={choice.finish_reason!r}). "
            "For a gpt-oss / reasoning model this usually means reasoning consumed the "
            "token budget — raise max_completion_tokens or lower reasoning_effort."
        )
    return content