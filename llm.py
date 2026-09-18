"""
llm.py — hosted LLM backend for YourWords.

Replaces the original local llama-cpp-python model with a hosted, OpenAI-compatible
inference API (Groq, Together, Fireworks, ...). The rest of the pipeline is
model-agnostic: it only ever calls generate_response(messages), so pointing at a
different provider is a config change, not a code change.

Why this swap: the local Q8 LLaMA 3.1 8B needs an 8 GB download and a GPU, which
rules out free CPU hosting. A hosted Llama endpoint keeps the exact same model
family, runs fast, and deploys free (Streamlit Community Cloud, etc.).

Configure via environment variables (see .env.example):
    LLM_API_KEY    your provider API key
    LLM_BASE_URL   provider's OpenAI-compatible endpoint
                   (default: Groq — https://api.groq.com/openai/v1)
    LLM_MODEL      model id (default: llama-3.1-8b-instant)
"""
from __future__ import annotations

import os
from openai import OpenAI

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.1-8b-instant"

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


def generate_response(
    messages: list[dict],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Drop-in replacement for the original local generate_response().

    Takes chat-style messages ([{role, content}, ...]) and returns the assistant's
    text. temperature=0 keeps output deterministic, which matters for a grammar
    tool. (The old repeat_penalty=2.0 is intentionally gone — it was aggressive
    enough to degrade phrasing; temperature=0 handles repeatability cleanly.)
    """
    client = _get_client()
    model = os.environ.get("LLM_MODEL", _DEFAULT_MODEL)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
