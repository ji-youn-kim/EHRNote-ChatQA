"""
Shared LLM client for Azure OpenAI (GPT) and Google Gemini.

All credentials and provider switches are read from environment variables.
Set them in your shell before running any script that calls an LLM, for
example::

    # Azure OpenAI (GPT)
    export AZURE_API_KEY=...
    export AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
    export AZURE_API_VERSION=2024-12-01-preview

    # Gemini via Vertex AI (requires `gcloud auth application-default login`)
    export GOOGLE_CLOUD_PROJECT=<gcp-project-id>
    export GOOGLE_CLOUD_LOCATION=us-central1

Gemini is always called through Vertex AI, which is the HIPAA-compliant
deployment required for MIMIC-derived content. The Gemini Developer API
(``GOOGLE_API_KEY``) is deliberately not supported.
"""

import os


def _require_env(name: str) -> str:
    """Read a required environment variable or raise with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name!r} is not set. "
            "Export it in your shell before running this script "
            "(see the docstring at the top of core/llm_client.py)."
        )
    return value


def _get_azure_client():
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_version=_require_env("AZURE_API_VERSION"),
        azure_endpoint=_require_env("AZURE_ENDPOINT"),
        api_key=_require_env("AZURE_API_KEY"),
    )


def _get_gemini_vertex_client():
    """
    Return a google-genai Client configured for Vertex AI.
    Uses Application Default Credentials (ADC) for authentication
    (set up via ``gcloud auth application-default login``).
    """
    from google import genai

    # Tell google-genai to use Vertex AI and where to dispatch requests.
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _require_env("GOOGLE_CLOUD_PROJECT"))
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", _require_env("GOOGLE_CLOUD_LOCATION"))

    return genai.Client()


def chat_completion(
    messages,
    model,
    temperature=1,
    max_completion_tokens=None,
):
    """
    Single function to call either Azure OpenAI (GPT) or Google Gemini.
    Automatically detects provider from model name.

    Args:
        messages: List of dicts with "role" ("system"|"user"|"assistant") and "content".
        model: Model name. If model name contains "gemini", uses Gemini provider, otherwise uses GPT.
               Examples: "gpt-5.4", "gpt-5.4-mini" -> GPT; "gemini-3-flash-preview" -> Gemini
        temperature: Sampling temperature.
        max_completion_tokens: Optional max tokens to generate.

    Returns:
        Assistant reply text.
    """
    model_lower = model.lower()
    if "gemini" in model_lower:
        provider = "gemini"
    else:
        provider = "gpt"

    print(f"Using {provider} provider (detected from model: {model})")
    if provider == "gpt":
        return _chat_completion_azure(
            messages, model, temperature, max_completion_tokens
        )
    if provider == "gemini":
        return _chat_completion_gemini(
            messages, model, temperature, max_completion_tokens
        )
    raise ValueError(f"Unknown provider: {provider}. Use a model name containing 'gemini' or a GPT model name.")


def _chat_completion_azure(messages, model, temperature, max_completion_tokens):
    client = _get_azure_client()
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _chat_completion_gemini(messages, model, temperature, max_completion_tokens):
    """Chat completion for Gemini on Vertex AI via google-genai."""
    from google.genai import types as genai_types

    client = _get_gemini_vertex_client()

    # Collect system messages and convert to Vertex contents
    system_text = None
    contents: list[genai_types.Content] = []

    for m in messages:
        role = (m.get("role") or "user").lower()
        # Preserve the caller's exact content (DO NOT strip trailing
        # whitespace). Gemini's reasoning depth is sensitive to prompt-end
        # whitespace: dropping a single trailing newline can cut the thinking
        # budget short and truncate the visible response. Use .strip() only as
        # the empty-check, never as the value passed to the API.
        content = m.get("content") or ""
        if not content.strip():
            continue

        if role == "system":
            if system_text is None:
                system_text = content
            else:
                system_text += "\n\n" + content
        elif role == "user":
            # Prepend system text to the first user message (Vertex doesn't support system role).
            if system_text is not None:
                content = f"{system_text}\n\n{content}"
                system_text = None
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=content)],
                )
            )
        elif role == "assistant":
            contents.append(
                genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=content)],
                )
            )

    # If we had only system messages, still send them as a single user message
    if not contents and system_text:
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=system_text)],
            )
        )

    if not contents:
        raise ValueError(
            "At least one non-empty user/assistant message is required for Gemini."
        )

    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_completion_tokens,
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return response.text or ""
