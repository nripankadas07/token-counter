"""Core token counting engine with multi-model support.

Uses a built-in regex-based BPE approximation by default.
Optionally uses tiktoken for exact OpenAI token counts when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

# Try to import tiktoken for exact counting
try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


# Model registry: maps model names to their encoding and pricing
MODEL_REGISTRY: dict[str, dict] = {
    # OpenAI models
    "gpt-4o": {
        "encoding": "o200k_base",
        "input_cost_per_1k": 0.005,
        "output_cost_per_1k": 0.015,
        "provider": "openai",
    },
    "gpt-4o-mini": {
        "encoding": "o200k_base",
        "input_cost_per_1k": 0.00015,
        "output_cost_per_1k": 0.0006,
        "provider": "openai",
    },
    "gpt-4-turbo": {
        "encoding": "cl100k_base",
        "input_cost_per_1k": 0.01,
        "output_cost_per_1k": 0.03,
        "provider": "openai",
    },
    "gpt-4": {
        "encoding": "cl100k_base",
        "input_cost_per_1k": 0.03,
        "output_cost_per_1k": 0.06,
        "provider": "openai",
    },
    "gpt-3.5-turbo": {
        "encoding": "cl100k_base",
        "input_cost_per_1k": 0.0005,
        "output_cost_per_1k": 0.0015,
        "provider": "openai",
    },
    # Anthropic models
    "claude-3.5-sonnet": {
        "encoding": "cl100k_base",
        "input_cost_per_1k": 0.003,
        "output_cost_per_1k": 0.015,
        "provider": "anthropic",
    },
    "claude-3-opus": {
        "encoding": "cl100k_base",
        "input_cost_per_1k": 0.015,
        "output_cost_per_1k": 0.075,
        "provider": "anthropic",
    },
    "claude-3-haiku": {
        "encoding": "cl100k_base",
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.00125,
        "provider": "anthropic",
    },
    # Open-source
    "llama-3-70b": {
        "encoding": "cl100k_base",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "provider": "meta",
    },
    "mistral-large": {
        "encoding": "cl100k_base",
        "input_cost_per_1k": 0.004,
        "output_cost_per_1k": 0.012,
        "provider": "mistral",
    },
}

SUPPORTED_MODELS: list[str] = sorted(MODEL_REGISTRY.keys())

# Regex pattern for BPE-style token approximation
# Splits on contractions, words, numbers, punctuation, and whitespace
_TOKEN_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"|[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]+"
    r"|\d{1,3}"
    r"| ?[^\s\w]+"
    r"|\s+",
)

# Whitespace tokenizer for quick estimates
_WHITESPACE_RE = re.compile(r"\S+")


def _regex_tokenize(text: str) -> list[str]:
    """Tokenize text using a regex pattern that approximates BPE splitting."""
    return _TOKEN_PATTERN.findall(text)


@dataclass
class TokenCount:
    """Result of a token counting operation."""

    text_length: int
    token_count: int
    model: str
    encoding_name: str
    provider: str
    exact: bool = False
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None


@dataclass
class TokenCounter:
    """Multi-model token counter with cost estimation.

    Supports exact counting via tiktoken (when available) for OpenAI models,
    and a regex-based BPE approximation for all models as fallback.
    """

    use_tiktoken: bool = True
    _encoding_cache: dict[str, object] = field(
        default_factory=dict, repr=False
    )

    def _get_tiktoken_encoding(self, encoding_name: str) -> Optional[object]:
        """Try to get a tiktoken encoding. Returns None if unavailable."""
        if not self.use_tiktoken or not _HAS_TIKTOKEN:
            return None
        if encoding_name not in self._encoding_cache:
            try:
                self._encoding_cache[encoding_name] = tiktoken.get_encoding(
                    encoding_name
                )
            except Exception:
                return None
        return self._encoding_cache.get(encoding_name)

    def count(self, text: str, model: str) -> TokenCount:
        """Count tokens in text for a given model.

        Args:
            text: The input text to tokenize.
            model: Model name (e.g., 'gpt-4o', 'claude-3.5-sonnet').

        Returns:
            TokenCount with token count and metadata.

        Raises:
            ValueError: If the model is not supported.
        """
        if model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported: {', '.join(SUPPORTED_MODELS)}"
            )

        info = MODEL_REGISTRY[model]
        encoding_name = info["encoding"]

        # Try tiktoken first for exact count
        enc = self._get_tiktoken_encoding(encoding_name)
        if enc is not None:
            tokens = enc.encode(text)  # type: ignore
            token_count = len(tokens)
            exact = True
        else:
            # Fallback: regex-based approximation
            token_count = len(_regex_tokenize(text))
            exact = False

        input_cost = (token_count / 1000) * info["input_cost_per_1k"]
        output_cost = (token_count / 1000) * info["output_cost_per_1k"]

        return TokenCount(
            text_length=len(text),
            token_count=token_count,
            model=model,
            encoding_name=encoding_name,
            provider=info["provider"],
            exact=exact,
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
        )

    def count_multi(
        self, text: str, models: Optional[list[str]] = None
    ) -> list[TokenCount]:
        """Count tokens across multiple models at once.

        Args:
            text: The input text to tokenize.
            models: List of model names. Defaults to all supported models.

        Returns:
            List of TokenCount results, one per model.
        """
        if models is None:
            models = SUPPORTED_MODELS
        return [self.count(text, m) for m in models]

    @staticmethod
    def estimate_whitespace(text: str) -> int:
        """Quick whitespace-based token estimate (rough: ~1.3 tokens per word)."""
        words = len(_WHITESPACE_RE.findall(text))
        return int(words * 1.3)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Convenience function: count tokens for a single model.

    Args:
        text: Input text.
        model: Model name (default: gpt-4o).

    Returns:
        Integer token count.
    """
    counter = TokenCounter()
    return counter.count(text, model).token_count


def estimate_cost(
    text: str, model: str = "gpt-4o", direction: str = "input"
) -> float:
    """Estimate the API cost for a text.

    Args:
        text: Input text.
        model: Model name (default: gpt-4o).
        direction: 'input' or 'output'.

    Returns:
        Estimated cost in USD.
    """
    counter = TokenCounter()
    result = counter.count(text, model)
    if direction == "output":
        return result.output_cost or 0.0
    return result.input_cost or 0.0
