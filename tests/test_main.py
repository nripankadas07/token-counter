"""Tests for token-counter core functionality."""

import pytest

from token_counter.main import (
    TokenCount,
    TokenCounter,
    count_tokens,
    estimate_cost,
    SUPPORTED_MODELS,
    MODEL_REGISTRY,
)


@pytest.fixture
def counter() -> TokenCounter:
    return TokenCounter()


class TestTokenCounter:
    """Tests for the TokenCounter class."""

    def test_count_returns_token_count(self, counter: TokenCounter) -> None:
        result = counter.count("Hello, world!", "gpt-4o")
        assert isinstance(result, TokenCount)
        assert result.token_count > 0
        assert result.model == "gpt-4o"

    def test_count_known_text(self, counter: TokenCounter) -> None:
        # "Hello" is a single token in most BPE encodings
        result = counter.count("Hello", "gpt-4o")
        assert result.token_count >= 1

    def test_count_empty_string(self, counter: TokenCounter) -> None:
        result = counter.count("", "gpt-4o")
        assert result.token_count == 0
        assert result.text_length == 0

    def test_count_unsupported_model(self, counter: TokenCounter) -> None:
        with pytest.raises(ValueError, match="Unsupported model"):
            counter.count("test", "nonexistent-model-xyz")

    def test_count_returns_correct_metadata(self, counter: TokenCounter) -> None:
        result = counter.count("Test text", "claude-3.5-sonnet")
        assert result.provider == "anthropic"
        assert result.encoding_name == "cl100k_base"
        assert result.model == "claude-3.5-sonnet"

    def test_count_cost_estimates(self, counter: TokenCounter) -> None:
        result = counter.count("Some text to count", "gpt-4o")
        assert result.input_cost is not None
        assert result.output_cost is not None
        assert result.input_cost >= 0
        assert result.output_cost >= 0

    def test_count_free_model(self, counter: TokenCounter) -> None:
        result = counter.count("Open source text", "llama-3-70b")
        assert result.input_cost == 0.0
        assert result.output_cost == 0.0

    def test_count_multi_default(self, counter: TokenCounter) -> None:
        results = counter.count_multi("Hello, world!")
        assert len(results) == len(SUPPORTED_MODELS)
        for r in results:
            assert r.token_count > 0

    def test_count_multi_subset(self, counter: TokenCounter) -> None:
        models = ["gpt-4o", "claude-3.5-sonnet"]
        results = counter.count_multi("Hello", models)
        assert len(results) == 2
        assert results[0].model == "gpt-4o"
        assert results[1].model == "claude-3.5-sonnet"

    def test_consistent_counts(self, counter: TokenCounter) -> None:
        """Same text + same model should always give the same count."""
        r1 = counter.count("test text here", "gpt-4o")
        r2 = counter.count("test text here", "gpt-4o")
        assert r1.token_count == r2.token_count

    def test_whitespace_estimate(self) -> None:
        estimate = TokenCounter.estimate_whitespace("Hello world this is a test")
        # 6 words * 1.3 â 7-8
        assert 5 <= estimate <= 12

    def test_whitespace_estimate_empty(self) -> None:
        assert TokenCounter.estimate_whitespace("") == 0

    def test_long_text(self, counter: TokenCounter) -> None:
        text = "word " * 1000  # 1000 words
        result = counter.count(text, "gpt-4o")
        assert result.token_count > 500  # BPE usually fewer tokens than words
        assert result.text_length == 5000


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_count_tokens_default(self) -> None:
        count = count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_specific_model(self) -> None:
        count = count_tokens("Hello", "gpt-3.5-turbo")
        assert count >= 1

    def test_estimate_cost_input(self) -> None:
        cost = estimate_cost("Some text", "gpt-4o", "input")
        assert isinstance(cost, float)
        assert cost >= 0

    def test_estimate_cost_output(self) -> None:
        cost = estimate_cost("Some text", "gpt-4o", "output")
        assert isinstance(cost, float)
        assert cost >= 0

    def test_estimate_cost_free_model(self) -> None:
        cost = estimate_cost("text", "llama-3-70b", "input")
        assert cost == 0.0


class TestModelRegistry:
    """Tests for the model registry."""

    def test_all_models_have_required_keys(self) -> None:
        required = {"encoding", "input_cost_per_1k", "output_cost_per_1k", "provider"}
        for model, info in MODEL_REGISTRY.items():
            assert required.issubset(info.keys()), f"{model} missing keys"

    def test_supported_models_sorted(self) -> None:
        assert SUPPORTED_MODELS == sorted(SUPPORTED_MODELS)

    def test_supported_models_matches_registry(self) -> None:
        assert set(SUPPORTED_MODELS) == set(MODEL_REGISTRY.keys())
