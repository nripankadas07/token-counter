# token-counter

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()

Token counting CLI and library for multiple LLM tokenizers. Count tokens and estimate API costs across OpenAI, Anthropic, Meta, and Mistral models from a single interface.

## Why?

Every LLM provider uses different tokenization. Before sending a prompt, you need to know how many tokens it contains â for cost estimation, context window management, and prompt optimization. `token-counter` gives you accurate counts for 10+ models in one command.

## Installation

```bash
pip install token-counter
```

Or install from source:

```bash
git clone https://github.com/nripankadas07/token-counter.git
cd token-counter
pip install -e .
```

## Quick Start

### CLI Usage

Count tokens across all supported models:

```bash
token-counter count "Your prompt text goes here"
```

Count for specific models:

```bash
token-counter count -m gpt-4o -m claude-3.5-sonnet "Hello, world!"
```

Count tokens in a file:

```bash
token-counter count --file prompt.txt
```

Pipe from stdin:

```bash
cat prompt.txt | token-counter count
```

List all supported models:

```bash
token-counter models
```

### Python API

```python
from token_counter import TokenCounter, count_tokens, estimate_cost

# Quick one-liner
tokens = count_tokens("Hello, world!", model="gpt-4o")
print(f"Tokens: {tokens}")

# Detailed counting
counter = TokenCounter()
result = counter.count("Your prompt here", "claude-3.5-sonnet")
print(f"Tokens: {result.token_count}")
print(f"Input cost: ${result.input_cost:.6f}")
print(f"Output cost: ${result.output_cost:.6f}")

# Compare across models
results = counter.count_multi("Your prompt here", ["gpt-4o", "gpt-4", "claude-3.5-sonnet"])
for r in results:
    print(f"{r.model}: {r.token_count} tokens (${r.input_cost:.6f})")

# Quick whitespace estimate (no tokenizer needed)
estimate = TokenCounter.estimate_whitespace("Hello world")
```

## Supported Models

| Model | Provider | Encoding | Input $/1K | Output $/1K |
|-------|----------|----------|------------|-------------|
| gpt-4o | OpenAI | o200k_base | $0.005 | $0.015 |
| gpt-4o-mini | OpenAI | o200k_base | $0.00015 | $0.0006 |
| gpt-4-turbo | OpenAI | cl100k_base | $0.01 | $0.03 |
| gpt-4 | OpenAI | cl100k_base | $0.03 | $0.06 |
| gpt-3.5-turbo | OpenAI | cl100k_base | $0.0005 | $0.0015 |
| claude-3.5-sonnet | Anthropic | cl100k_base* | $0.003 | $0.015 |
| claude-3-opus | Anthropic | cl100k_base* | $0.015 | $0.075 |
| claude-3-haiku | Anthropic | cl100k_base* | $0.00025 | $0.00125 |
| llama-3-70b | Meta | cl100k_base* | free | free |
| mistral-large | Mistral | cl100k_base* | $0.004 | $0.012 |

*Non-OpenAI models use cl100k_base as a close approximation. Actual token counts may differ by ~5%.

## API Reference

### `TokenCounter`

The main class for counting tokens.

- **`count(text, model)`** â Count tokens for a single model. Returns `TokenCount`.
- **`count_multi(text, models=None)`** â Count across multiple models. Returns list of `TokenCount`.
- **`estimate_whitespace(text)`** â Quick word-based estimate (~1.3x word count). Static method.

### `TokenCount` (dataclass)

- `text_length: int` â Character count of input text
- `token_count: int` â Number of tokens
- `model: str` â Model name used
- `encoding_name: str` â Tiktoken encoding name
- `provider: str` â Model provider
- `input_cost: float | None` â Estimated input cost in USD
- `output_cost: float | None` â Estimated output cost in USD

### Convenience Functions

- **`count_tokens(text, model="gpt-4o")`** â Returns token count as `int`.
- **`estimate_cost(text, model="gpt-4o", direction="input")`** â Returns cost as `float`.

## Architecture

```
token_counter/
âââ main.py      # Core engine: TokenCounter, MODEL_REGISTRY, cost estimation
âââ cli.py       # Click CLI with Rich table output
```

The design separates the tokenization engine from the CLI layer. `TokenCounter` caches tiktoken encodings for performance when counting across multiple inputs. The model registry is a simple dict that maps model names to their encoding and pricing, making it trivial to add new models.

## License

MIT License â Copyright 2024 Nripanka Das
