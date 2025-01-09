# datasette-llm-usage

[![PyPI](https://img.shields.io/pypi/v/datasette-llm-usage.svg)](https://pypi.org/project/datasette-llm-usage/)
[![Changelog](https://img.shields.io/github/v/release/datasette/datasette-llm-usage?include_prereleases&label=changelog)](https://github.com/datasette/datasette-llm-usage/releases)
[![Tests](https://github.com/datasette/datasette-llm-usage/actions/workflows/test.yml/badge.svg)](https://github.com/datasette/datasette-llm-usage/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/datasette/datasette-llm-usage/blob/main/LICENSE)

Track usage of LLM tokens in a SQLite table

This is a **very early alpha**.

## Installation

Install this plugin in the same environment as Datasette.
```bash
datasette install datasette-llm-usage
```
## Usage

This plugin adds functionality to track and manage LLM token usage in Datasette. It creates several tables.

- `_llm_usage`: Records usage of LLM models, one row per prompt executed
- `_llm_usage_allowance`: Tracks the remaining credits for each provider
- `_llm_usage_provider`: LLM providers such as OpenAI and Anthropic
- `_llm_usage_model`: LLM models such as `gpt-4o` and `claude-3.5-sonnet`

### Configuration

By default the tables are created in the internal database passed to Datasette using `--internal internal.db`. You can change that by setting the following in your Datasette plugin configuration:

```yaml
plugins:
  datasette-llm-usage:
    database: your_database_name
```
The internal database is recommended so users cannot reset their token allowances by modifying the tables directly.

You'll need to configure the models that should be made available by this plugin, in addition to installing the relevant LLM plugins (such as `llm-gemini`).

That configuration looks like this:
```yaml
plugins:
  datasette-llm-usage:
    allowances:
    - provider: gemini
      daily_reset_amount: 100000
    models:
    - provider: gemini
      model_id: gemini-1.5-flash
      tiers:
      - max_tokens: 128000
        input_cost: 7
        output_cost: 30
      - max_tokens: null
        input_cost: 15
        output_cost: 60
```
This defines a single model, `gemini-1.5-flash`, from a single provider, `gemini`.

The `daily_reset_amount` is the number of credits that will be made available for the specified provider each day.

In tis example prompts below 128,000 input tokens are changed at 7 credits per input token and 30 credits per output token, and prompts above 128,000 input tokens are charged at 15 credits per input token and 60 credits per output token.

A credit is worth 10,000th of a cent. The easier way to think about these values is in terms of cents-per-million tokens. For Gemini 1.5 Flash those values are:

- $0.07/million input tokens
- $0.3/million output tokens
- $0.15/million input tokens above 128,000 tokens
- $0.6/million output tokens above 128,000 tokens

Which translates to the 7/30 and 15/60 credit values shown above.

### Using the LLM wrapper

The plugin provides an `LLM` class that wraps the `llm` library to track token usage:

```python
from datasette_llm_usage import LLM, TokensExhausted

llm = LLM(datasette)

# Get available models
models = await llm.get_async_models()

# Get a specific model
model = await llm.get_async_model("gpt-4o-mini", purpose="my_purpose")

# Use the model
try:
    response = await model.prompt("Your prompt here")
    text = await response.text()
except TokensExhausted:
    print("Tokens exhausted")
```
If the number of available tokens for the model is exceeded, the `model.prompt()` method will raise a `TokensExhausted` exception

Token usage will be automatically recorded in the tables documented above.

### Built-in endpoint

The plugin provides a simple demo endpoint at `/-/llm-usage-simple-prompt` that requires authentication and allows a single prompt to be executed against a selected model.

### Supported Models and Pricing

The plugin includes pricing information for various models including:

- Gemini models (1.5-flash, 1.5-pro)
- Claude models (3.5-sonnet, 3-opus, 3-haiku)
- GPT models (gpt-4o, gpt-4o-mini, o1-preview, o1-mini)

Different models have different input and output token costs.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd datasette-llm-usage
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
